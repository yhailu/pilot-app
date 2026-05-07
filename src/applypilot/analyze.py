"""Post-run failure analyzer and auto-fixer for ApplyPilot.

Scans the database and logs from the last apply run, categorizes failures
by pattern, and applies automatic fixes:
  - Adds consistently-failing domains to blocked/manual_ats lists
  - Identifies MFA/email-verification failures for prompt improvement
  - Reports form-filling issues by site for debugging
  - Resets retryable failures after fixes are applied
"""

import json
import logging
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from applypilot.config import (
    APP_DIR, LOG_DIR, CONFIG_DIR, DB_PATH,
    load_sites_config,
)
from applypilot.database import get_connection, init_db

logger = logging.getLogger(__name__)
console = Console()

MFA_PATTERNS = [
    r"verification.?code", r"mfa", r"two.?factor", r"2fa",
    r"authenticat", r"one.?time.?(code|password|passcode)",
    r"otp", r"verify.?your.?(email|identity|account)",
    r"code.?sent", r"check.?your.?(email|inbox)",
    r"email.?code", r"sms.?code", r"security.?code", r"confirmation.?code",
]

SSO_PATTERNS = [
    r"sso", r"single.?sign.?on", r"oauth",
    r"google.?(sign|log|auth)", r"microsoft.?(sign|log|auth)",
    r"okta", r"saml",
]

CAPTCHA_PATTERNS = [
    r"captcha", r"cloudflare", r"turnstile",
    r"recaptcha", r"hcaptcha", r"blocked", r"bot.?detect",
]

FORM_PATTERNS = [
    r"stuck", r"validation", r"field.?(not|won|can)",
    r"dropdown", r"select", r"upload.?(fail|error)",
    r"form.?error", r"no_result_line", r"page_error", r"timeout",
]

EXPIRED_PATTERNS = [
    r"expired", r"closed", r"no.?longer.?accept", r"position.?filled",
]


def _match_category(error_text: str) -> str:
    if not error_text:
        return "unknown"
    text = error_text.lower()
    for pattern in MFA_PATTERNS:
        if re.search(pattern, text):
            return "mfa_email"
    for pattern in SSO_PATTERNS:
        if re.search(pattern, text):
            return "sso"
    for pattern in CAPTCHA_PATTERNS:
        if re.search(pattern, text):
            return "captcha"
    for pattern in EXPIRED_PATTERNS:
        if re.search(pattern, text):
            return "expired"
    for pattern in FORM_PATTERNS:
        if re.search(pattern, text):
            return "form_issue"
    if "login" in text:
        return "login_issue"
    if "not_eligible" in text:
        return "not_eligible"
    if "not_a_job" in text:
        return "not_a_job"
    return "other"


def _extract_domain(url: str) -> str:
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return "unknown"


def _scan_logs_for_mfa(log_dir: Path) -> list[dict]:
    findings = []
    mfa_log_patterns = [
        (r"(?:verification|confirmation|security)\s+code", "verification_code"),
        (r"code\s+(?:sent|was sent|has been sent)", "code_sent"),
        (r"check\s+(?:your\s+)?(?:email|inbox)", "check_email"),
        (r"(?:sign\s+in\s+with\s+a\s+code|email\s+code)", "email_code_login"),
        (r"(?:two.?factor|2fa|mfa|authenticator)", "mfa_required"),
        (r"(?:sms|text\s+message)\s+(?:code|verification)", "sms_code"),
        (r"don'?t\s+have\s+access.*email", "no_email_access"),
    ]
    log_files = sorted(log_dir.glob("worker-*.log")) + \
                sorted(log_dir.glob("claude_*.txt"), key=lambda p: p.stat().st_mtime)[-50:]
    for log_file in log_files:
        try:
            content = log_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        sections = re.split(r"={50,}", content)
        for section in sections:
            url_match = re.search(r"URL:\s*(https?://\S+)", section)
            if not url_match:
                continue
            url = url_match.group(1)
            for pattern, label in mfa_log_patterns:
                match = re.search(pattern, section, re.IGNORECASE)
                if match:
                    start = max(0, match.start() - 100)
                    end = min(len(section), match.end() + 100)
                    snippet = section[start:end].strip()
                    findings.append({
                        "url": url, "domain": _extract_domain(url),
                        "pattern": label, "snippet": snippet[:200],
                        "log_file": log_file.name,
                    })
                    break
    return findings


def get_failed_jobs(conn=None) -> list[dict]:
    if conn is None:
        conn = get_connection()
    rows = conn.execute("""
        SELECT url, title, site, application_url, apply_status,
               apply_error, apply_attempts, last_attempted_at,
               fit_score, apply_duration_ms
        FROM jobs
        WHERE apply_status IS NOT NULL
          AND apply_status != 'applied'
          AND apply_status != 'in_progress'
        ORDER BY last_attempted_at DESC
    """).fetchall()
    if rows:
        columns = rows[0].keys()
        return [dict(zip(columns, row)) for row in rows]
    return []


def get_applied_jobs(conn=None) -> list[dict]:
    if conn is None:
        conn = get_connection()
    rows = conn.execute("""
        SELECT url, title, site, application_url, applied_at,
               apply_duration_ms, fit_score
        FROM jobs
        WHERE apply_status = 'applied'
        ORDER BY applied_at DESC
    """).fetchall()
    if rows:
        columns = rows[0].keys()
        return [dict(zip(columns, row)) for row in rows]
    return []


def analyze_failures(conn=None) -> dict:
    if conn is None:
        conn = get_connection()
    failed = get_failed_jobs(conn)
    applied = get_applied_jobs(conn)
    by_category = Counter()
    by_status = Counter()
    domain_failures = defaultdict(lambda: {
        "count": 0, "categories": Counter(), "urls": [],
        "errors": [], "statuses": Counter()
    })
    for job in failed:
        error = job.get("apply_error") or job.get("apply_status") or "unknown"
        status = job.get("apply_status", "unknown")
        category = _match_category(error)
        if status == "captcha" and category != "captcha":
            category = "captcha"
        elif status == "login_issue" and category not in ("mfa_email", "sso"):
            category = "login_issue"
        elif status == "expired":
            category = "expired"
        by_category[category] += 1
        by_status[status] += 1
        apply_url = job.get("application_url") or job.get("url", "")
        domain = _extract_domain(apply_url)
        d = domain_failures[domain]
        d["count"] += 1
        d["categories"][category] += 1
        d["urls"].append(apply_url)
        d["errors"].append(error[:100])
        d["statuses"][status] += 1
    mfa_findings = _scan_logs_for_mfa(LOG_DIR)
    recommendations = _build_recommendations(
        domain_failures, by_category, mfa_findings, applied
    )
    return {
        "summary": {
            "total_failed": len(failed),
            "total_applied": len(applied),
            "by_category": dict(by_category.most_common()),
            "by_status": dict(by_status.most_common()),
        },
        "domains": dict(domain_failures),
        "mfa_findings": mfa_findings,
        "recommendations": recommendations,
    }


def _build_recommendations(domain_failures, by_category, mfa_findings, applied) -> list[dict]:
    recs = []
    applied_domains = {_extract_domain(j.get("application_url") or j.get("url", ""))
                       for j in applied}
    for domain, data in domain_failures.items():
        if not domain or domain == "unknown":
            continue
        if data["categories"].get("sso", 0) >= 2:
            recs.append({
                "action": "add_blocked_sso", "target": domain,
                "reason": f"{data['categories']['sso']} SSO failures",
                "priority": "high",
            })
        if data["categories"].get("captcha", 0) >= 2 and domain not in applied_domains:
            recs.append({
                "action": "add_manual_ats", "target": domain,
                "reason": f"{data['categories']['captcha']} CAPTCHA failures, never succeeded",
                "priority": "high",
            })
        if data["count"] >= 3 and domain not in applied_domains:
            dominant_cat = data["categories"].most_common(1)[0][0]
            if dominant_cat in ("captcha", "sso", "login_issue"):
                recs.append({
                    "action": "add_blocked_url", "target": f"%{domain}%",
                    "reason": f"{data['count']} failures (mostly {dominant_cat}), never succeeded",
                    "priority": "medium",
                })
    mfa_domains = Counter()
    for f in mfa_findings:
        mfa_domains[f["domain"]] += 1
    for domain, count in mfa_domains.most_common():
        if count >= 2:
            recs.append({
                "action": "needs_mfa_handling", "target": domain,
                "reason": f"{count} MFA/email-code failures detected in logs",
                "priority": "high",
            })
    if by_category.get("form_issue", 0) >= 3:
        recs.append({
            "action": "review_form_issues", "target": "prompt.py",
            "reason": f"{by_category['form_issue']} form-filling failures",
            "priority": "medium",
        })
    return sorted(recs, key=lambda r: {"high": 0, "medium": 1, "low": 2}.get(r["priority"], 3))


def apply_fixes(recommendations: list[dict], dry_run: bool = False) -> list[str]:
    import yaml
    sites_path = CONFIG_DIR / "sites.yaml"
    if not sites_path.exists():
        return ["sites.yaml not found -- cannot apply fixes"]
    config = yaml.safe_load(sites_path.read_text(encoding="utf-8")) or {}
    changes = []
    modified = False
    if "blocked_sso" not in config:
        config["blocked_sso"] = []
    if "manual_ats" not in config:
        config["manual_ats"] = []
    if "blocked" not in config:
        config["blocked"] = {}
    if "url_patterns" not in config["blocked"]:
        config["blocked"]["url_patterns"] = []
    if "sites" not in config["blocked"]:
        config["blocked"]["sites"] = []
    for rec in recommendations:
        action = rec["action"]
        target = rec["target"]
        if action == "add_blocked_sso":
            if target not in config["blocked_sso"]:
                desc = f"Add '{target}' to blocked_sso (reason: {rec['reason']})"
                changes.append(desc)
                if not dry_run:
                    config["blocked_sso"].append(target)
                    modified = True
        elif action == "add_manual_ats":
            if target not in config["manual_ats"]:
                desc = f"Add '{target}' to manual_ats (reason: {rec['reason']})"
                changes.append(desc)
                if not dry_run:
                    config["manual_ats"].append(target)
                    modified = True
        elif action == "add_blocked_url":
            if target not in config["blocked"]["url_patterns"]:
                desc = f"Add '{target}' to blocked URL patterns (reason: {rec['reason']})"
                changes.append(desc)
                if not dry_run:
                    config["blocked"]["url_patterns"].append(target)
                    modified = True
        elif action == "needs_mfa_handling":
            changes.append(
                f"[info] Domain '{target}' needs MFA/email-code handling -- "
                f"prompt.py MFA section covers this ({rec['reason']})"
            )
        elif action == "review_form_issues":
            changes.append(f"[info] {rec['reason']} -- check worker logs for recurring form patterns")
    if modified and not dry_run:
        header = f"# Auto-updated by 'applypilot analyze' on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        content = yaml.dump(config, default_flow_style=False, sort_keys=False)
        sites_path.write_text(header + content, encoding="utf-8")
        changes.append(f"\nWrote updates to {sites_path}")
    return changes


def reset_fixable_failures(recommendations: list[dict], dry_run: bool = False) -> int:
    conn = get_connection()
    reset_count = 0
    mfa_domains = [
        rec["target"] for rec in recommendations
        if rec["action"] == "needs_mfa_handling"
    ]
    if mfa_domains and not dry_run:
        for domain in mfa_domains:
            pattern = f"%{domain}%"
            cursor = conn.execute("""
                UPDATE jobs SET apply_status = NULL, apply_error = NULL,
                               apply_attempts = 0, agent_id = NULL
                WHERE (apply_status = 'failed' OR apply_status = 'login_issue')
                  AND (url LIKE ? OR application_url LIKE ?)
                  AND apply_attempts < 99
            """, (pattern, pattern))
            reset_count += cursor.rowcount
        conn.commit()
    return reset_count


def print_report(analysis: dict) -> None:
    summary = analysis["summary"]
    console.print("\n[bold blue]ApplyPilot Failure Analysis[/bold blue]\n")
    total = summary["total_failed"] + summary["total_applied"]
    success_rate = summary["total_applied"] / total * 100 if total > 0 else 0
    console.print(f"  Total applied:  [green]{summary['total_applied']}[/green]")
    console.print(f"  Total failed:   [red]{summary['total_failed']}[/red]")
    console.print(f"  Success rate:   {success_rate:.1f}%")
    console.print()
    if summary["by_category"]:
        cat_table = Table(title="Failures by Category", show_header=True, header_style="bold yellow")
        cat_table.add_column("Category")
        cat_table.add_column("Count", justify="right")
        cat_table.add_column("Action")
        category_actions = {
            "mfa_email": "[green]Auto-fix: prompt now handles email MFA[/green]",
            "sso": "[green]Auto-fix: adding to blocked_sso[/green]",
            "captcha": "[yellow]Review CapSolver config[/yellow]",
            "login_issue": "[green]Auto-fix: improved login flow[/green]",
            "form_issue": "[yellow]Check logs for form patterns[/yellow]",
            "expired": "[dim]No action (jobs expired)[/dim]",
            "not_eligible": "[dim]No action (location/auth filter)[/dim]",
            "not_a_job": "[dim]No action (not a job application)[/dim]",
            "other": "[yellow]Review logs manually[/yellow]",
        }
        for category, count in summary["by_category"].items():
            action = category_actions.get(category, "")
            cat_table.add_row(category, str(count), action)
        console.print(cat_table)
        console.print()
    recs = analysis["recommendations"]
    if recs:
        rec_table = Table(title="Recommendations", show_header=True, header_style="bold green")
        rec_table.add_column("Priority")
        rec_table.add_column("Action")
        rec_table.add_column("Target")
        rec_table.add_column("Reason")
        for rec in recs:
            priority_style = {"high": "[red]HIGH[/red]", "medium": "[yellow]MED[/yellow]",
                              "low": "[dim]LOW[/dim]"}.get(rec["priority"], rec["priority"])
            rec_table.add_row(priority_style, rec["action"], rec["target"], rec["reason"])
        console.print(rec_table)
    else:
        console.print("[green]No actionable recommendations -- looking good![/green]")
    console.print()


# ── Per-job analysis & fix ────────────────────────────────────────────────

# Same registry as apply/launcher.py:_PLATFORM_HOSTS but kept local so
# analyze can run against rows that predate the captured-detail column.
_PLATFORM_HOSTS_FOR_ANALYZE: tuple[tuple[str, str], ...] = (
    ("myworkdayjobs.com", "workday"),
    ("workday.com", "workday"),
    ("greenhouse.io", "greenhouse"),
    ("lever.co", "lever"),
    ("smartrecruiters.com", "smartrecruiters"),
    ("icims.com", "icims"),
    ("ashbyhq.com", "ashby"),
    ("breezy.hr", "breezy"),
    ("recruitee.com", "recruitee"),
    ("bamboohr.com", "bamboohr"),
    ("jobvite.com", "jobvite"),
)


def _detect_platform_from_host(host: str | None) -> tuple[str | None, str | None]:
    if not host:
        return (None, None)
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    for needle, platform in _PLATFORM_HOSTS_FOR_ANALYZE:
        if needle in host:
            tenant = host.split(".", 1)[0] if host else None
            return (platform, tenant)
    return (None, None)


def _fetch_job(url: str) -> dict | None:
    conn = get_connection()
    row = conn.execute(
        "SELECT url, title, site, application_url, apply_status, apply_error, "
        "apply_attempts, last_attempted_at, apply_failure_detail, "
        "tailored_resume_path, fit_score "
        "FROM jobs WHERE url = ? OR application_url = ?",
        (url, url),
    ).fetchone()
    if row is None:
        return None
    return {k: row[k] for k in row.keys()}


def _find_matching_log(job: dict) -> Path | None:
    """Pick the most recent claude_*.txt log for this job's site."""
    site = (job.get("site") or "").split(":", 1)[0]
    if not site:
        return None
    safe_site = re.sub(r"[^\w-]", "", site)[:20]
    if not safe_site:
        return None
    candidates = sorted(
        LOG_DIR.glob(f"claude_*_{safe_site}*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def classify_job(job: dict) -> dict:
    """Return a fix recommendation for one failed job.

    Reads `apply_failure_detail` if captured at run time, falls back to
    pattern-matching against `apply_error` for older rows.
    """
    detail_raw = job.get("apply_failure_detail") or ""
    detail: dict = {}
    if detail_raw:
        try:
            detail = json.loads(detail_raw)
        except json.JSONDecodeError:
            pass

    stage = detail.get("stage") or _match_category(job.get("apply_error") or "")
    platform = detail.get("platform")
    tenant = detail.get("tenant")
    has_mfa = bool(detail.get("mfa_signals"))

    # Decision matrix.
    fix_action = "review"
    fix_reason = "no automatic fix; review log manually"
    blocklist_pattern: str | None = None

    if stage in {"mfa_email", "mfa"} or has_mfa:
        # The most concrete user request: Workday + email/MFA → block this
        # tenant on future runs. For non-Workday MFA we still block by host.
        if platform == "workday" and tenant:
            blocklist_pattern = f"%{tenant}.%myworkdayjobs.com%"
            fix_action = "block_pattern"
            fix_reason = f"Workday tenant '{tenant}' requires email MFA — adding blocklist pattern so future runs skip it"
        elif tenant:
            blocklist_pattern = f"%{tenant}%"
            fix_action = "block_pattern"
            fix_reason = f"Email/MFA gate detected at {tenant}; blocklisting"
        else:
            fix_action = "manual"
            fix_reason = "MFA detected but no platform tenant identified — marking manual"
    elif stage == "captcha":
        # CAPTCHAs we can't solve → mark site manual.
        fix_action = "manual_ats"
        fix_reason = "Unsolvable CAPTCHA detected"
    elif stage in {"login", "login_issue", "sso"}:
        # Only blocklist by host when this is a tenant-based ATS (workday,
        # greenhouse, icims, lever, etc.). For general job boards (dice,
        # indeed, linkedin) blocking the whole site would lose all future
        # postings — mark this single job manual instead.
        host = ""
        try:
            host = urlparse(detail.get("last_url") or job.get("application_url") or "").hostname or ""
        except Exception:
            pass
        platform_for_host, tenant_for_host = _detect_platform_from_host(host)
        if platform_for_host and tenant_for_host:
            blocklist_pattern = f"%{tenant_for_host}.%{platform_for_host}%" if platform_for_host == "workday" else f"%{host}%"
            fix_action = "block_pattern"
            fix_reason = f"{platform_for_host} tenant '{tenant_for_host}' gates with SSO/login — blocklisting"
        else:
            fix_action = "manual"
            fix_reason = f"Login gate at {host or 'unknown host'} — marking this job manual (won't blocklist whole site)"
    elif stage in {"form", "upload", "submit"}:
        fix_action = "reset"
        fix_reason = "Form/upload/submit failure — resetting for retry with current prompt fixes"
    elif stage == "transient" or "broken pipe" in (job.get("apply_error") or "").lower() or "timeout" in (job.get("apply_error") or "").lower():
        fix_action = "reset"
        fix_reason = "Transient (timeout/broken pipe) — safe to retry"
    elif stage in {"expired", "not_eligible"}:
        fix_action = "manual"
        fix_reason = f"{stage} — won't succeed on retry, marking manual"
    elif stage == "no_result_line":
        fix_action = "reset"
        fix_reason = "Agent didn't print RESULT line — likely truncated; reset to retry"

    return {
        "stage": stage,
        "platform": platform,
        "tenant": tenant,
        "mfa_signals": detail.get("mfa_signals", []),
        "last_url": detail.get("last_url"),
        "last_tool": detail.get("last_tool"),
        "fix_action": fix_action,
        "fix_reason": fix_reason,
        "blocklist_pattern": blocklist_pattern,
        "raw_detail": detail,
    }


def print_job_report(job: dict, classification: dict, log_path: Path | None) -> None:
    """Pretty-print a single job's failure analysis."""
    title = job.get("title") or "?"
    site = job.get("site") or "?"
    console.print(Panel.fit(
        f"[bold]{title[:60]}[/bold]  @  {site}\n"
        f"[dim]{job.get('url')}[/dim]",
        border_style="cyan",
    ))
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("Status", str(job.get("apply_status")))
    table.add_row("Error", str(job.get("apply_error")))
    table.add_row("Attempts", str(job.get("apply_attempts")))
    table.add_row("Last attempt", str(job.get("last_attempted_at")))
    table.add_row("Stage", classification["stage"] or "?")
    table.add_row("Platform", f"{classification['platform'] or '-'} (tenant: {classification['tenant'] or '-'})")
    table.add_row("Last URL", str(classification.get("last_url") or "-"))
    table.add_row("Last tool", str(classification.get("last_tool") or "-"))
    if classification["mfa_signals"]:
        table.add_row("MFA signals", ", ".join(classification["mfa_signals"]))
    console.print(table)
    console.print()
    console.print(f"[bold yellow]Recommended fix:[/bold yellow] [green]{classification['fix_action']}[/green] — {classification['fix_reason']}")
    if classification["blocklist_pattern"]:
        console.print(f"[dim]Pattern to add:[/dim] {classification['blocklist_pattern']}")
    console.print()
    if log_path:
        console.print(f"[dim]Log:[/dim] {log_path}")
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-20:]
            for line in tail:
                console.print(f"  [dim]{line[:200]}[/dim]")
        except Exception as exc:
            console.print(f"  [red]could not read log: {exc}[/red]")
    else:
        console.print("[dim]No matching claude_*.txt log found.[/dim]")


def apply_job_fix(url: str, *, dry_run: bool = False) -> dict:
    """Run the per-job recommended fix. Returns {ok, action, message}."""
    job = _fetch_job(url)
    if job is None:
        return {"ok": False, "action": "none", "message": f"no job found for {url}"}

    classification = classify_job(job)
    action = classification["fix_action"]
    pattern = classification["blocklist_pattern"]
    msg_parts: list[str] = []
    conn = get_connection()

    if action == "reset":
        if not dry_run:
            conn.execute(
                "UPDATE jobs SET apply_status = NULL, apply_error = NULL, "
                "apply_attempts = 0, agent_id = NULL WHERE url = ?",
                (job["url"],),
            )
            conn.commit()
        msg_parts.append("reset for retry")

    elif action == "manual":
        if not dry_run:
            conn.execute(
                "UPDATE jobs SET apply_status = 'manual', apply_attempts = 99 WHERE url = ?",
                (job["url"],),
            )
            conn.commit()
        msg_parts.append("marked manual (won't be retried)")

    elif action == "manual_ats":
        # Add the host to manual_ats list in sites.yaml.
        host = ""
        try:
            host = urlparse(job.get("application_url") or job.get("url", "")).hostname or ""
            if host.startswith("www."):
                host = host[4:]
        except Exception:
            pass
        if host:
            _add_to_sites_yaml(field="manual_ats", value=host, dry_run=dry_run)
            msg_parts.append(f"added '{host}' to manual_ats")
        if not dry_run:
            conn.execute(
                "UPDATE jobs SET apply_status = 'manual', apply_attempts = 99 WHERE url = ?",
                (job["url"],),
            )
            conn.commit()
        msg_parts.append("marked manual")

    elif action == "block_pattern" and pattern:
        _add_to_sites_yaml(field=("blocked", "url_patterns"), value=pattern, dry_run=dry_run)
        msg_parts.append(f"added blocklist pattern '{pattern}'")
        # Also mark every other future job that matches this pattern as manual
        # so the queue is cleaned in one shot.
        if not dry_run:
            cur = conn.execute(
                "UPDATE jobs SET apply_status = 'manual', apply_attempts = 99 "
                "WHERE (url LIKE ? OR COALESCE(application_url,'') LIKE ?) "
                "AND apply_status != 'applied'",
                (pattern, pattern),
            )
            conn.commit()
            msg_parts.append(f"marked {cur.rowcount} matching jobs manual")

    else:
        msg_parts.append("no automatic fix — manual review")

    return {
        "ok": True,
        "action": action,
        "stage": classification["stage"],
        "pattern": pattern,
        "message": "; ".join(msg_parts),
        "dry_run": dry_run,
    }


def _add_to_sites_yaml(*, field, value: str, dry_run: bool = False) -> bool:
    """Add `value` to a field in sites.yaml. Idempotent.

    `field` is either a string (top-level key like "manual_ats") or a tuple
    of nested keys ("blocked", "url_patterns"). Returns True if a change
    would be made.
    """
    import yaml
    sites_path = CONFIG_DIR / "sites.yaml"
    if not sites_path.exists():
        return False
    config = yaml.safe_load(sites_path.read_text(encoding="utf-8")) or {}

    if isinstance(field, str):
        bucket = config.setdefault(field, [])
    else:
        cur = config
        for key in field[:-1]:
            cur = cur.setdefault(key, {})
        bucket = cur.setdefault(field[-1], [])

    if value in bucket:
        return False
    if dry_run:
        return True
    bucket.append(value)
    header = f"# Auto-updated by 'applypilot analyze' on {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    sites_path.write_text(header + yaml.dump(config, default_flow_style=False, sort_keys=False), encoding="utf-8")
    return True


def run_auto_fix_all(*, dry_run: bool = False) -> dict:
    """Apply per-job fixes across every failed job.

    Walks `apply_status='failed'` (and other non-applied non-in_progress
    states) and runs `apply_job_fix` on each. Aggregates by action so the
    user sees one summary instead of 149 lines.
    """
    init_db()
    conn = get_connection()
    rows = conn.execute(
        "SELECT url FROM jobs WHERE apply_status IS NOT NULL "
        "AND apply_status NOT IN ('applied', 'in_progress', 'manual') "
        "ORDER BY last_attempted_at DESC"
    ).fetchall()
    counts: Counter = Counter()
    patterns_added: set[str] = set()

    for row in rows:
        result = apply_job_fix(row["url"], dry_run=dry_run)
        counts[result["action"]] += 1
        if result.get("pattern"):
            patterns_added.add(result["pattern"])

    console.print()
    console.print(f"[bold]Auto-fix sweep ({'dry run' if dry_run else 'applied'}):[/bold] {len(rows)} failed jobs")
    for action, n in counts.most_common():
        console.print(f"  [green]{action}[/green]: {n}")
    if patterns_added:
        console.print()
        console.print("[bold]Blocklist patterns added:[/bold]")
        for p in sorted(patterns_added):
            console.print(f"  {p}")
    console.print()
    return {"total": len(rows), "by_action": dict(counts), "patterns": sorted(patterns_added)}


def run_job_analysis(url: str, *, fix: bool = False, dry_run: bool = False) -> int:
    """CLI entry: analyze (and optionally fix) one failing job."""
    init_db()
    job = _fetch_job(url)
    if job is None:
        console.print(f"[red]No job found for url[/red]: {url}")
        return 1

    classification = classify_job(job)
    log_path = _find_matching_log(job)
    print_job_report(job, classification, log_path)

    if not fix:
        return 0

    result = apply_job_fix(url, dry_run=dry_run)
    prefix = "[dim]would:[/dim] " if dry_run else "[green]done:[/green] "
    console.print(f"\n{prefix}{result['message']}")
    return 0


def run_analysis(dry_run: bool = False, reset: bool = False) -> dict:
    init_db()
    analysis = analyze_failures()
    print_report(analysis)
    recs = analysis["recommendations"]
    if recs:
        console.print("[bold]Applying fixes...[/bold]" if not dry_run
                      else "[bold]Dry run -- showing what would change:[/bold]")
        changes = apply_fixes(recs, dry_run=dry_run)
        for change in changes:
            prefix = "[dim]would:[/dim] " if dry_run else "[green]done:[/green] "
            console.print(f"  {prefix}{change}")
        console.print()
        if reset and not dry_run:
            count = reset_fixable_failures(recs)
            if count:
                console.print(f"[green]Reset {count} job(s) for retry with improved handling.[/green]")
            else:
                console.print("[dim]No jobs eligible for reset.[/dim]")
        elif reset and dry_run:
            console.print("[dim]Would reset MFA-failed jobs for retry.[/dim]")
        console.print()
    else:
        console.print("[green]No fixes needed.[/green]\n")
    return analysis
