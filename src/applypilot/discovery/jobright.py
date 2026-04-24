"""JobRight.ai recommended-jobs scraper.

Logs into jobright.ai with stored credentials, walks the recommended-jobs
panel, follows each card to its external apply URL, and stores the result
in the ApplyPilot jobs table. Adapted from the standalone jobright_agent
project so it runs as one of the discovery sub-stages.

Credentials come from env vars JOBRIGHT_EMAIL / JOBRIGHT_PASSWORD. If
either is missing the scraper logs a warning and exits cleanly so the
rest of the discover stage still runs.
"""

import asyncio
import logging
import os
import re
import sqlite3
from datetime import datetime, timezone

from playwright.async_api import async_playwright, TimeoutError as PWTimeout

from applypilot import config
from applypilot.database import get_connection, init_db

log = logging.getLogger(__name__)

JOBRIGHT_URL = "https://jobright.ai"
RECOMMEND_URL = "https://jobright.ai/jobs/recommended"

ATS_PATTERNS = {
    "workday":         re.compile(r"myworkdayjobs\.com|wd\d+\.myworkday|workday\.com", re.I),
    "greenhouse":      re.compile(r"boards\.greenhouse\.io|job-boards\.greenhouse\.io|greenhouse\.io", re.I),
    "lever":           re.compile(r"jobs\.lever\.co|lever\.co", re.I),
    "ashby":           re.compile(r"jobs\.ashbyhq\.com|ashbyhq\.com", re.I),
    "icims":           re.compile(r"icims\.com", re.I),
    "oracle":          re.compile(r"oraclecloud\.com|taleo\.net|oracle\.com/careers", re.I),
    "smartrecruiters": re.compile(r"smartrecruiters\.com", re.I),
    "jobvite":         re.compile(r"jobvite\.com", re.I),
    "brassring":       re.compile(r"brassring\.com|kenexa\.com", re.I),
    "successfactors":  re.compile(r"successfactors\.com|sap\.com/careers", re.I),
    "rippling":        re.compile(r"ats\.rippling\.com", re.I),
    "workable":        re.compile(r"apply\.workable\.com|workable\.com", re.I),
    "ultipro":         re.compile(r"recruiting\.ultipro\.com", re.I),
}


def detect_ats(url: str) -> str:
    if not url:
        return "unknown"
    for name, pat in ATS_PATTERNS.items():
        if pat.search(url):
            return name
    return "other"


_URL_COMPANY_PATTERNS = [
    re.compile(r"job-boards\.greenhouse\.io/([^/?#]+)", re.I),
    re.compile(r"boards\.greenhouse\.io/([^/?#]+)",     re.I),
    re.compile(r"boards\.eu\.greenhouse\.io/([^/?#]+)", re.I),
    re.compile(r"jobs\.lever\.co/([^/?#]+)",            re.I),
    re.compile(r"jobs\.ashbyhq\.com/([^/?#]+)",         re.I),
    re.compile(r"apply\.workable\.com/([^/?#]+)",       re.I),
    re.compile(r"ats\.rippling\.com/([^/?#]+?)-careers", re.I),
    re.compile(r"ats\.rippling\.com/([^/?#]+)",         re.I),
    re.compile(r"([a-z0-9][a-z0-9-]+)\.wd\d+\.myworkdayjobs\.com", re.I),
    re.compile(r"careers-page\.com/([^/?#]+)",          re.I),
    re.compile(r"jobs\.smartrecruiters\.com/([^/?#]+)", re.I),
    re.compile(r"jobs\.jobvite\.com/careers/([^/?#]+)", re.I),
]

_SLUG_BLOCKLIST = {
    "embed", "external", "public", "careers", "jobs", "job", "apply",
    "boards", "job_app", "company", "c",
}


def company_from_url(url: str) -> str:
    if not url:
        return ""
    for pat in _URL_COMPANY_PATTERNS:
        m = pat.search(url)
        if not m:
            continue
        slug = m.group(1).lower()
        if slug in _SLUG_BLOCKLIST:
            continue
        if slug.isdigit() or re.fullmatch(r"[a-z0-9]{16,}", slug):
            continue
        name = slug.replace("-", " ").replace("_", " ").strip()
        name = re.sub(r"\s+(careers|jobs|inc|llc)$", "", name, flags=re.I).strip()
        if not name:
            continue
        return " ".join(w.upper() if len(w) <= 3 and w.isalpha() else w.capitalize()
                        for w in name.split())
    return ""


def _load_filters(search_cfg: dict | None) -> tuple[list[str], list[str]]:
    """Pull title include/exclude lists out of the user's search config."""
    if not search_cfg:
        return [], []
    include = search_cfg.get("title_keywords") or []
    exclude = search_cfg.get("exclude_keywords") or []
    return include, exclude


def _passes_filters(title: str, include: list[str], exclude: list[str]) -> bool:
    t = (title or "").lower()
    if include and not any(kw.lower() in t for kw in include):
        return False
    if any(kw.lower() in t for kw in exclude):
        return False
    return True


async def _dismiss_popup(page):
    close_selectors = [
        'button[aria-label="Close"]',
        'button[aria-label="close"]',
        '.ant-modal-close',
        '.ant-modal-wrap button.ant-modal-close',
        '[class*="modal"] button[class*="close"]',
        '[class*="modal"] [class*="close-icon"]',
        '[class*="popup"] button[class*="close"]',
        '[role="dialog"] button[aria-label="Close"]',
        '[role="dialog"] button[class*="close"]',
    ]
    for sel in close_selectors:
        try:
            btn = page.locator(sel).first
            if await btn.count() > 0 and await btn.is_visible():
                await btn.click(timeout=3_000)
                await page.wait_for_timeout(500)
                return
        except Exception:
            continue
    try:
        await page.keyboard.press("Escape")
        await page.wait_for_timeout(500)
    except Exception:
        pass


async def _login(page, email: str, password: str):
    log.info("Logging into JobRight as %s", email)
    await page.goto(JOBRIGHT_URL, wait_until="domcontentloaded")
    await page.click('//*[@id="__next"]/main/div/header/div[2]/span')
    await page.wait_for_timeout(2000)
    await page.wait_for_selector('//*[@id="basic_email"]', state="visible", timeout=15_000)
    await page.fill('//*[@id="basic_email"]', email)
    await page.fill('//*[@id="basic_password"]', password)
    await page.click('//*[@id="basic"]/div/div[3]/div/div/div/div/button')
    await page.wait_for_url("**/jobs/**", timeout=15_000)
    log.info("JobRight login successful")


async def _scrape_external_description(page, url: str) -> str:
    try:
        await page.goto(url, wait_until="domcontentloaded", timeout=20_000)
        await page.wait_for_timeout(2000)
        for sel in [
            '[class*="job-description"]',
            '[class*="description"]',
            '[data-testid="job-description"]',
            'article',
            'main',
        ]:
            el = page.locator(sel).first
            if await el.count() > 0:
                return (await el.inner_text()).strip()
    except PWTimeout:
        log.warning("Timeout loading external posting: %s", url)
    return ""


def _store(conn: sqlite3.Connection, job: dict) -> bool:
    """Insert one JobRight result. Returns True if it was new."""
    now = datetime.now(timezone.utc).isoformat()
    description = job.get("description") or None
    full_description = description if description and len(description) > 200 else None
    detail_scraped_at = now if full_description else None

    location = job.get("location") or None
    company = job.get("company") or None
    site_label = "jobright"
    if company:
        site_label = f"jobright:{company}"

    try:
        conn.execute(
            "INSERT INTO jobs (url, title, description, location, site, strategy, "
            "discovered_at, full_description, application_url, detail_scraped_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                job["url"],
                job.get("title"),
                description,
                location,
                site_label,
                "jobright",
                now,
                full_description,
                job.get("apply_url") or None,
                detail_scraped_at,
            ),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


async def _scrape_recommended(email: str, password: str, headless: bool) -> dict:
    search_cfg = config.load_search_config() or {}
    include, exclude = _load_filters(search_cfg)

    conn = init_db()
    new_count = 0
    dupe_count = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        ctx = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            )
        )
        page = await ctx.new_page()

        try:
            await _login(page, email, password)
            await _dismiss_popup(page)
            await page.goto(RECOMMEND_URL, wait_until="domcontentloaded", timeout=60_000)
            await page.wait_for_timeout(3000)
            await _dismiss_popup(page)

            await page.evaluate("() => document.getElementById('scrollableDiv')?.scrollTo(0, 0)")
            await page.wait_for_timeout(500)
            await page.evaluate("""() => {
                const c = document.getElementById('scrollableDiv');
                if (c) c.scrollTo({top: c.scrollHeight, behavior: 'instant'});
            }""")
            await page.wait_for_timeout(1500)
            await page.evaluate("() => document.getElementById('scrollableDiv')?.scrollTo(0, 0)")
            await page.wait_for_timeout(1000)

            cards_raw = await page.evaluate("""() => {
                const cards = document.querySelectorAll('.job-card-flag-classname');
                // Pick the first <h2> whose text is NOT a match-percentage
                // badge (e.g. "86%") and is longer than 3 chars — that's
                // the real job title. Falls back to 'Unknown' if nothing
                // reasonable is found.
                const pickTitle = (el) => {
                    const h2s = el.querySelectorAll('h2');
                    for (const h of h2s) {
                        const t = (h.innerText || '').trim();
                        if (t && !/^\\d{1,3}%$/.test(t) && t.length > 3) return t;
                    }
                    return 'Unknown';
                };
                return Array.from(cards).filter(el => el.id).map(el => ({
                    id:       el.id,
                    title:    pickTitle(el),
                    location: el.querySelector('[class*="location"]')?.innerText?.trim() ?? '',
                }));
            }""")
            log.info("JobRight: found %d cards", len(cards_raw))

            for card in cards_raw:
                try:
                    card_id = card["id"]
                    title = card["title"]
                    location = card["location"]
                    job_url = f"{JOBRIGHT_URL}/jobs/{card_id}"

                    if not _passes_filters(title, include, exclude):
                        log.debug("Filtered out: %s", title)
                        continue

                    # Quick dedup check before doing the expensive panel work
                    existing = conn.execute(
                        "SELECT 1 FROM jobs WHERE url = ?", (job_url,)
                    ).fetchone()
                    if existing:
                        dupe_count += 1
                        continue

                    await page.goto(RECOMMEND_URL, wait_until="domcontentloaded", timeout=60_000)
                    await page.wait_for_timeout(2000)
                    await _dismiss_popup(page)

                    card_el = page.locator(f'[id="{card_id}"]').first
                    h2_el = page.locator(f'[id="{card_id}"] h2').first
                    try:
                        await card_el.scroll_into_view_if_needed(timeout=8_000)
                        await page.wait_for_timeout(400)
                        if await h2_el.count() > 0:
                            await h2_el.click(force=True)
                        else:
                            await card_el.click(force=True)
                    except Exception as click_err:
                        log.warning("Could not click card %s: %s", card_id, click_err)

                    await page.wait_for_timeout(3000)

                    apply_url = ""
                    for label in [
                        "Original Job Post", "Apply Now", "Apply Here",
                        "View Job", "Apply Externally", "Easy Apply", "Apply",
                    ]:
                        link = page.locator(f'a:has-text("{label}")').first
                        if await link.count() > 0:
                            href = await link.get_attribute("href") or ""
                            if href and "jobright.ai" not in href and href.startswith("http"):
                                apply_url = href
                                break

                    if not apply_url:
                        apply_url = await page.evaluate("""() => {
                            const ats = [
                                'greenhouse.io', 'lever.co', 'ashbyhq.com',
                                'myworkdayjobs.com', 'myworkday.com',
                                'wd1.myworkday', 'wd3.myworkday', 'wd5.myworkday',
                                'icims.com', 'smartrecruiters.com', 'jobvite.com',
                                'taleo.net', 'successfactors.com', 'brassring.com',
                                'oraclecloud.com',
                            ];
                            for (const a of document.querySelectorAll('a[href]')) {
                                if (ats.some(p => a.href.includes(p))) return a.href;
                            }
                            return '';
                        }""")

                    company = await page.evaluate("""() => {
                        const clean = (t) => (t || '').replace(/\\s+/g, ' ').trim();
                        const skip = /jobright|linkedin|glassdoor|greenhouse|lever|ashby|rippling|indeed|google|patent|businesswire|prnewswire|newsfile|businessghana|graphic\\.com|solutionsreview|medcitynews|bhbusiness/i;
                        for (const a of document.querySelectorAll('a[href*="/company/"]')) {
                            const t = clean(a.innerText);
                            if (t && t.length > 1 && t.length < 80 && !skip.test(t)) return t;
                        }
                        for (const a of document.querySelectorAll('a[href]')) {
                            const href = a.href || '';
                            const text = clean(a.innerText);
                            if (text && /^https?:\\/\\//.test(text) && !skip.test(href)) {
                                try {
                                    const host = new URL(href).hostname.replace(/^www\\./, '');
                                    const name = host.split('.')[0];
                                    return name.charAt(0).toUpperCase() + name.slice(1);
                                } catch(e) {}
                            }
                        }
                        return '';
                    }""") or ""

                    if not company or company.lower() == "unknown":
                        slug_company = company_from_url(apply_url)
                        if slug_company:
                            company = slug_company

                    description = ""
                    if apply_url:
                        description = await _scrape_external_description(page, apply_url)

                    job = {
                        "url":         job_url,
                        "title":       title,
                        "company":     company or None,
                        "location":    location,
                        "apply_url":   apply_url,
                        "description": description,
                    }

                    if _store(conn, job):
                        new_count += 1
                        log.info(
                            "JobRight new: %s @ %s [%s]",
                            title, company or "?", detect_ats(apply_url),
                        )
                    else:
                        dupe_count += 1

                    await page.goto(RECOMMEND_URL, wait_until="domcontentloaded", timeout=60_000)
                    await page.wait_for_timeout(2000)

                except Exception as e:
                    log.warning("JobRight: error on card %s: %s", card.get("id", "?"), e)
                    try:
                        await page.goto(RECOMMEND_URL, wait_until="domcontentloaded", timeout=60_000)
                        await page.wait_for_timeout(2000)
                    except Exception:
                        pass

        finally:
            await browser.close()

    log.info("JobRight scrape complete: %d new, %d dupes", new_count, dupe_count)
    return {"new": new_count, "existing": dupe_count}


def run_jobright_discovery(headless: bool | None = None) -> dict:
    """Entry point used by the pipeline `discover` stage.

    Reads JOBRIGHT_EMAIL / JOBRIGHT_PASSWORD from the environment. Returns
    a stats dict; if credentials are missing it returns a skipped result
    instead of raising so the rest of discovery still runs.
    """
    email = os.environ.get("JOBRIGHT_EMAIL", "").strip()
    password = os.environ.get("JOBRIGHT_PASSWORD", "").strip()
    if not email or not password:
        log.warning(
            "JobRight scraper skipped — set JOBRIGHT_EMAIL and JOBRIGHT_PASSWORD "
            "in ~/.applypilot/.env to enable"
        )
        return {"new": 0, "existing": 0, "skipped": True}

    if headless is None:
        headless = os.environ.get("JOBRIGHT_HEADLESS", "0") not in ("0", "false", "False", "")

    return asyncio.run(_scrape_recommended(email, password, headless))
