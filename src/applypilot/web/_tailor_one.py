"""Single-URL retailor helper that streams LLM chunks to stdout.

Invoked as `python -m applypilot.web._tailor_one --url <url>` from the
web UI. Streams `litellm.completion(..., stream=True)` chunks via
`print(..., flush=True)` so the parent SSE pipe forwards tokens as
they arrive — the user sees the resume materializing live.

We intentionally bypass the heavier validate/judge loop in
`scoring/tailor.py` here: this is the "give me a quick alternate take"
button in the UI, not the full nightly pipeline. The on-disk artifacts
(.txt and _REPORT.json) are still written so the existing job_detail
view picks them up.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import litellm

from applypilot.config import RESUME_PATH, TAILORED_DIR, ensure_dirs, load_env, load_profile
from applypilot.database import get_connection, init_db
from applypilot.llm import resolve_llm_config

log = logging.getLogger(__name__)


def _safe_prefix(job: dict) -> str:
    title = re.sub(r"[^\w\s-]", "", job.get("title") or "untitled")[:50].strip().replace(" ", "_")
    site = re.sub(r"[^\w\s-]", "", job.get("site") or "site")[:20].strip().replace(" ", "_")
    job_id = hashlib.md5((job.get("url") or "").encode("utf-8")).hexdigest()[:6]
    return f"{site}_{title}_{job_id}"


def _build_prompt(resume_text: str, job: dict, profile: dict) -> list[dict]:
    description = job.get("full_description") or job.get("description") or ""
    title = job.get("title") or ""
    summary = (
        "You are a senior recruiter rewriting a resume to win an interview for the role below. "
        "Return the full tailored resume as plain text. Keep it human and specific — no buzzword stacks. "
        "Reorder skills so the role's must-haves come first; reframe bullets around the target work."
    )
    user = (
        f"## Target role\n{title}\n\n"
        f"## Job description\n{description[:6000]}\n\n"
        f"## Candidate profile (factual, do not invent)\n{json.dumps(profile, indent=2)[:4000]}\n\n"
        f"## Base resume\n{resume_text}\n\n"
        "Write the tailored resume now."
    )
    return [
        {"role": "system", "content": summary},
        {"role": "user", "content": user},
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Retailor a single job's resume with live streaming.")
    parser.add_argument("--url", required=True, help="Exact URL of the job in the DB.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_env()
    ensure_dirs()
    init_db()

    conn = get_connection()
    row = conn.execute("SELECT * FROM jobs WHERE url = ?", (args.url,)).fetchone()
    if row is None:
        print(f"ERROR: no job found for url={args.url!r}", flush=True)
        return 2
    job = dict(zip(row.keys(), row))

    try:
        profile = load_profile()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", flush=True)
        return 3

    if not RESUME_PATH.exists():
        print(f"ERROR: resume.txt missing at {RESUME_PATH}", flush=True)
        return 4
    resume_text = RESUME_PATH.read_text(encoding="utf-8")

    cfg = resolve_llm_config()
    print(f"-- model: {cfg.model} --", flush=True)
    print(f"-- url:   {args.url} --", flush=True)
    print(f"-- title: {job.get('title')!r} --", flush=True)

    messages = _build_prompt(resume_text, job, profile)
    started = time.time()
    chunks: list[str] = []

    try:
        response = litellm.completion(
            model=cfg.model,
            messages=messages,
            api_key=cfg.api_key or None,
            api_base=cfg.api_base or None,
            stream=True,
            max_tokens=4096,
            drop_params=True,
        )
        for chunk in response:
            try:
                delta = chunk.choices[0].delta
            except (IndexError, AttributeError):
                continue
            piece = getattr(delta, "content", None)
            if not piece:
                continue
            chunks.append(piece)
            # Token-by-token forward to parent SSE pipe.
            print(piece, end="", flush=True)
        print()  # final newline so the buffer flushes a clean event
    except Exception as exc:
        print(f"\nERROR: LLM stream failed: {exc}", flush=True)
        return 5

    elapsed = time.time() - started

    text = "".join(chunks).strip()
    if not text:
        print("ERROR: empty response", flush=True)
        return 6

    prefix = _safe_prefix(job)
    out_dir = Path(TAILORED_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    txt_path = out_dir / f"{prefix}.txt"
    txt_path.write_text(text, encoding="utf-8")

    report = {
        "status": "approved_via_web",
        "attempts": 1,
        "elapsed_sec": round(elapsed, 2),
        "model": cfg.model,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "web._tailor_one",
    }
    report_path = out_dir / f"{prefix}_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Best-effort PDF; safe to fail silently.
    try:
        from applypilot.scoring.pdf import convert_to_pdf

        pdf_path = convert_to_pdf(txt_path)
        print(f"\n-- pdf: {pdf_path} --", flush=True)
    except Exception as exc:
        log.debug("PDF conversion failed: %s", exc)

    # Update DB last so a partial failure above doesn't claim success.
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE jobs SET tailored_resume_path = ?, tailored_at = ?, "
        "tailor_attempts = COALESCE(tailor_attempts, 0) + 1 WHERE url = ?",
        (str(txt_path), now, args.url),
    )
    conn.commit()
    print(f"\n-- saved: {txt_path} ({elapsed:.1f}s) --", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
