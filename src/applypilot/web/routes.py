"""All REST + SSE endpoints for the ApplyPilot web UI.

Read endpoints query SQLite directly via the WAL-friendly thread-local
connection helper. Write endpoints either issue a single safe UPDATE
or — for retailor / reapply — spawn a fresh subprocess so they never
contend with the running auto-loop's writes.

API keys are masked in every response that touches them. The grep
guard in the smoke test (`AIzaSyC` must not appear in /api/models)
catches accidental leaks.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sse_starlette.sse import EventSourceResponse

from applypilot.config import APP_DIR, ENV_PATH, LOG_DIR
from applypilot.database import get_connection, get_stats
from applypilot.web import streams
from applypilot.web.models import (
    JobMarkRequest,
    ModelEntry,
    ModelSelectRequest,
    ModelsPayload,
    StreamHandle as StreamHandleModel,
)

log = logging.getLogger(__name__)

# Hardcoded model list — matches the plan's "decisions made without asking" section.
_MODEL_OPTIONS: list[ModelEntry] = [
    ModelEntry(id="gemini/gemini-2.5-flash", label="Gemini 2.5 Flash", provider="gemini"),
    ModelEntry(id="gemini/gemini-flash-latest", label="Gemini Flash (latest)", provider="gemini"),
    ModelEntry(id="gemini/gemini-2.0-flash-001", label="Gemini 2.0 Flash 001", provider="gemini"),
    ModelEntry(id="anthropic/claude-haiku-4-5", label="Claude Haiku 4.5", provider="anthropic"),
    ModelEntry(id="anthropic/claude-sonnet-4-5", label="Claude Sonnet 4.5", provider="anthropic"),
    ModelEntry(id="openai/gpt-4o-mini", label="GPT-4o mini", provider="openai"),
    ModelEntry(id="openai/gpt-5-mini", label="GPT-5 mini", provider="openai"),
    ModelEntry(id="deepseek/deepseek-chat", label="DeepSeek Chat", provider="deepseek"),
    ModelEntry(id="deepseek/deepseek-reasoner", label="DeepSeek Reasoner", provider="deepseek"),
]

_PROVIDER_KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

# Cap pagination so a typo in the query string can't OOM the box.
_MAX_LIMIT = 500
_DEFAULT_LIMIT = 50

# Job table sortable columns we expose. Anything else is rejected so we
# don't construct unsanitized SQL ORDER BY fragments.
_SORTABLE = {
    "discovered_at",
    "fit_score",
    "applied_at",
    "tailored_at",
    "title",
    "site",
}


def _mask_key(value: str | None) -> str | None:
    """Mask all but the first 5 + last 4 chars of an API key."""
    if not value:
        return None
    s = value.strip()
    if len(s) <= 9:
        # Too short to safely show end characters.
        return s[:2] + "..."
    return f"{s[:5]}...{s[-4:]}"


def _row_to_dict(row) -> dict:
    return dict(zip(row.keys(), row))


def _query_jobs(
    *,
    limit: int,
    offset: int,
    sort: str,
    direction: str,
    status: str | None,
    site: str | None,
    min_score: int | None,
    max_score: int | None,
    q: str | None,
) -> list[dict]:
    """Build and run the parameterized SELECT for /api/jobs."""
    conn = get_connection()
    where: list[str] = []
    params: list[Any] = []

    if status == "applied":
        where.append("applied_at IS NOT NULL")
    elif status == "ready":
        where.append("tailored_resume_path IS NOT NULL AND applied_at IS NULL")
    elif status == "tailored":
        where.append("tailored_resume_path IS NOT NULL")
    elif status == "scored":
        where.append("fit_score IS NOT NULL")
    elif status == "failed":
        where.append("apply_status = 'failed'")
    elif status == "pending":
        where.append("fit_score IS NULL")

    if site:
        where.append("site = ?")
        params.append(site)
    if min_score is not None:
        where.append("fit_score >= ?")
        params.append(min_score)
    if max_score is not None:
        where.append("fit_score <= ?")
        params.append(max_score)
    if q:
        where.append("(title LIKE ? OR full_description LIKE ? OR description LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like, like])

    where_clause = (" WHERE " + " AND ".join(where)) if where else ""
    order = sort if sort in _SORTABLE else "discovered_at"
    dir_kw = "DESC" if direction.lower() != "asc" else "ASC"

    sql = (
        "SELECT url, title, site, location, fit_score, applied_at, "
        "tailored_resume_path, cover_letter_path, apply_status, apply_error, "
        "discovered_at, scored_at, tailored_at, last_attempted_at, application_url, "
        "salary "
        f"FROM jobs{where_clause} "
        f"ORDER BY {order} {dir_kw} NULLS LAST "
        "LIMIT ? OFFSET ?"
    )
    params.extend([limit, offset])
    rows = conn.execute(sql, params).fetchall()
    return [_row_to_dict(r) for r in rows]


def _job_count(*, status, site, min_score, max_score, q) -> int:
    """Mirror of _query_jobs filters but COUNT(*) only — used for pagination."""
    conn = get_connection()
    where: list[str] = []
    params: list[Any] = []
    if status == "applied":
        where.append("applied_at IS NOT NULL")
    elif status == "ready":
        where.append("tailored_resume_path IS NOT NULL AND applied_at IS NULL")
    elif status == "tailored":
        where.append("tailored_resume_path IS NOT NULL")
    elif status == "scored":
        where.append("fit_score IS NOT NULL")
    elif status == "failed":
        where.append("apply_status = 'failed'")
    elif status == "pending":
        where.append("fit_score IS NULL")
    if site:
        where.append("site = ?")
        params.append(site)
    if min_score is not None:
        where.append("fit_score >= ?")
        params.append(min_score)
    if max_score is not None:
        where.append("fit_score <= ?")
        params.append(max_score)
    if q:
        where.append("(title LIKE ? OR full_description LIKE ? OR description LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like, like])
    where_clause = (" WHERE " + " AND ".join(where)) if where else ""
    sql = f"SELECT COUNT(*) FROM jobs{where_clause}"
    return conn.execute(sql, params).fetchone()[0]


def _models_payload(env: dict[str, str] | None = None) -> ModelsPayload:
    """Return the model list + masked key previews + currently-selected model."""
    e = env if env is not None else os.environ
    keys = {provider: _mask_key(e.get(envkey)) for provider, envkey in _PROVIDER_KEY_ENV.items()}
    return ModelsPayload(
        current=e.get("LLM_MODEL"),
        options=_MODEL_OPTIONS,
        keys=keys,
    )


def _parse_cost_logs() -> dict:
    """Walk worker-*.log files, sum total_cost_usd from `result` lines, group by mtime day."""
    by_day: dict[str, dict[str, float | int]] = {}
    total = 0.0
    if not LOG_DIR.exists():
        return {"total_usd": 0.0, "by_day": []}
    for path in LOG_DIR.glob("worker-*.log"):
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d")
        except OSError:
            continue
        run_cost = 0.0
        run_recorded = False
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if "total_cost_usd" not in line:
                        continue
                    line = line.strip()
                    if not line.startswith("{"):
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if obj.get("type") != "result":
                        continue
                    cost = obj.get("total_cost_usd")
                    if isinstance(cost, (int, float)):
                        run_cost += float(cost)
                        run_recorded = True
        except OSError:
            continue
        if not run_recorded:
            continue
        bucket = by_day.setdefault(mtime, {"date": mtime, "cost_usd": 0.0, "runs": 0})
        bucket["cost_usd"] = round(float(bucket["cost_usd"]) + run_cost, 4)
        bucket["runs"] = int(bucket["runs"]) + 1
        total += run_cost

    rows = sorted(by_day.values(), key=lambda r: r["date"], reverse=True)
    return {"total_usd": round(total, 4), "by_day": rows}


def _read_report_for(job_row: dict) -> dict | None:
    """Resolve `<prefix>_REPORT.json` next to the tailored resume."""
    path = job_row.get("tailored_resume_path")
    if not path:
        return None
    p = Path(path)
    report = p.with_name(p.stem + "_REPORT.json")
    if report.exists():
        try:
            return json.loads(report.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _is_loop_active(window_sec: int = 60) -> bool:
    """True if any chain/auto-loop log has been written in the last `window_sec` seconds."""
    if not LOG_DIR.exists():
        return False
    cutoff = datetime.now().timestamp() - window_sec
    for pattern in ("chain-*.log", "auto-loop-*.log"):
        for path in LOG_DIR.glob(pattern):
            try:
                if path.stat().st_mtime >= cutoff:
                    return True
            except OSError:
                continue
    return False


# ---------------------------------------------------------------------------
# Route registration — called from server.create_app().
# ---------------------------------------------------------------------------


def register_routes(app: FastAPI, *, templates_dir: Path) -> None:
    templates = Jinja2Templates(directory=str(templates_dir))

    # --- HTML pages ---

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/dashboard")

    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    async def dashboard_page(request: Request):
        return templates.TemplateResponse("dashboard.html", {"request": request, "loop_active": _is_loop_active()})

    @app.get("/jobs", response_class=HTMLResponse, include_in_schema=False)
    async def jobs_page(request: Request):
        return templates.TemplateResponse("jobs.html", {"request": request, "loop_active": _is_loop_active()})

    @app.get("/jobs/{url:path}", response_class=HTMLResponse, include_in_schema=False)
    async def job_detail_page(url: str, request: Request):
        conn = get_connection()
        row = conn.execute("SELECT * FROM jobs WHERE url = ?", (url,)).fetchone()
        if row is None:
            raise HTTPException(404, f"No job for url {url}")
        job = _row_to_dict(row)
        report = _read_report_for(job)
        tailored_text = ""
        if job.get("tailored_resume_path"):
            try:
                tailored_text = Path(job["tailored_resume_path"]).read_text(encoding="utf-8", errors="replace")
            except OSError:
                tailored_text = ""
        cover_text = ""
        if job.get("cover_letter_path"):
            try:
                cover_text = Path(job["cover_letter_path"]).read_text(encoding="utf-8", errors="replace")
            except OSError:
                cover_text = ""
        return templates.TemplateResponse(
            "job_detail.html",
            {
                "request": request,
                "job": job,
                "report": report,
                "tailored_text": tailored_text,
                "cover_text": cover_text,
                "loop_active": _is_loop_active(),
            },
        )

    @app.get("/settings", response_class=HTMLResponse, include_in_schema=False)
    async def settings_page(request: Request):
        # Convert the Pydantic model to a plain dict before passing to Jinja —
        # the `| tojson` filter uses json.dumps, which can't serialize Pydantic.
        return templates.TemplateResponse(
            "settings.html",
            {
                "request": request,
                "models": _models_payload().model_dump(),
                "loop_active": _is_loop_active(),
            },
        )

    # --- Read API ---

    @app.get("/api/jobs")
    async def api_jobs(
        limit: int = Query(_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
        offset: int = Query(0, ge=0),
        sort: str = Query("discovered_at"),
        direction: str = Query("desc"),
        status: str | None = Query(None),
        site: str | None = Query(None),
        min_score: int | None = Query(None, ge=0, le=10),
        max_score: int | None = Query(None, ge=0, le=10),
        q: str | None = Query(None, max_length=200),
    ):
        rows = _query_jobs(
            limit=limit,
            offset=offset,
            sort=sort,
            direction=direction,
            status=status,
            site=site,
            min_score=min_score,
            max_score=max_score,
            q=q,
        )
        total = _job_count(status=status, site=site, min_score=min_score, max_score=max_score, q=q)
        return {"items": rows, "total": total, "limit": limit, "offset": offset}

    @app.get("/api/jobs/{url:path}")
    async def api_job_detail(url: str):
        conn = get_connection()
        row = conn.execute("SELECT * FROM jobs WHERE url = ?", (url,)).fetchone()
        if row is None:
            raise HTTPException(404, f"No job for url {url}")
        job = _row_to_dict(row)
        return {"job": job, "report": _read_report_for(job)}

    @app.get("/api/stats")
    async def api_stats():
        return get_stats()

    @app.get("/api/cost")
    async def api_cost():
        return _parse_cost_logs()

    @app.get("/api/models")
    async def api_models():
        return _models_payload().model_dump()

    @app.get("/api/sites")
    async def api_sites():
        """Distinct site list, used by the jobs filter dropdown."""
        conn = get_connection()
        rows = conn.execute(
            "SELECT site, COUNT(*) c FROM jobs WHERE site IS NOT NULL "
            "GROUP BY site ORDER BY c DESC"
        ).fetchall()
        return [{"site": r[0], "count": r[1]} for r in rows]

    # --- File downloads ---

    @app.get("/api/files/resume/{url:path}")
    async def api_resume(url: str):
        conn = get_connection()
        row = conn.execute(
            "SELECT tailored_resume_path FROM jobs WHERE url = ?", (url,)
        ).fetchone()
        if row is None or not row[0]:
            raise HTTPException(404, "No tailored resume for this job.")
        path = Path(row[0])
        # Prefer PDF when present.
        pdf = path.with_suffix(".pdf")
        if pdf.exists():
            return FileResponse(str(pdf), media_type="application/pdf", filename=pdf.name)
        if path.exists():
            return FileResponse(str(path), media_type="text/plain", filename=path.name)
        raise HTTPException(404, "Tailored resume file is missing on disk.")

    @app.get("/api/files/cover/{url:path}")
    async def api_cover(url: str):
        conn = get_connection()
        row = conn.execute(
            "SELECT cover_letter_path FROM jobs WHERE url = ?", (url,)
        ).fetchone()
        if row is None or not row[0]:
            raise HTTPException(404, "No cover letter for this job.")
        path = Path(row[0])
        pdf = path.with_suffix(".pdf")
        if pdf.exists():
            return FileResponse(str(pdf), media_type="application/pdf", filename=pdf.name)
        if path.exists():
            return FileResponse(str(path), media_type="text/plain", filename=path.name)
        raise HTTPException(404, "Cover letter file is missing on disk.")

    # --- Mutations: state changes that don't fight the auto-loop ---

    @app.post("/api/jobs/{url:path}/mark")
    async def api_mark(url: str, body: JobMarkRequest):
        conn = get_connection()
        row = conn.execute("SELECT url FROM jobs WHERE url = ?", (url,)).fetchone()
        if row is None:
            raise HTTPException(404, f"No job for url {url}")
        from datetime import timezone as _tz

        now = datetime.now(_tz.utc).isoformat()
        if body.status == "applied":
            conn.execute(
                "UPDATE jobs SET apply_status = 'applied', applied_at = ?, "
                "apply_error = NULL, agent_id = NULL WHERE url = ?",
                (now, url),
            )
        elif body.status == "failed":
            conn.execute(
                "UPDATE jobs SET apply_status = 'failed', apply_error = ?, agent_id = NULL "
                "WHERE url = ?",
                (body.reason or "manual", url),
            )
        elif body.status == "reset":
            conn.execute(
                "UPDATE jobs SET apply_status = NULL, apply_error = NULL, "
                "apply_attempts = 0, agent_id = NULL WHERE url = ?",
                (url,),
            )
        else:
            raise HTTPException(400, f"Unsupported status {body.status!r}")
        conn.commit()
        return {"ok": True, "status": body.status}

    @app.post("/api/jobs/{url:path}/retailor")
    async def api_retailor(url: str):
        argv = streams.build_retailor_argv(url)
        handle = streams.spawn("tailor", url, argv)
        return StreamHandleModel(stream_id=handle.stream_id, kind=handle.kind, url=handle.url).model_dump()

    @app.post("/api/jobs/{url:path}/reapply")
    async def api_reapply(url: str):
        argv = streams.build_reapply_argv(url)
        handle = streams.spawn("apply", url, argv)
        return StreamHandleModel(stream_id=handle.stream_id, kind=handle.kind, url=handle.url).model_dump()

    @app.post("/api/models/select")
    async def api_models_select(body: ModelSelectRequest):
        valid_ids = {m.id for m in _MODEL_OPTIONS}
        # Allow Custom… input but require provider/model shape.
        if body.model_id not in valid_ids and "/" not in body.model_id:
            raise HTTPException(400, "Custom model_id must include provider prefix (e.g. openai/gpt-4o-mini).")

        # Update os.environ so the in-process LLM client picks it up immediately.
        os.environ["LLM_MODEL"] = body.model_id

        # Reset the singleton so the next call rebuilds with the new config.
        try:
            import applypilot.llm as llm_mod

            llm_mod._instance = None
        except Exception:
            pass

        # Persist to ~/.applypilot/.env so it sticks across restarts.
        try:
            _persist_env_var("LLM_MODEL", body.model_id)
        except OSError as exc:
            log.warning("Could not persist LLM_MODEL to %s: %s", ENV_PATH, exc)

        return {"ok": True, "model_id": body.model_id}

    # --- SSE streams ---

    @app.get("/api/streams/{stream_id}")
    async def api_stream(stream_id: str):
        async def generator():
            async for evt in streams.stream_events(stream_id):
                yield {"event": evt["event"], "data": evt["data"]}

        return EventSourceResponse(generator())

    @app.get("/api/activity")
    async def api_activity():
        async def generator():
            try:
                async for evt in streams.tail_activity(LOG_DIR):
                    yield {"event": evt["event"], "data": evt["data"]}
            except asyncio.CancelledError:  # pragma: no cover
                return

        return EventSourceResponse(generator())

    @app.get("/api/health")
    async def api_health():
        return {
            "ok": True,
            "loop_active": _is_loop_active(),
            "app_dir": str(APP_DIR),
        }

    # --- Pipeline control: start / stop / status of the chain run+apply ---

    @app.get("/api/pipeline/status")
    async def api_pipeline_status():
        return streams.get_pipeline_status()

    @app.post("/api/pipeline/start")
    async def api_pipeline_start(request: Request):
        body: dict[str, Any] = {}
        try:
            body = await request.json()
        except Exception:
            body = {}
        mode = (body.get("mode") or "chain").strip().lower()
        if mode not in {"chain", "run", "apply"}:
            raise HTTPException(400, f"Invalid mode: {mode!r}. Use chain/run/apply.")
        try:
            handle = streams.start_pipeline(mode)
        except RuntimeError as exc:
            raise HTTPException(409, str(exc))
        return {
            "ok": True,
            "stream_id": handle.stream_id,
            "kind": handle.kind,
            "mode": mode,
        }

    @app.post("/api/pipeline/stop")
    async def api_pipeline_stop():
        killed = streams.stop_pipeline()
        return {"ok": True, "stopped": killed}

    @app.post("/api/streams/{stream_id}/stop")
    async def api_stream_stop(stream_id: str):
        killed = streams.kill_stream(stream_id)
        if not killed:
            raise HTTPException(404, "Stream not found or already finished.")
        return {"ok": True, "stopped": True}

    # --- Process tracker: list / kill rogue ApplyPilot subprocesses ---

    @app.get("/api/processes")
    async def api_processes():
        from applypilot.web import procs
        return {"items": [p.to_dict() for p in procs.list_processes()]}

    @app.post("/api/processes/{pid}/kill")
    async def api_process_kill(pid: int):
        from applypilot.web import procs
        result = procs.kill_pid(pid, force=True, include_children=True)
        if not result.get("ok"):
            raise HTTPException(400, result.get("error", "kill failed"))
        return result

    @app.post("/api/processes/kill-rogue-chrome")
    async def api_kill_rogue_chrome():
        from applypilot.web import procs
        return procs.kill_rogue_chrome()


# ---------------------------------------------------------------------------
# Helpers used only by the route handlers above.
# ---------------------------------------------------------------------------


_LLM_MODEL_LINE = re.compile(r"^LLM_MODEL=.*$", re.MULTILINE)


def _persist_env_var(key: str, value: str) -> None:
    """Idempotently update KEY=VALUE in ~/.applypilot/.env."""
    ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
    existing = ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.exists() else ""
    line = f"{key}={value}"
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
    if pattern.search(existing):
        new = pattern.sub(line, existing)
    else:
        new = existing.rstrip() + ("\n" if existing.strip() else "") + line + "\n"
    ENV_PATH.write_text(new, encoding="utf-8")
