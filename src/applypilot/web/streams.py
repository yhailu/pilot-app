"""Subprocess and log streaming helpers for the web UI.

We never call into the apply/tailor pipelines inline — the auto-loop
that runs in another process holds the SQLite writer lock at unpredictable
times, and Playwright/Claude Code processes need their own stdio. So
re-tailor and re-apply endpoints spawn `python -m applypilot ...` (or
`python -m applypilot.web._tailor_one`) subprocesses and the UI tails
their stdout/stderr via Server-Sent Events.

The `parse_claude_jsonl_line` helper turns Claude Code stream-json log
lines into compact `▸ tool_name "arg"` action strings so the live pane
in the browser stays human-readable instead of dumping raw JSON.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import platform
import shlex
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

log = logging.getLogger(__name__)

# Windows process group flag — allows Ctrl+C-style termination of the whole tree.
_CREATE_NEW_PROCESS_GROUP = 0x00000200 if platform.system() == "Windows" else 0


@dataclass
class StreamHandle:
    """Tracks one in-flight subprocess + ring buffer of recent lines."""

    stream_id: str
    kind: str  # "tailor" | "apply"
    url: str
    proc: subprocess.Popen
    started_at: float
    buffer: list[dict] = field(default_factory=list)  # last ~500 events
    done: bool = False
    rc: int | None = None


# Module-level registry. Personal-tool scope, single-process server, so a plain
# dict guarded by a lock is enough — no need for Redis.
_STREAMS: dict[str, StreamHandle] = {}
_STREAMS_LOCK = threading.Lock()
_BUFFER_MAX = 500


def _parse_claude_action(raw: str) -> dict | None:
    """If `raw` is Claude Code stream-json with a tool_use, return a compact action dict.

    Claude's stream-json emits one JSON object per line. We surface only the
    tool_use entries because they correspond to visible browser/agent steps.
    """
    raw = raw.strip()
    if not raw or not raw.startswith("{"):
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None

    msg = obj.get("message") or obj
    content = msg.get("content")
    if not isinstance(content, list):
        return None
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_use":
            continue
        tool = block.get("name", "tool")
        tool_input = block.get("input") or {}
        # Pick the most descriptive single field for the summary.
        snippet = ""
        for key in ("query", "url", "selector", "text", "description", "command"):
            val = tool_input.get(key)
            if isinstance(val, str) and val.strip():
                snippet = val.strip()
                break
        if not snippet and tool_input:
            try:
                snippet = json.dumps(tool_input)[:120]
            except TypeError:
                snippet = str(tool_input)[:120]
        return {"tool": tool, "summary": snippet}
    return None


def parse_claude_jsonl_line(raw: str) -> dict:
    """Convert one log line into an SSE-ready event payload.

    Returns a dict with `event` ('action' or 'line') and `data` (str).
    """
    action = _parse_claude_action(raw)
    if action:
        formatted = f"▸ {action['tool']} {json.dumps(action['summary'])}" if action["summary"] else f"▸ {action['tool']}"
        return {"event": "action", "data": formatted}
    return {"event": "line", "data": raw.rstrip("\r\n")}


def _drain_pipe(handle: StreamHandle, pipe, label: str) -> None:
    """Background thread: read subprocess output line-by-line into the handle's buffer."""
    try:
        for raw in iter(pipe.readline, b""):
            try:
                text = raw.decode("utf-8", errors="replace")
            except Exception:
                text = repr(raw)
            event = parse_claude_jsonl_line(text)
            with _STREAMS_LOCK:
                handle.buffer.append(event)
                if len(handle.buffer) > _BUFFER_MAX:
                    del handle.buffer[: len(handle.buffer) - _BUFFER_MAX]
    except Exception as exc:  # pragma: no cover — pipe close races
        log.debug("%s reader for %s ended: %s", label, handle.stream_id, exc)
    finally:
        try:
            pipe.close()
        except Exception:
            pass


def _wait_proc(handle: StreamHandle) -> None:
    """Wait for the subprocess and mark the handle done."""
    rc = handle.proc.wait()
    with _STREAMS_LOCK:
        handle.rc = rc
        handle.done = True
        handle.buffer.append({"event": "done", "data": json.dumps({"rc": rc})})


def spawn(kind: str, url: str, argv: list[str]) -> StreamHandle:
    """Launch a subprocess in a new process group and register a stream handle."""
    stream_id = uuid.uuid4().hex[:12]
    log.info("Spawning %s stream %s: %s", kind, stream_id, " ".join(shlex.quote(a) for a in argv))

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")

    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        creationflags=_CREATE_NEW_PROCESS_GROUP,
        env=env,
    )

    handle = StreamHandle(
        stream_id=stream_id,
        kind=kind,
        url=url,
        proc=proc,
        started_at=time.time(),
    )
    with _STREAMS_LOCK:
        _STREAMS[stream_id] = handle

    threading.Thread(target=_drain_pipe, args=(handle, proc.stdout, "stdout"), daemon=True).start()
    threading.Thread(target=_wait_proc, args=(handle,), daemon=True).start()
    return handle


def get_stream(stream_id: str) -> StreamHandle | None:
    with _STREAMS_LOCK:
        return _STREAMS.get(stream_id)


def shutdown_all_streams() -> None:
    """Best-effort kill on shutdown so the dev server can restart cleanly."""
    with _STREAMS_LOCK:
        handles = list(_STREAMS.values())
    for h in handles:
        if not h.done:
            try:
                h.proc.terminate()
            except Exception:
                pass


async def stream_events(stream_id: str) -> AsyncIterator[dict]:
    """Async generator yielding SSE events for a stream until it finishes."""
    handle = get_stream(stream_id)
    if handle is None:
        yield {"event": "error", "data": "unknown stream"}
        return

    cursor = 0
    while True:
        with _STREAMS_LOCK:
            pending = handle.buffer[cursor:]
            cursor = len(handle.buffer)
            done = handle.done

        for evt in pending:
            yield evt
            if evt["event"] == "done":
                return

        if done and cursor >= len(handle.buffer):
            return

        await asyncio.sleep(0.2)


# ---------------------------------------------------------------------------
# Activity log tailer — newest run/chain/auto-loop/apply log under LOG_DIR.
# ---------------------------------------------------------------------------


def _newest_log(log_dir: Path) -> Path | None:
    if not log_dir.exists():
        return None
    candidates = []
    for pattern in ("run-*.log", "chain-*.log", "auto-loop-*.log", "apply-*.log", "worker-*.log"):
        candidates.extend(log_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


async def tail_activity(log_dir: Path, history_lines: int = 200) -> AsyncIterator[dict]:
    """Yield lines from the freshest log under LOG_DIR, then follow it."""
    path = _newest_log(log_dir)
    if path is None:
        yield {"event": "line", "data": f"no activity logs in {log_dir}"}
        return

    yield {"event": "line", "data": f"-- tailing {path.name} --"}

    # Replay tail of file
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)
            size = f.tell()
            block = max(0, size - 64 * 1024)
            f.seek(block)
            tail = f.read().splitlines()[-history_lines:]
            for line in tail:
                yield parse_claude_jsonl_line(line)
            position = f.tell()
    except FileNotFoundError:
        return

    # Follow: poll for new bytes
    while True:
        await asyncio.sleep(1.0)
        try:
            current = path.stat().st_size
        except FileNotFoundError:
            return
        if current < position:
            position = 0  # log was rotated
        if current > position:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(position)
                for line in f:
                    if line.endswith("\n"):
                        yield parse_claude_jsonl_line(line)
                position = f.tell()


# ---------------------------------------------------------------------------
# Entry-point command builders for retailor / reapply spawns.
# ---------------------------------------------------------------------------


def build_retailor_argv(url: str) -> list[str]:
    """Argv for `python -m applypilot.web._tailor_one --url <url>`."""
    return [sys.executable, "-m", "applypilot.web._tailor_one", "--url", url]


def build_reapply_argv(url: str) -> list[str]:
    """Argv for `python -m applypilot apply --url <url> --limit 1`.

    The launcher already respects the in_progress lock so it cooperates
    with the running auto-loop instead of fighting it.
    """
    return [sys.executable, "-m", "applypilot", "apply", "--url", url, "--limit", "1"]


# ---------------------------------------------------------------------------
# Pipeline control — start/stop the full chain (run + apply) from the UI.
# A reserved stream_id ("pipeline-main") gives us a singleton we can find
# without bookkeeping IDs across HTTP requests. Multiple chains can't run
# in parallel from the UI; that would just thrash the DB and Anthropic budget.
# ---------------------------------------------------------------------------

PIPELINE_STREAM_ID = "pipeline-main"


def kill_stream(stream_id: str) -> bool:
    """Terminate a running subprocess by stream_id. Returns True if killed."""
    handle = get_stream(stream_id)
    if handle is None or handle.done:
        return False
    try:
        if platform.system() == "Windows":
            # CTRL_BREAK_EVENT works because we spawned with CREATE_NEW_PROCESS_GROUP.
            handle.proc.send_signal(subprocess.signal.CTRL_BREAK_EVENT)
        else:
            handle.proc.terminate()
    except Exception:
        try:
            handle.proc.kill()
        except Exception:
            return False
    return True


def get_pipeline_status() -> dict:
    """Return status dict for the singleton pipeline stream."""
    handle = get_stream(PIPELINE_STREAM_ID)
    if handle is None:
        return {"running": False, "stream_id": None}
    return {
        "running": not handle.done,
        "stream_id": handle.stream_id,
        "kind": handle.kind,
        "started_at": handle.started_at,
        "elapsed_s": time.time() - handle.started_at,
        "rc": handle.rc,
        "done": handle.done,
        "buffer_size": len(handle.buffer),
    }


def start_pipeline(mode: str = "chain") -> StreamHandle:
    """Start the singleton pipeline stream.

    mode='chain': run + skip-filter + apply (the bash chain script if present,
                  otherwise sequential `applypilot run` then `applypilot apply`).
    mode='run':   just `applypilot run --workers 4` (discover/tailor/score, no apply).
    mode='apply': just `applypilot apply --limit 100`.
    """
    existing = get_stream(PIPELINE_STREAM_ID)
    if existing and not existing.done:
        raise RuntimeError(
            f"Pipeline already running (stream_id={existing.stream_id}, "
            f"started {int(time.time() - existing.started_at)}s ago). Stop it first."
        )

    if mode == "chain":
        # Prefer the existing chain script if it's there; else fall back to inline.
        chain_sh = Path("/tmp/full_chain.sh")
        if chain_sh.exists() and platform.system() != "Windows":
            argv = ["bash", str(chain_sh)]
        else:
            # Use a Python entry point that runs both stages sequentially.
            argv = [sys.executable, "-m", "applypilot.web._chain"]
    elif mode == "run":
        argv = [sys.executable, "-m", "applypilot", "run", "--workers", "4"]
    elif mode == "apply":
        argv = [sys.executable, "-m", "applypilot", "apply", "--limit", "100"]
    else:
        raise ValueError(f"Unknown pipeline mode: {mode!r}")

    # Reserve the singleton id by spawning with the standard helper, then
    # rebinding under the well-known id so subsequent callers can find it.
    handle = spawn(f"pipeline-{mode}", "(pipeline)", argv)
    with _STREAMS_LOCK:
        _STREAMS.pop(handle.stream_id, None)
        handle.stream_id = PIPELINE_STREAM_ID
        _STREAMS[PIPELINE_STREAM_ID] = handle
    return handle


def stop_pipeline() -> bool:
    """Terminate the singleton pipeline stream if running."""
    return kill_stream(PIPELINE_STREAM_ID)


# ---------------------------------------------------------------------------
# Pipeline progress parsing — convert the raw stdout buffer into a structured
# view the dashboard can render: current stage, current job, recent completions.
# ---------------------------------------------------------------------------

import re as _re

_STAGE_RE = _re.compile(r"STAGE:\s+([a-z]+)\s+[—-]")  # e.g. "STAGE: tailor — Resume tailoring..."
_PDF_RE = _re.compile(r"PDF generated:\s*(.+)$")
_APPLY_START_RE = _re.compile(r"\[W\d+\]\s+Starting:\s+(.+?)\s+@\s+(.+)$")
_APPLY_DONE_RE = _re.compile(r"\[W\d+\]\s+(APPLIED|EXPIRED|CAPTCHA|FAILED|NO RESULT|SSO|LOGIN)[\s\S]*?:\s*(.+?)(?:\s|$)")
_TAILORING_RE = _re.compile(r"Tailoring\s+(\d+)/(\d+)[:\s]\s*(.+)$", _re.IGNORECASE)
_SCORING_RE = _re.compile(r"Scoring\s+(\d+)/(\d+)[:\s]\s*(.+)$", _re.IGNORECASE)

# Per-stage activity hints — pulled from real log line shapes.
_STAGE_HINTS = (
    # discover
    (_re.compile(r"\[([^\]]+)\]\s+Done:\s+(\d+)\s+found,\s+(\d+)\s+new"),
     lambda m: f"Discover [{m.group(1)}]: found {m.group(2)}, {m.group(3)} new"),
    (_re.compile(r"^([\w\s.&/'-]+):\s+searching\s+\"([^\"]+)\""),
     lambda m: f"Searching {m.group(1).strip()}: \"{m.group(2)}\""),
    (_re.compile(r"^([\w\s.&/'-]+):\s+(\d+)\s+total results"),
     lambda m: f"{m.group(1).strip()}: {m.group(2)} total results"),
    (_re.compile(r"^([\w\s.&/'-]+):\s+(\d+)\s+jobs found"),
     lambda m: f"{m.group(1).strip()}: {m.group(2)} jobs matched filter"),
    (_re.compile(r"JobRight:\s+found\s+(\d+)\s+cards"),
     lambda m: f"JobRight: scraped {m.group(1)} cards"),
    (_re.compile(r"JobRight new:\s+(.+)$"),
     lambda m: f"JobRight: + {m.group(1)}"),
    (_re.compile(r"JobRight scrape complete:\s+(.+)$"),
     lambda m: f"JobRight done — {m.group(1)}"),
    (_re.compile(r"Full crawl:\s+(\d+)\s+search combinations"),
     lambda m: f"JobSpy: starting full crawl, {m.group(1)} search combinations"),
    (_re.compile(r"Sites:\s+(.+?)\s+\|"),
     lambda m: f"JobSpy: querying {m.group(1)}"),
    (_re.compile(r"Smart-extract:\s+(.+?)$"),
     lambda m: f"Smart-extract: {m.group(1)}"),
    # enrich
    (_re.compile(r"Enriching\s+(\d+)/(\d+):\s*(.+)$"),
     lambda m: f"Enrich {m.group(1)}/{m.group(2)}: {m.group(3)}"),
    (_re.compile(r"detail\.py.*Fetched\s+(.+?)\s"),
     lambda m: f"Enrich: fetched {m.group(1)}"),
    # score
    (_re.compile(r"Scored:\s+(\d+)\s+jobs"),
     lambda m: f"Score: {m.group(1)} jobs scored"),
    (_re.compile(r"Done:\s+(\d+)\s+scored in"),
     lambda m: f"Score: {m.group(1)} jobs scored, stage complete"),
    # general subprocess starts
    (_re.compile(r"^\s*Started thread:\s+(\w+)"),
     lambda m: f"Started thread: {m.group(1)}"),
    (_re.compile(r"^\s*JobSpy full crawl"),
     lambda m: "JobSpy: launching full crawl…"),
    (_re.compile(r"\bChrome started on port (\d+).*pid (\d+)"),
     lambda m: f"Chrome worker started on port {m.group(1)} (pid {m.group(2)})"),
)

# Lines we consider noise — don't show these as activity / don't include in tail.
_NOISE_RE = _re.compile(
    r"^\s*$"                                   # blank
    r"|^\s*[┌└│├─=+|—-]+\s*$"                 # box-drawing / horizontal rules
    r"|^\s*\|.*\|\s*$"                         # table rows of just pipes
    r"|HTTP Request:"                          # litellm chatter
    r"|LiteLLM:"                               # internal noise
    r"|^[0-9:]{8}\s+-\s+INFO\s+-\s+selected\s+model\s+name"
    r"|^[0-9:]{8}\s+-\s+INFO\s+-\s+HTTP/"
    r"|Task was destroyed but it is pending"   # LiteLLM async cleanup warning (harmless)
    r"|RuntimeWarning:"                        # generic Python warnings
    r"|coroutine\s+'[^']+'\s+was never awaited"
    r"|Enable tracemalloc"                     # Python's tracemalloc hint that follows
)


def _pretty_from_pdf_path(path: str) -> dict:
    """Extract a readable {kind, name} from a PDF generated path."""
    p = path.replace("\\", "/").strip()
    base = p.rsplit("/", 1)[-1]
    if "/cover_letters/" in p or base.endswith("_CL.pdf") or base.endswith("_CL.txt"):
        kind = "cover_letter"
    else:
        kind = "tailored"
    name = base.rsplit(".", 1)[0]
    name = _re.sub(r"_[a-f0-9]{6}(_CL)?$", "", name)  # drop hash suffix
    name = name.replace("_", " ").strip()
    return {"kind": kind, "name": name}


def _events_from_latest_log() -> list[dict]:
    """Fallback: read the latest pipeline log file and synthesize events.

    Used when the in-memory StreamHandle is missing (e.g. UI restarted while
    the chain subprocess kept running in its own process group). Picks the
    newest log matching the pipeline-related prefixes by mtime, reads the
    last ~2000 lines, and runs each line through the same parser the live
    stream uses so callers can treat the result identically.
    """
    from applypilot.config import LOG_DIR

    candidates: list[Path] = []
    for prefix in ("chain-", "auto-loop-", "run-", "apply-"):
        candidates.extend(LOG_DIR.glob(f"{prefix}*.log"))
    if not candidates:
        return []
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    try:
        with latest.open("r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-2000:]
    except OSError:
        return []
    return [parse_claude_jsonl_line(line) for line in lines]


def get_progress(stream_id: str = PIPELINE_STREAM_ID, *, recent_limit: int = 15) -> dict:
    """Parse the most recent buffer of a stream into structured progress info."""
    handle = get_stream(stream_id)
    if handle is None:
        # Fall back to the most recent log file so progress survives UI restarts
        # and page reloads after the chain finishes.
        events = _events_from_latest_log()
        if not events:
            return {"running": False, "stream_id": stream_id, "events": 0,
                    "completed": [], "stage_counts": {}}
        # Continue with the parser using log-file events
        return _parse_events(events, recent_limit=recent_limit, running=False,
                             stream_id=stream_id, started_at=None, kind="pipeline-from-log")

    with _STREAMS_LOCK:
        events = list(handle.buffer)
    return _parse_events(events, recent_limit=recent_limit, running=not handle.done,
                         stream_id=stream_id, started_at=handle.started_at, kind=handle.kind)


def _parse_events(events: list[dict], *, recent_limit: int, running: bool,
                  stream_id: str, started_at: float | None, kind: str) -> dict:
    """Shared parser used by both live-buffer and log-file paths."""

    current_stage = None
    stage_started_idx = -1
    current_action = None
    completed: list[dict] = []
    stage_counts = {"tailored": 0, "cover_letter": 0, "applied": 0, "failed": 0, "scored": 0}
    recent_lines: list[str] = []  # last N non-noise log lines for the live tail

    for i, evt in enumerate(events):
        data = evt.get("data") or ""
        if not data:
            continue
        # Strip "HH:MM:SS - INFO - " or similar prefix for the tail display
        clean = _re.sub(r"^[0-9:]{8}\s+-\s+\w+\s+-\s+", "", data).strip()

        if not _NOISE_RE.search(clean):
            recent_lines.append(clean)
            if len(recent_lines) > 25:
                recent_lines = recent_lines[-25:]

        m = _STAGE_RE.search(data)
        if m:
            current_stage = m.group(1).lower()
            stage_started_idx = i
            current_action = f"Entering stage: {current_stage}"
            continue

        m = _PDF_RE.search(data)
        if m:
            info = _pretty_from_pdf_path(m.group(1))
            completed.append({**info, "at": _extract_ts(data)})
            stage_counts[info["kind"]] = stage_counts.get(info["kind"], 0) + 1
            current_action = f"Generated: {info['name']}"
            continue

        m = _APPLY_START_RE.search(data)
        if m:
            current_action = f"Applying to: {m.group(1)} @ {m.group(2)}"
            continue

        if "APPLIED " in data or " APPLIED" in data:
            completed.append({"kind": "applied", "name": _extract_after_label(data, "APPLIED"), "at": _extract_ts(data)})
            stage_counts["applied"] += 1
            current_action = None
            continue
        outcome_hit = False
        for label in ("EXPIRED", "CAPTCHA", "FAILED", "NO RESULT", "SSO", "LOGIN"):
            if f" {label} " in data or data.endswith(label):
                completed.append({"kind": "failed", "subkind": label.lower(), "name": _extract_after_label(data, label), "at": _extract_ts(data)})
                stage_counts["failed"] += 1
                current_action = None
                outcome_hit = True
                break
        if outcome_hit:
            continue

        m = _TAILORING_RE.search(data)
        if m:
            current_action = f"Tailoring {m.group(1)}/{m.group(2)}: {m.group(3)}"
            continue
        m = _SCORING_RE.search(data)
        if m:
            current_action = f"Scoring {m.group(1)}/{m.group(2)}: {m.group(3)}"
            continue

        # General per-stage activity hints — fills the gap during discover/score/etc.
        for pattern, formatter in _STAGE_HINTS:
            mm = pattern.search(clean)
            if mm:
                current_action = formatter(mm)
                break

    # Drop very old completions; show only the newest N
    completed_recent = completed[-recent_limit:][::-1]

    elapsed_s = (time.time() - started_at) if started_at else None
    # If no specific action matched, fall back to the most recent meaningful log line
    # so the user always has SOMETHING to look at instead of "starting…" forever.
    if not current_action and recent_lines:
        current_action = recent_lines[-1][:200]

    return {
        "running": running,
        "stream_id": stream_id,
        "kind": kind,
        "started_at": started_at,
        "elapsed_s": elapsed_s,
        "current_stage": current_stage,
        "current_action": current_action,
        "completed": completed_recent,
        "stage_counts": stage_counts,
        "recent_activity": recent_lines[-15:],
        "total_events": len(events),
    }


def _extract_ts(line: str) -> str | None:
    """Pull HH:MM:SS off the front of a log line if present."""
    m = _re.match(r"^(\d{2}:\d{2}:\d{2})", line)
    return m.group(1) if m else None


def _extract_after_label(line: str, label: str) -> str:
    """Pull the trailing job title after an apply outcome label."""
    idx = line.find(label)
    if idx == -1:
        return line.strip()[:80]
    after = line[idx + len(label):].strip(" :()")
    # Trim duration suffix and any worker tags
    after = _re.sub(r"\(\d+s\):?\s*", "", after)
    return after.strip()[:80]
