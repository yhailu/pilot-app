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
