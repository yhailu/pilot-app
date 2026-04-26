"""Process inventory + kill helpers for the web UI.

Tracks Chrome/Python/Node/Claude subprocesses that ApplyPilot itself
spawned (apply workers, retailor jobs, MCP servers, etc.) so the UI can
show a live list and let the user kill rogue ones — useful when a
Playwright Chrome window gets stuck and won't close.

Filters to ApplyPilot-related processes only, so we never accidentally
list / kill the user's regular browser, IDE, or other Python work.
"""

from __future__ import annotations

import logging
import os
import platform
import time
from dataclasses import dataclass
from typing import Any

import psutil

log = logging.getLogger(__name__)

# Substrings in cmdline that mark a process as ApplyPilot-related.
# Anything matching at least one of these is fair game for the UI list.
_APPLYPILOT_MARKERS = (
    "applypilot",                          # python -m applypilot ...
    "apply-workers",                       # Playwright chrome user-data-dir
    "chrome-workers",
    ".applypilot",                         # ~/.applypilot path leak
    "@playwright/mcp",                     # Playwright MCP server
    "@gongrzhe/server-gmail",              # Gmail MCP server
    "remote-debugging-port=9222",          # auto-apply CDP port
    "remote-debugging-port=9223",
    "remote-debugging-port=9224",
    "remote-debugging-port=9225",
)

# Names we care about (lowercased). Other binaries we ignore even if
# their cmdline matches a marker, to keep the list focused.
_TRACKED_NAMES = {
    "python.exe", "python", "python3",
    "chrome.exe", "chrome", "chromium", "chromium.exe",
    "node.exe", "node",
    "claude.exe", "claude.cmd", "claude",
    "uvicorn", "uvicorn.exe",
    "playwright.exe", "playwright",
}


@dataclass
class ProcInfo:
    pid: int
    name: str
    role: str             # "chrome" | "python" | "node" | "claude" | "other"
    cmdline_short: str    # truncated, no secrets
    cpu_pct: float
    mem_mb: float
    age_s: int
    parent_pid: int | None
    is_self: bool         # True if this is the UI server process itself

    def to_dict(self) -> dict[str, Any]:
        return {
            "pid": self.pid,
            "name": self.name,
            "role": self.role,
            "cmdline": self.cmdline_short,
            "cpu_pct": round(self.cpu_pct, 1),
            "mem_mb": round(self.mem_mb, 1),
            "age_s": self.age_s,
            "parent_pid": self.parent_pid,
            "is_self": self.is_self,
        }


def _classify(name: str, cmdline: str) -> str:
    nm = name.lower()
    if "chrome" in nm or "chromium" in nm:
        return "chrome"
    if nm.startswith("python"):
        return "python"
    if nm.startswith("node") or "playwright" in nm:
        return "node"
    if "claude" in nm:
        return "claude"
    return "other"


def _is_applypilot_proc(name: str, cmdline: str) -> bool:
    nm = name.lower()
    # startswith match handles python3.13.exe, python3.11, etc.
    name_match = (
        nm in _TRACKED_NAMES
        or nm.startswith("python")
        or nm.startswith("chrome")
        or nm.startswith("chromium")
        or nm.startswith("node")
        or nm.startswith("claude")
        or nm.startswith("uvicorn")
    )
    if not name_match:
        return False
    cl = cmdline.lower()
    return any(marker in cl for marker in _APPLYPILOT_MARKERS)


def _redact(cmdline: str) -> str:
    """Drop API keys / passwords from cmdline before shipping to the UI."""
    out = cmdline
    for needle in ("API_KEY=", "api-key=", "password=", "PASSWORD=", "token="):
        idx = out.lower().find(needle.lower())
        if idx == -1:
            continue
        # Replace value up to next whitespace
        end = out.find(" ", idx + len(needle))
        if end == -1:
            end = len(out)
        out = out[:idx + len(needle)] + "***" + out[end:]
    return out[:240]


def list_processes() -> list[ProcInfo]:
    """Return all ApplyPilot-related processes currently alive."""
    self_pid = os.getpid()
    out: list[ProcInfo] = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "create_time", "ppid"]):
        try:
            info = proc.info or {}
            name = (info.get("name") or "") or proc.name() or ""
            cmdline_list = info.get("cmdline") or []
            cmdline = " ".join(cmdline_list) if isinstance(cmdline_list, list) else str(cmdline_list)
            if not _is_applypilot_proc(name, cmdline):
                continue
            try:
                cpu = proc.cpu_percent(interval=0)
            except Exception:
                cpu = 0.0
            try:
                mem_mb = proc.memory_info().rss / (1024 * 1024)
            except Exception:
                mem_mb = 0.0
            create_time = info.get("create_time") or 0
            age = max(0, int(time.time() - create_time))
            out.append(ProcInfo(
                pid=proc.pid,
                name=name,
                role=_classify(name, cmdline),
                cmdline_short=_redact(cmdline),
                cpu_pct=cpu,
                mem_mb=mem_mb,
                age_s=age,
                parent_pid=info.get("ppid"),
                is_self=(proc.pid == self_pid),
            ))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    # Newest first; chrome and rogue claude tend to be the interesting ones.
    out.sort(key=lambda p: (p.role != "chrome", -p.age_s))
    return out


def kill_pid(pid: int, *, force: bool = True, include_children: bool = True) -> dict[str, Any]:
    """Kill a process by pid, optionally with its children. Refuses to kill self."""
    self_pid = os.getpid()
    if pid == self_pid:
        return {"ok": False, "error": "refusing to kill the UI server itself", "pid": pid}

    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return {"ok": False, "error": "no such process", "pid": pid}

    # Only allow killing if the process is actually ApplyPilot-related — defends
    # against pasting a random pid that would terminate the user's regular Chrome.
    try:
        cmdline = " ".join(proc.cmdline() or [])
        name = proc.name()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return {"ok": False, "error": "cannot inspect process", "pid": pid}

    if not _is_applypilot_proc(name, cmdline):
        return {
            "ok": False,
            "error": "process is not ApplyPilot-related; refusing to kill",
            "pid": pid,
            "name": name,
        }

    killed: list[int] = []
    targets: list[psutil.Process] = []
    if include_children:
        try:
            targets.extend(proc.children(recursive=True))
        except psutil.Error:
            pass
    targets.append(proc)

    for p in targets:
        try:
            if force:
                p.kill()
            else:
                p.terminate()
            killed.append(p.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            log.warning("Could not kill pid=%s: %s", p.pid, exc)
    return {"ok": True, "pid": pid, "killed": killed}


def kill_rogue_chrome() -> dict[str, Any]:
    """Kill all ApplyPilot-spawned Chrome processes in one shot."""
    killed_pids: list[int] = []
    errors: list[str] = []
    for info in list_processes():
        if info.role != "chrome" or info.is_self:
            continue
        result = kill_pid(info.pid, force=True, include_children=True)
        if result.get("ok"):
            killed_pids.extend(result.get("killed", []))
        else:
            errors.append(f"pid={info.pid}: {result.get('error')}")
    return {"ok": True, "killed": killed_pids, "errors": errors}
