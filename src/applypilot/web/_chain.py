"""Fallback chain runner for the UI's "Start pipeline" button.

When `/tmp/full_chain.sh` isn't available (Windows, fresh checkouts), the
streams.start_pipeline() path invokes this module as a subprocess. It runs
`applypilot run` followed by `applypilot apply --limit 100` sequentially.

Output is dual-piped: to stdout (so the UI's StreamHandle picks it up) AND
to a timestamped log file in `~/.applypilot/logs/` so the progress panel
+ activity tailer can find it across UI restarts.

Usage: `python -m applypilot.web._chain`
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _log_path() -> Path:
    """Pick a fresh log filename so the activity tailer's mtime sort finds us."""
    log_dir = Path.home() / ".applypilot" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    return log_dir / f"chain-ui-{ts}.log"


def _section(log, title: str) -> None:
    line = f"\n{'=' * 70}\n  {title}  [{datetime.now().strftime('%H:%M:%S')}]\n{'=' * 70}"
    print(line, flush=True)
    log.write(line + "\n")
    log.flush()


def _run_streamed(argv: list[str], log) -> int:
    """Run a subprocess, tee each line to stdout AND the log file."""
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"},
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        log.write(line)
        log.flush()
    return proc.wait()


def main() -> int:
    log_file = _log_path()
    print(f"[chain] log: {log_file}", flush=True)
    with log_file.open("w", encoding="utf-8") as log:
        log.write(f"[chain] started {datetime.now().isoformat()}\n[chain] log: {log_file}\n")
        log.flush()

        _section(log, "STAGE 1: applypilot run --workers 4")
        rc1 = _run_streamed([sys.executable, "-m", "applypilot", "run", "--workers", "4"], log)
        if rc1 != 0:
            _section(log, f"STAGE 1 exited rc={rc1}, skipping apply")
            log.write(f"[chain] finished {datetime.now().isoformat()} rc={rc1}\n")
            return rc1

        _section(log, "STAGE 2: applypilot apply --limit 100")
        rc2 = _run_streamed([sys.executable, "-m", "applypilot", "apply", "--limit", "100"], log)
        _section(log, f"CHAIN DONE  (rc1={rc1}, rc2={rc2})")
        log.write(f"[chain] finished {datetime.now().isoformat()} rc1={rc1} rc2={rc2}\n")
        return rc2


if __name__ == "__main__":
    sys.exit(main())
