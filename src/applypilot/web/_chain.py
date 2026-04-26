"""Fallback chain runner for the UI's "Start pipeline" button.

When `/tmp/full_chain.sh` isn't available (Windows, fresh checkouts), the
streams.start_pipeline() path invokes this module as a subprocess. It runs
`applypilot run` followed by `applypilot apply --limit 100` sequentially and
streams stdout to its own stdout so the UI's SSE pane forwards everything.

Usage: `python -m applypilot.web._chain`
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime


def _section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}  [{datetime.now().strftime('%H:%M:%S')}]")
    print(f"{'=' * 70}", flush=True)


def main() -> int:
    _section("STAGE 1: applypilot run --workers 4")
    rc1 = subprocess.call([sys.executable, "-m", "applypilot", "run", "--workers", "4"])
    if rc1 != 0:
        _section(f"STAGE 1 exited rc={rc1}, skipping apply")
        return rc1

    _section("STAGE 2: applypilot apply --limit 100")
    rc2 = subprocess.call([sys.executable, "-m", "applypilot", "apply", "--limit", "100"])
    _section(f"CHAIN DONE  (rc1={rc1}, rc2={rc2})")
    return rc2


if __name__ == "__main__":
    sys.exit(main())
