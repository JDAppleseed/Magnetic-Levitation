#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from argparse import ArgumentParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _framework_python_candidate() -> Path | None:
    base_prefix = Path(sys.base_prefix)
    candidate = base_prefix / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
    return candidate if candidate.exists() else None


def _should_relaunch_via_python_app() -> bool:
    if sys.platform != "darwin":
        return False
    if os.environ.get("MAGLEV_UI_SKIP_RELAUNCH") == "1":
        return False
    executable = Path(sys.executable).resolve()
    return "Python.app/Contents/MacOS/Python" not in str(executable)


def _relaunch_via_python_app() -> None:
    candidate = _framework_python_candidate()
    if candidate is None:
        return
    env = os.environ.copy()
    entries = [str(ROOT / "src")]
    current_pythonpath = env.get("PYTHONPATH", "")
    if current_pythonpath:
        entries.extend(part for part in current_pythonpath.split(os.pathsep) if part)
    deduped: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if entry not in seen:
            seen.add(entry)
            deduped.append(entry)
    env["PYTHONPATH"] = os.pathsep.join(deduped)
    env["MAGLEV_UI_SKIP_RELAUNCH"] = "1"
    os.execve(str(candidate), [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]], env)


def _parse_args() -> object:
    parser = ArgumentParser(description="Run the magnetic levitation desktop UI.")
    parser.add_argument("--smoke-test", action="store_true", help="Create the UI, process events, and exit.")
    return parser.parse_args()


if __name__ == "__main__":
    _relaunch_via_python_app() if _should_relaunch_via_python_app() else None
    sys.path.insert(0, str(ROOT / "src"))
    args = _parse_args()
    from ui.app import main

    raise SystemExit(main(smoke_test=bool(args.smoke_test)))
