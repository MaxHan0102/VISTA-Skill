"""Run one Phase3A stage with an append-only console log and runtime manifest.

This wrapper is intended to be launched inside a detached tmux session.  It
records enough process metadata to diagnose interrupted overnight experiments
without changing the experiment command itself.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_tree_sha256() -> str:
    digest = hashlib.sha256()
    roots = (Path("vista_skill"), Path("scripts"))
    files = sorted(
        path
        for root in roots
        if root.exists()
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    for path in files:
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--console-log", type=Path, required=True)
    parser.add_argument("--endpoint", action="append", default=[])
    parser.add_argument("--config-file", type=Path, action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    return args


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    args = _args()
    for target in (args.runtime_manifest, args.console_log):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
    config_hashes = {
        str(path): _sha256(path) for path in args.config_file if path.is_file()
    }
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
    try:
        git_status = subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).splitlines()
        git_diff = subprocess.check_output(
            ["git", "diff", "--binary", "--", "."],
        )
        git_diff_sha256 = hashlib.sha256(git_diff).hexdigest()
    except (OSError, subprocess.CalledProcessError):
        git_status = ["unavailable"]
        git_diff_sha256 = "unknown"
    manifest: dict[str, Any] = {
        "stage": args.stage,
        "status": "starting",
        "started_at": _now(),
        "finished_at": None,
        "wrapper_pid": os.getpid(),
        "child_pid": None,
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "git_commit": git_commit,
        "git_status": git_status,
        "git_diff_sha256": git_diff_sha256,
        "source_tree_sha256": _source_tree_sha256(),
        "endpoints": list(args.endpoint),
        "config_sha256": config_hashes,
        "command": list(args.command),
        "console_log": str(args.console_log),
        "exit_code": None,
    }
    _write(args.runtime_manifest, manifest)
    with args.console_log.open("x", encoding="utf-8", buffering=1) as log:
        def emit(line: str) -> None:
            stamped = f"[{_now()}] {line}"
            print(stamped, flush=True)
            log.write(stamped + "\n")

        emit(f"START stage={args.stage}")
        emit("COMMAND " + json.dumps(args.command, ensure_ascii=False))
        process = subprocess.Popen(
            args.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        manifest["status"] = "running"
        manifest["child_pid"] = process.pid
        _write(args.runtime_manifest, manifest)
        assert process.stdout is not None
        for line in process.stdout:
            clean = line.rstrip("\n")
            print(clean, flush=True)
            log.write(clean + "\n")
        exit_code = process.wait()
        emit(f"END stage={args.stage} exit_code={exit_code}")
    manifest["status"] = "completed" if exit_code == 0 else "failed"
    manifest["finished_at"] = _now()
    manifest["exit_code"] = exit_code
    _write(args.runtime_manifest, manifest)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
