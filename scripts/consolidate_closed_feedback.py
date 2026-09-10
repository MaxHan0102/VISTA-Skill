"""Copy accepted legacy resume episodes into the canonical native layout.

Keep original attempts and API audits as provenance. Archive interrupted native
artifacts before installing a complete episode; publish its result file last.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import fcntl
import json
from pathlib import Path
import shutil
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.resume_closed_source_wo_feedback import (
    check_config, checkpoint, datasets, read, refresh_native_summaries, scan, sha, write,
)
from scripts.evaluate_closed_loop_feedback import allocate_output


def episode_files(root, split, ordinal):
    folder = root / split
    paths = list(folder.glob(f"episode_{ordinal}_*"))
    paths += list((folder / "images" / f"episode_{ordinal}").rglob("*"))
    paths += list((folder / "video").glob(f"video_episode_{ordinal}_*"))
    paths += list((folder / "results").glob(f"episode_{ordinal}_*"))
    return sorted(p for p in paths if p.is_file())


def copy_checked(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha(source) != sha(destination):
            raise ValueError(f"refusing to overwrite different file: {destination}")
        return
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst)
    if sha(source) != sha(destination):
        raise ValueError(f"copy verification failed: {destination}")


def consolidate(root, *, dry_run=False):
    root = root.resolve()
    if REPO / "running" not in root.parents:
        raise ValueError("run must be under repository running/")
    config = read(root / "config.json")
    check_config(config)
    with (root / ".resume.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ranges, hashes = datasets(config)
        state = read(root / "resume_state.json") if (root / "resume_state.json").exists() else {}
        if state and state["dataset_hashes"] != hashes:
            raise ValueError("dataset changed since checkpoint")
        records, identities, partial = scan(root, config, ranges, hashes, state.get("records"))
        incoming = {k: v for k, v in records.items() if Path(v["path"]).parents[2] != root}
        print(f"{len(records)} complete episodes; {len(incoming)} to consolidate", flush=True)
        if dry_run or not incoming:
            return records
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d%H%M")
        backup = allocate_output(REPO / "running/migrations" / f"closed_feedback_consolidation_{stamp}", stamp, create=True)
        for name in ("resume_state.json", "resume_summary.json", "resume_expected_episodes.json", "episode_provenance.json"):
            if (root / name).exists():
                copy_checked(root / name, backup / name)
        for path in root.glob("*/results/summary*.json"):
            copy_checked(path, backup / path.relative_to(root))
        provenance = read(root / "episode_provenance.json") if (root / "episode_provenance.json").exists() else {}
        plan = {"root": str(root), "status": "planned", "episodes": incoming, "copies": [], "archived": []}
        for key, item in incoming.items():
            split, number = key.split("/")
            origin = Path(item["path"]).parents[2]
            target_result = root / split / "results" / Path(item["path"]).name
            if target_result.exists():
                # A different complete result must never be replaced.
                try:
                    existing = read(target_result)
                except ValueError:
                    existing = {}
                if "task_success" in existing and sha(target_result) != item["sha256"]:
                    raise ValueError(f"conflicting complete result: {target_result}")
            for path in episode_files(root, split, int(number)):
                archived = root / "_interrupted" / backup.name / path.relative_to(root)
                plan["archived"].append({"source": str(path), "destination": str(archived), "sha256": sha(path)})
            paths = episode_files(origin, split, int(number))
            paths.sort(key=lambda p: p == Path(item["path"]))  # Result is the completion marker.
            for source in paths:
                target = root / source.relative_to(origin)
                plan["copies"].append({"source": str(source), "destination": str(target), "sha256": sha(source)})
            provenance[key] = {**item, "path": str(target_result), "source_path": item["path"],
                               "source_run": str(origin), "consolidation_manifest": str(backup / "manifest.json")}
        write(backup / "manifest.json", plan)
        for item in plan["archived"]:
            source, destination = Path(item["source"]), Path(item["destination"])
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise ValueError(f"archive already exists: {destination}")
            source.rename(destination)
        # Install provenance before completion markers, so interrupted copies are recoverable.
        write(root / "episode_provenance.json", provenance)
        for item in plan["copies"]:
            copy_checked(Path(item["source"]), Path(item["destination"]))
        for key in incoming:
            records[key] = provenance[key]
        records, identities, partial = scan(root, config, ranges, hashes, records)
        checkpoint(root, config, ranges, hashes, records, identities, partial)
        refresh_native_summaries(root, config, records)
        plan["status"] = "complete"
        write(backup / "manifest.json", plan)
        print(f"Consolidated {len(incoming)} episodes; manifest: {backup / 'manifest.json'}", flush=True)
        return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    consolidate(args.run_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
