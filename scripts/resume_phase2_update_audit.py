"""Resume only the independent update audit of an interrupted EB-Hab run."""
from __future__ import annotations

import argparse
import json
import os
import uuid
from argparse import Namespace
from pathlib import Path

from vista_skill.config import load_config
from vista_skill.integrations.embodiedbench.cli import (
    _audit_rollout_seeds,
    _load_verified_manifest,
    _make_audit_evaluator,
)
from vista_skill.lineage import LineageStore
from vista_skill.update_audit import make_rotated_audit_plan, run_rotated_update_audit


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest", default="configs/eb_hab_train_validation_manifest.json"
    )
    parser.add_argument("--config", default="configs/vista_fault_repair_fullsel_p10.json")
    parser.add_argument("--executor-base-url", required=True)
    return parser.parse_args()


def _load_protocol(run_dir: Path) -> dict[str, object]:
    lineage_path = run_dir / "lineage.jsonl"
    records = [
        json.loads(line)
        for line in lineage_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    protocols = {json.dumps(record["protocol"], sort_keys=True) for record in records}
    if len(protocols) != 1:
        raise ValueError("lineage does not contain one consistent run protocol")
    return json.loads(next(iter(protocols)))


def main() -> int:
    args = _args()
    output = args.run_dir / "update_audit.json"
    if output.exists():
        raise FileExistsError(f"completed update audit already exists: {output}")
    protocol = _load_protocol(args.run_dir)
    config = load_config(args.config)
    manifest = _load_verified_manifest(args.manifest)
    if protocol.get("config_sha256") != config.digest:
        raise ValueError("resume config digest differs from lineage")
    if protocol.get("manifest_sha256") != manifest.digest:
        raise ValueError("resume manifest digest differs from lineage")

    evolution_seed = int(protocol["evolution_seed"])
    rotation_index = int(protocol["split_rotation_index"])
    plan = make_rotated_audit_plan(
        manifest,
        rotation_index=rotation_index,
        rollout_seeds=_audit_rollout_seeds(evolution_seed),
    )
    if protocol.get("split_sha256") != plan.split_sha256:
        raise ValueError("resume split digest differs from lineage")

    os.environ["remote_url"] = args.executor_base_url
    evaluator_args = Namespace(
        env="eb-hab",
        model_name=str(protocol["executor_model"]),
        model_type=str(protocol["executor_model_type"]),
        n_shots=int(protocol["n_shots"]),
        resolution=int(protocol["resolution"]),
        tp=int(protocol["tensor_parallel"]),
    )
    rotated = manifest.rotate_split(rotation_index)
    resume_run_id = f"{protocol['run_id']}_resume_{uuid.uuid4().hex[:8]}"
    evaluator = _make_audit_evaluator(
        evaluator_args,
        rotated,
        plan.coordinates,
        args.run_dir,
        run_id=resume_run_id,
    )
    report = run_rotated_update_audit(
        LineageStore(args.run_dir / "lineage.jsonl").accepted_snapshots,
        evaluator,
        plan=plan,
        output_path=output,
    )
    print(
        f"completed update audit: updates={report.update_count} "
        f"pairs={report.pair_count} reliability={dict(report.reliability)}"
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
