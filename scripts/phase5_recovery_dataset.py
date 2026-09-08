"""Freeze new goal combinations using only the released development layouts.

This does not create new scenes. Source layouts are historically exposed and
are disjoint across the four roles of this prospective diagnostic.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import random
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "EmbodiedBench/embodiedbench/envs/eb_habitat/datasets/train_validation.pickle"
RECEPTACLES = {
    "receptacle_aabb_Sofa_frl_apartment_sofa": "sofa",
    "receptacle_aabb_Tbl1_Top1_frl_apartment_table_01": "table 1",
    "receptacle_aabb_Tbl2_Top1_frl_apartment_table_02": "table 2",
    "receptacle_aabb_TvStnd1_Top1_frl_apartment_tvstand": "TV stand",
    "receptacle_aabb_counter_left_kitchen_counter": "left kitchen counter",
    "receptacle_aabb_counter_right_kitchen_counter": "right kitchen counter",
}


def build_dataset(source: dict, *, seed: int = 570907) -> tuple[dict, dict]:
    rng = random.Random(seed)
    eligible = [e for e in source["all_eps"] if (
        "target_object_name" in e["sampled_entities"]
        and "target_receptacle_name" in e["sampled_entities"]
        and any("holding(" in p for group in (e.get("subgoals") or []) for p in group)
    )]
    rng.shuffle(eligible)
    roles = {"acquisition": 12, "proxy": 8, "finalist": 8, "audit": 8}
    if len(eligible) < sum(roles.values()):
        raise ValueError("insufficient independent development source episodes")
    generated, coordinates = [], []
    layout_hashes = set()
    for role, count in roles.items():
        for index in range(count):
            base = eligible.pop()
            layout = hashlib.sha256(pickle.dumps((
                base["scene_id"],
                [(int(n), source["all_transforms"][int(t)].tolist())
                 for n, t in base["rigid_objs"]],
            ))).hexdigest()
            if layout in layout_hashes:
                raise ValueError("source layout duplicated across diagnostic tasks")
            layout_hashes.add(layout)
            entity = base["sampled_entities"]["target_object_name"]
            old_destination = base["sampled_entities"]["target_receptacle_name"]
            source_destination = base["sampled_entities"].get("source_receptacle_name")
            destinations = sorted(set(RECEPTACLES) - {old_destination, source_destination})
            destination = rng.choice(destinations)
            protected = role != "acquisition" and index >= count - 2
            episode = copy.deepcopy(base)
            episode_id = f"p57_{role}_{index:02d}"
            episode["episode_id"] = episode_id
            episode["start_preds"] = []
            episode["sampler_info"] = {}
            episode["sampled_entities"] = {}
            episode["instruct_id"] = "p57_nav" if protected else "p57_transfer"
            if protected:
                predicate = f"robot_at({destination})"
                episode["instruction"] = f"Navigate to the {RECEPTACLES[destination]}."
                episode["goal_preds"] = {"expr_type": "AND", "sub_exprs": [predicate]}
                episode["subgoals"] = [[predicate]]
            else:
                handles = [re.fullmatch(r"holding\(([^)]+)\)", p.replace(" ", ""))
                           for group in base["subgoals"] for p in group]
                handle = next(m.group(1) for m in handles if m)
                episode["instruction"] = f"Move a {entity} to the {RECEPTACLES[destination]}."
                episode["goal_preds"] = {
                    "expr_type": "AND", "quantifier": "EXISTS",
                    "inputs": [{"name": "X", "expr_type": entity}],
                    "sub_exprs": [f"on_top(X, {destination})", "not_holding()"],
                }
                episode["subgoals"] = [
                    [f"holding({handle})"],
                    [f"on_top({handle},{destination})", "not_holding()"],
                ]
            generated.append(episode)
            coordinates.append({
                "episode_id": episode_id, "role": role,
                "source_episode_id": str(base["episode_id"]),
                "source_layout_sha256": layout,
                "scope": "protected" if protected else "affected",
                "subgroup": "scene:" + Path(base["scene_id"]).stem,
                # Pair members and every seed for a task stay on one endpoint.
                "endpoint_index": index % 2,
                "goal_sha256": hashlib.sha256(json.dumps(
                    episode["goal_preds"], sort_keys=True).encode()).hexdigest(),
            })
    dataset = {**source, "all_eps": generated}
    manifest = {
        "seed": seed, "roles": roles, "tasks": coordinates,
        "historical_layout_exposure": True, "new_scenes": False,
        "official_data_read": False, "claim_eligible": False,
        "boundary": "prospective goal recombinations on previously exposed development layouts",
    }
    return dataset, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    if (output / "development.pickle").exists():
        raise FileExistsError("generated development data must not be overwritten")
    raw = SOURCE.read_bytes()
    dataset, manifest = build_dataset(pickle.loads(raw))
    output.mkdir(parents=True, exist_ok=True)
    encoded = pickle.dumps(dataset, protocol=4)
    manifest.update({
        "source": str(SOURCE.relative_to(REPO)),
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "dataset_sha256": hashlib.sha256(encoded).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    })
    (output / "development.pickle").write_bytes(encoded)
    (output / "development_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "tasks"}, indent=2))


if __name__ == "__main__":
    main()
