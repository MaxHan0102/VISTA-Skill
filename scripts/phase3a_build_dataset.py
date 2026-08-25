"""Build the human-free 300-transition Phase3A evidence dataset.

Natural samples come from frozen-executor rollouts; stress samples come from a
fixed, prediction-blind primitive schedule.  Both have independent Habitat
state-oracle labels, and the three evaluation splits are scene-disjoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Iterable


NATURAL_TARGETS = {"dev": 80, "selection": 53, "audit": 67}
STRESS_TARGETS = {"dev": 40, "selection": 27, "audit": 33}
ACTION_TYPES = ("nav", "pick", "place", "open", "close")
STRESS_ACTION_MINIMUMS = {
    "dev": {"open": 12, "close": 12, "place": 5},
    "selection": {"open": 9, "close": 8, "place": 4},
    "audit": {"open": 10, "close": 10, "place": 5},
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--natural-events", type=Path, required=True)
    parser.add_argument("--stress-events", type=Path, required=True)
    parser.add_argument(
        "--dataset-pickle",
        type=Path,
        default=Path(
            "EmbodiedBench/embodiedbench/envs/eb_habitat/datasets/"
            "train_validation.pickle"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _paired_rows(path: Path, regime: str) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    labels = {
        (str(item["payload"]["episode_id"]), int(item["payload"]["step_id"])):
        item["payload"]
        for item in records
        if item["event_type"] == "state_oracle_label"
    }
    rows = []
    for item in records:
        if item["event_type"] != "transition":
            continue
        transition = item["payload"]
        coordinate = (str(transition["episode_id"]), int(transition["step_id"]))
        oracle = labels.get(coordinate)
        if oracle is None:
            continue
        if not Path(transition["pre_image"]).is_file() or not Path(
            transition["post_image"]
        ).is_file():
            continue
        definite = sum(x["value"] != "unknown" for x in oracle["post"])
        if definite == 0:
            continue
        rows.append(
            {
                "sample_id": f"{regime}:{coordinate[0]}:s{coordinate[1]}",
                "regime": regime,
                "stress_type": (
                    None if regime == "natural" else "scripted_primitive_schedule"
                ),
                "source_sample_id": None,
                "transition": transition,
                "oracle": oracle,
            }
        )
    return rows


def _scene_map(path: Path) -> dict[str, str]:
    payload = pickle.loads(path.read_bytes())
    return {
        str(item["episode_id"]): str(item["scene_id"])
        for item in payload["all_eps"]
    }


def _round_robin(rows: Iterable[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    buckets: dict[str, deque[dict[str, Any]]] = {
        action: deque() for action in ACTION_TYPES
    }
    other: deque[dict[str, Any]] = deque()
    for row in sorted(rows, key=lambda x: x["sample_id"]):
        action = str(row["transition"]["action"]["action_type"])
        (buckets[action] if action in buckets else other).append(row)
    selected: list[dict[str, Any]] = []
    while len(selected) < count and (any(buckets.values()) or other):
        progressed = False
        for action in ACTION_TYPES:
            if buckets[action] and len(selected) < count:
                selected.append(buckets[action].popleft())
                progressed = True
        if other and len(selected) < count:
            selected.append(other.popleft())
            progressed = True
        if not progressed:
            break
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} eligible samples for requested {count}")
    return selected


def _select_with_minimums(
    rows: Iterable[dict[str, Any]],
    count: int,
    minimums: dict[str, int],
) -> list[dict[str, Any]]:
    rows = sorted(rows, key=lambda x: x["sample_id"])
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for action, minimum in minimums.items():
        candidates = [
            row for row in rows if row["transition"]["action"]["action_type"] == action
        ]
        if len(candidates) < minimum:
            raise ValueError(
                f"stress split has {len(candidates)} {action} rows, needs {minimum}"
            )
        for row in candidates[:minimum]:
            selected.append(row)
            selected_ids.add(row["sample_id"])
    remainder = _round_robin(
        (row for row in rows if row["sample_id"] not in selected_ids),
        count - len(selected),
    )
    return [*selected, *remainder]


def _assign_scenes(
    rows: list[dict[str, Any]], scene_by_episode: dict[str, str], seed: int
) -> dict[str, tuple[str, ...]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        episode = str(row["transition"]["episode_id"])
        scene = scene_by_episode.get(episode)
        if scene is None:
            raise ValueError(f"missing scene for episode {episode}")
        by_scene[scene].append(row)
    scenes = sorted(
        by_scene,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).hexdigest(),
    )
    if len(scenes) < 6:
        raise ValueError(f"scene-disjoint 3-way split requires >=6 scenes, got {len(scenes)}")
    # Search all disjoint non-empty scene assignments.  We prefer two scenes per
    # split and the smallest capacity surplus to avoid over-representing one scene.
    best: tuple[int, dict[str, tuple[str, ...]]] | None = None
    import itertools

    for dev_n in range(1, len(scenes) - 1):
        for dev in itertools.combinations(scenes, dev_n):
            remaining = [scene for scene in scenes if scene not in dev]
            for selection_n in range(1, len(remaining)):
                for selection in itertools.combinations(remaining, selection_n):
                    audit = tuple(scene for scene in remaining if scene not in selection)
                    assignment = {
                        "dev": tuple(dev),
                        "selection": tuple(selection),
                        "audit": audit,
                    }
                    capacities = {
                        split: {
                            regime: sum(
                                row["regime"] == regime
                                for scene in split_scenes
                                for row in by_scene[scene]
                            )
                            for regime in ("natural", "stress")
                        }
                        for split, split_scenes in assignment.items()
                    }
                    if any(
                        capacities[s]["natural"] < NATURAL_TARGETS[s]
                        or capacities[s]["stress"] < STRESS_TARGETS[s]
                        for s in assignment
                    ):
                        continue
                    surplus = sum(
                        capacities[s]["natural"]
                        - NATURAL_TARGETS[s]
                        + capacities[s]["stress"]
                        - STRESS_TARGETS[s]
                        for s in assignment
                    )
                    imbalance = sum(abs(len(assignment[s]) - 2) for s in assignment)
                    score = surplus * 10 + imbalance
                    if best is None or score < best[0]:
                        best = (score, assignment)
    if best is None:
        capacities = {scene: len(items) for scene, items in by_scene.items()}
        raise ValueError(f"no scene-disjoint assignment has required capacity: {capacities}")
    return best[1]


def main() -> int:
    args = _args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    natural_rows = _paired_rows(args.natural_events, "natural")
    stress_rows = _paired_rows(args.stress_events, "stress")
    rows = [*natural_rows, *stress_rows]
    scene_by_episode = _scene_map(args.dataset_pickle)
    assignment = _assign_scenes(rows, scene_by_episode, args.seed)
    natural_by_split: dict[str, list[dict[str, Any]]] = {}
    stress_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in ("dev", "selection", "audit"):
        natural_candidates = [
            row
            for row in rows
            if row["regime"] == "natural"
            if scene_by_episode[str(row["transition"]["episode_id"])]
            in assignment[split]
        ]
        stress_candidates = [
            row
            for row in rows
            if row["regime"] == "stress"
            if scene_by_episode[str(row["transition"]["episode_id"])]
            in assignment[split]
        ]
        natural = _round_robin(natural_candidates, NATURAL_TARGETS[split])
        stress = _select_with_minimums(
            stress_candidates,
            STRESS_TARGETS[split],
            STRESS_ACTION_MINIMUMS[split],
        )
        for row in (*natural, *stress):
            row["split"] = split
            row["transition"]["pre_image"] = str(
                Path(row["transition"]["pre_image"]).resolve()
            )
            row["transition"]["post_image"] = str(
                Path(row["transition"]["post_image"]).resolve()
            )
            row["scene_id"] = scene_by_episode[str(row["transition"]["episode_id"])]
        natural_by_split[split] = natural
        stress_by_split[split] = stress
    selected = [
        row
        for split in ("dev", "selection", "audit")
        for row in (*natural_by_split[split], *stress_by_split[split])
    ]
    action_counts = Counter(
        row["transition"]["action"]["action_type"] for row in selected
    )
    split_counts = Counter(row["split"] for row in selected)
    regime_counts = Counter(row["regime"] for row in selected)
    stress_counts = Counter(row["stress_type"] for row in selected if row["stress_type"])
    scene_sets = {
        split: sorted({row["scene_id"] for row in selected if row["split"] == split})
        for split in split_counts
    }
    if any(set(scene_sets[left]) & set(scene_sets[right]) for left, right in (
        ("dev", "selection"), ("dev", "audit"), ("selection", "audit")
    )):
        raise AssertionError("scene leakage across splits")
    payload = {
        "dataset_id": "vista_phase3a_evidence_state_oracle_cachefix_v3",
        "natural_events": str(args.natural_events.resolve()),
        "natural_events_sha256": _sha256(args.natural_events),
        "stress_events": str(args.stress_events.resolve()),
        "stress_events_sha256": _sha256(args.stress_events),
        "dataset_pickle": str(args.dataset_pickle.resolve()),
        "dataset_pickle_sha256": _sha256(args.dataset_pickle),
        "seed": args.seed,
        "sample_count": len(selected),
        "counts": {
            "split": dict(split_counts),
            "regime": dict(regime_counts),
            "action": dict(action_counts),
            "stress_type": dict(stress_counts),
        },
        "scene_assignment": scene_sets,
        "scene_disjoint": True,
        "task_disjoint": True,
        "samples": selected,
    }
    if len(selected) != 300 or split_counts != Counter({"dev": 120, "audit": 100, "selection": 80}):
        raise AssertionError(f"unexpected sample/split counts: {len(selected)}, {split_counts}")
    shortfall = {action: 30 - action_counts[action] for action in ACTION_TYPES if action_counts[action] < 30}
    if shortfall:
        raise ValueError(f"primitive action quotas remain unmet: {shortfall}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ("sample_count", "counts", "scene_assignment")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
