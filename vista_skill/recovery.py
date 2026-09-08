"""Bounded temporal discovery from natural, evidence-grounded failure chains.

The grammar is supplied by the method; the action, predicate, and recovery
policy are admitted only after independent acquisition episodes support them.
This is a narrow discovery hypothesis, not unrestricted procedure induction.
"""
from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from vista_skill.schemas import (
    PatchOperation, SkillField, SkillPatch, SkillSpec, TemporalSkillRule, TruthValue,
)


def _near_evidence(event: Mapping[str, Any], target: str, value: str) -> tuple[str, ...]:
    return tuple(
        str(item["evidence_id"])
        for item in event.get("evidence_delta", ())
        if item["key"] == {"name": "near", "arguments": [target]}
        and item["after"] == value
        and float(item["confidence"]) >= 0.75
        and float(item.get("coverage", 1.0)) >= 0.5
        and int(item["timestamp"]) == int(event["step_id"])
    )


def recovery_chains(events: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Find failed pick -> successful nav -> same-target failed pick chains.

    A target-resolving observation or successful same-target pick ends a chain.
    Multiple failures within one episode never become independent support.
    No task outcomes, oracle labels, expected transitions, or task IDs select
    the pattern; episode IDs are used solely for independent recurrence.
    """
    active: dict[tuple[str, str], dict[str, Any]] = {}
    chains = []
    for event in events:
        episode = str(event["episode_id"])
        action = event["action"]
        arguments = action.get("arguments", ())
        target = str(arguments[0]) if arguments else ""
        success = event.get("last_action_success")
        for key, pending in tuple(active.items()):
            if key[0] != episode:
                continue
            if _near_evidence(event, key[1], "true") or (
                action["action_type"] == "pick" and target == key[1] and success is True
            ):
                del active[key]
            elif action["action_type"] == "nav" and success is True:
                pending["navigation_event_ids"].append(str(event["event_id"]))
        failed = _near_evidence(event, target, "false")
        if action["action_type"] != "pick" or success is not False or not failed:
            continue
        key = (episode, target)
        pending = active.get(key)
        if pending and pending["navigation_event_ids"] and pending["trigger_vtca_identified"]:
            chains.append({
                **pending,
                "repeat_event_id": str(event["event_id"]),
                "evidence_ids": [*pending["evidence_ids"], *failed],
                "repeat_step": int(event["step_id"]),
            })
        active[key] = {
            "episode_id": episode,
            "target": target,
            "trigger_event_id": str(event["event_id"]),
            "trigger_step": int(event["step_id"]),
            "evidence_ids": list(failed),
            "navigation_event_ids": [],
            "trigger_vtca_identified": bool(
                (event.get("attribution") or {}).get("target") == "skill_update"
                and (event.get("attribution") or {}).get("identifiability", {}).get("identified")
            ),
        }
    return tuple(chains)


def propose_recovery(
    parent: SkillSpec,
    events: Sequence[Mapping[str, Any]],
    *,
    minimum_episodes: int = 2,
) -> tuple[SkillPatch | None, dict[str, Any]]:
    if minimum_episodes < 2:
        raise ValueError("recovery discovery requires at least two independent episodes")
    chains = recovery_chains(events)
    episodes = sorted({item["episode_id"] for item in chains})
    evidence_ids = tuple(sorted({eid for item in chains for eid in item["evidence_ids"]}))
    audit = {
        "grammar": "failed_pick_nav_failed_pick_target_evidence_v1",
        "chains": chains,
        "independent_episodes": episodes,
        "evidence_ids": evidence_ids,
        "minimum_episodes": minimum_episodes,
        "candidate_generated": len(episodes) >= minimum_episodes,
        "model_calls": 0,
        "uses_task_outcomes": False,
    }
    if len(episodes) < minimum_episodes:
        return None, audit
    statement = (
        "After a pick fails because the selected object is not near, pause picks "
        "of that object. Navigate to obtain a useful new view and require fresh "
        "evidence that the same object is near before retrying. Successful "
        "navigation alone is insufficient; if the object remains unconfirmed, "
        "search another location."
    )
    rule = TemporalSkillRule(
        rule_id="discovered_pick_target_evidence_v1",
        field=SkillField.PROCEDURE,
        trigger_action_type="pick",
        trigger_success=False,
        trigger_predicate="near({arg0})",
        trigger_value=TruthValue.FALSE,
        blocked_action_type="pick",
        recovery_action_types=("nav",),
        recovery_release="target_evidence",
    )
    identity = "|".join((parent.skill_id, str(parent.version), statement, *evidence_ids))
    patch = SkillPatch(
        patch_id="recovery_" + hashlib.sha256(identity.encode()).hexdigest()[:16],
        skill_id=parent.skill_id,
        parent_version=parent.version,
        field=SkillField.PROCEDURE,
        operation=PatchOperation.APPEND,
        old="",
        new=statement,
        evidence_ids=evidence_ids,
        scope="generalized:pick|near({arg0})|false;discovery",
        rationale="Repeated acquisition chains refute navigation success as sufficient recovery.",
        temporal_rules=(*(
            item for item in parent.temporal_rules if item.field is SkillField.PROCEDURE
        ), rule),
    )
    return patch, audit
