#!/usr/bin/env bash
# =============================================================================
# clone_refs.sh — restore the read-only reference repos used as Agent context.
#
# These repos live under context4agent/repository/ and are git-ignored by the
# main VISTA-Skill repo. They are NEVER developed in; they exist only so the
# Agent can read them (git log / blame / browse) for related-work context.
#
# THIS FILE IS THE SOURCE OF TRUTH for the pinned commits. If you bump a pin,
# update the table below AND the matching clone_ref call.
#
#   repo        | remote                                          | pinned commit                              | date       | why referenced
#   ------------|-------------------------------------------------|--------------------------------------------|------------|-------------------
#   EmbodiSkill | https://github.com/air-embodied-brain/EmbodiSkill.git | 760126030eab1d33ec6a6f30988f0f1fb58df3a7 | 2026-07-11 | principal controlled baseline (EmbodiSkill*)
#   SkillOpt    | https://github.com/microsoft/SkillOpt.git        | 0f76ab4c1d5f3b01c47fa7b4926015389aab3748 | 2026-08-12 | skill-optimization related work
#   Skill-Pro   | https://github.com/Miracle1207/Skill-Pro.git     | 3be7a9be2d4c024d132efe394d537404eba7e4c8 | 2026-05-11 | skill-evolution related work
#
# Usage:
#   bash context4agent/clone_refs.sh            # clone any that are missing
#   bash context4agent/clone_refs.sh --force     # re-clone (deletes existing)
# =============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/repository"
FORCE=0
[ "${1:-}" = "--force" ] && FORCE=1

mkdir -p "$DIR"

clone_ref() {
  local name="$1" url="$2" sha="$3"
  local dest="$DIR/$name"

  if [ "$FORCE" -eq 1 ] && [ -d "$dest" ]; then
    echo ">> --force: removing existing $name"
    rm -rf "$dest"
  fi

  if [ -d "$dest/.git" ]; then
    local cur; cur=$(git -C "$dest" rev-parse HEAD 2>/dev/null || echo "?")
    if [ "$cur" = "$sha" ]; then
      echo ">> $name already at pinned commit, skipping"
    else
      echo ">> $name present at ${cur:0:7} but pin is ${sha:0:7}; checking out pin"
      git -C "$dest" fetch --all --tags
      git -C "$dest" checkout "$sha"
    fi
    return
  fi

  echo ">> cloning $name @ ${sha:0:7}"
  git clone "$url" "$dest"
  git -C "$dest" checkout "$sha"
}

clone_ref EmbodiSkill https://github.com/air-embodied-brain/EmbodiSkill.git 760126030eab1d33ec6a6f30988f0f1fb58df3a7
clone_ref SkillOpt     https://github.com/microsoft/SkillOpt.git             0f76ab4c1d5f3b01c47fa7b4926015389aab3748
clone_ref Skill-Pro    https://github.com/Miracle1207/Skill-Pro.git          3be7a9be2d4c024d132efe394d537404eba7e4c8

echo ""
echo "Done. Reference repos are git-ignored by the main repo; do not commit them."
echo "To bump a pin, edit this script's table + clone_ref call, then run with --force."
