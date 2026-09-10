"""Rewrite archived experiment paths and verified hash dependencies, then remove aliases.

Dry-run by default. Reads the previous directory migration's explicit moves.json.
Original changed files are archived before applying; no simulator/model is invoked.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import tarfile
import tempfile


TEXT_SUFFIXES = {".json", ".jsonl", ".md", ".py", ".sh", ".txt", ".log", ".yaml", ".yml", ".csv"}
HASH = re.compile(r"(?<![a-f0-9])[a-f0-9]{64}(?![a-f0-9])")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


class PathRewriter:
    def __init__(self, repo: Path, moves: list[dict]):
        self.mapping = {entry["source"]: entry["destination"] for entry in moves}
        # Also handle historical globs and f-strings (e.g. seed{seed}) for
        # namespaces whose phase is unambiguous in the explicit move manifest.
        partials = set()
        for prefix, phase in (("phase2_", 2), ("phase5_", 5), ("fault_repair_", 1)):
            old, new = f"running/{prefix}", f"running/Phase{phase}/{prefix}"
            matching = [(s, d) for s, d in self.mapping.items() if s.startswith(old)]
            if matching and all(d == new + s[len(old):] for s, d in matching):
                self.mapping[old] = new
                partials.add(old)
        boundary = r"(?=$|[/\s\"'`),;:}\\\]]|\.(?:\s|$))"
        sources = "|".join(re.escape(s) + ("" if s in partials else boundary)
                           for s in sorted(self.mapping, key=len, reverse=True))
        # Match complete repo-relative/absolute prefixes, never a substring of a
        # canonical destination such as Phase5/simulator/EmbodiedBench/running/...
        self.pattern = re.compile(
            r"(?<![\w./-])(?P<absolute>" + re.escape(str(repo)) + r"/)?"
            r"(?P<source>" + sources + r")"
        )

    def __call__(self, text: str) -> str:
        return self.pattern.sub(
            lambda m: (m["absolute"] or "") + self.mapping[m["source"]], text
        )


def signed_fields(value: object, trail: tuple = ()) -> list[tuple]:
    """Recognize signatures only if they validate against the original payload."""
    found = []
    if isinstance(value, dict):
        if {"schema_version", "skill", "protocol", "artifact_sha256"} <= value.keys():
            envelope = {k: value[k] for k in ("schema_version", "skill", "protocol")}
            if canonical(envelope) != value["artifact_sha256"]:
                raise ValueError(f"Pre-existing Skill artifact digest mismatch: {trail}")
            found.append((trail, "artifact_sha256", tuple(envelope), value["artifact_sha256"]))
        for key, item in value.items():
            if isinstance(item, str) and HASH.fullmatch(item):
                unsigned = {k: v for k, v in value.items() if k != key}
                if canonical(unsigned) == item:
                    found.append((trail, key, tuple(unsigned), item))
            found.extend(signed_fields(item, (*trail, key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(signed_fields(item, (*trail, index)))
    return found


def at(value: object, trail: tuple) -> dict:
    for part in trail:
        value = value[part]
    return value


def build_changes(originals: dict[str, bytes], rewrite: PathRewriter) -> tuple[dict, dict, int]:
    bases = {name: rewrite(data.decode("utf-8")) for name, data in originals.items()}
    # The current CLI uses the active Phase5 protocol. Its generic defaults
    # should not inherit Phase1 merely because early outputs used vista_skill/.
    cli = "vista_skill/integrations/embodiedbench/cli.py"
    if cli in bases:
        bases[cli] = bases[cli].replace('default="running/Phase1/vista_skill/',
                                       'default="running/Phase5/vista_skill/')
    signatures = {}
    old_hashes = {name: sha(data) for name, data in originals.items()}
    for name, data in originals.items():
        if name.endswith(".json"):
            try:
                value = json.loads(data)
            except ValueError:
                continue  # Preserve incomplete historical outputs as text.
            signatures[name] = signed_fields(value)
    replacements = {}
    for iteration in range(64):
        rendered = {
            name: HASH.sub(lambda m: replacements.get(m[0], m[0]), text).encode()
            for name, text in bases.items()
        }
        updated = {}

        def bind(old: str, new: str) -> None:
            if old in updated and updated[old] != new:
                raise ValueError(f"Ambiguous digest dependency: {old}")
            updated[old] = new

        for name, recipes in signatures.items():
            if not recipes:
                continue
            value = json.loads(rendered[name])
            for trail, key, keys, old in recipes:
                node = at(value, trail)
                bind(old, canonical({k: node[k] for k in keys}))
        for name, data in rendered.items():
            bind(old_hashes[name], sha(data))
        updated = {old: new for old, new in updated.items() if old != new}
        if updated == replacements:
            return ({n: d for n, d in rendered.items() if d != originals[n]}, updated, iteration + 1)
        replacements = updated
    raise ValueError("Hash dependencies did not converge; refusing all mutations")


def collect(repo: Path, script_path: Path) -> dict[str, bytes]:
    roots = [repo / f"running/Phase{i}" for i in range(1, 6)]
    roots += [repo / name for name in ("docs", "configs", "scripts", "tests", "vista_skill")]
    paths = [repo / "README.md", repo / "running/README.md"]
    for root in roots:
        # os.walk does not follow directory aliases. Explicitly prune all aliases.
        for parent, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = [d for d in dirs if not (Path(parent) / d).is_symlink()
                       and d not in {"__pycache__", "closed_feedback", ".git"}]
            paths.extend(Path(parent) / name for name in files)
    result = {}
    for path in sorted(set(paths)):
        if (not path.is_file() or path.is_symlink() or path == script_path
                or path == repo / "tests/test_experiment_path_migration.py"
                or path.suffix not in TEXT_SUFFIXES):
            continue
        data = path.read_bytes()
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        result[str(path.relative_to(repo))] = data
    return result


def atomic_write(path: Path, data: bytes, mode: int) -> None:
    fd, temporary = tempfile.mkstemp(prefix=".path_migration_", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, stat.S_IMODE(mode))
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_aliases(repo: Path, moves: list[dict], *, allow_absent: bool = False) -> None:
    for entry in moves:
        source, target = repo / entry["source"], repo / entry["destination"]
        if not target.exists():
            raise ValueError(f"Missing destination: {target}")
        if allow_absent and not source.exists() and not source.is_symlink():
            continue
        if not source.is_symlink() or source.resolve() != target.resolve():
            raise ValueError(f"Old path is no longer the expected alias: {source}")


def run(repo: Path, manifest: Path, output: Path, apply: bool) -> dict:
    moves = json.loads(manifest.read_text())
    for entry in moves:
        for key in ("source", "destination"):
            path = Path(entry[key])
            if (path.is_absolute() or ".." in path.parts or "closed_feedback" in path.parts
                    or not str(path).startswith(("running/", "EmbodiedBench/running/"))):
                raise ValueError(f"Unsafe migration path: {path}")
        if not re.match(r"running/Phase[1-5]/", entry["destination"]):
            raise ValueError("Destination must be in a Phase directory")
    validate_aliases(repo, moves, allow_absent=not apply)
    rewrite = PathRewriter(repo, moves)
    originals = collect(repo, Path(__file__).resolve())
    changes, hashes, rounds = build_changes(originals, rewrite)
    report = {
        "mode": "apply" if apply else "dry_run", "changed_files": len(changes),
        "changed_by_root": {}, "hash_dependency_rounds": rounds,
        "updated_digest_values": len(hashes), "alias_count": len(moves),
        "excluded": ["running/closed_feedback", "running/stock_v2_*",
                     "running/migrations/phase_reorganization_20260908", "context4agent/latex",
                     "migration script and its deliberate legacy-path test fixtures",
                     "binary files and original source archives"],
        "provenance": "Relocated derivatives; pre-migration bytes preserved in originals.tar.gz. "
                      "New hashes are storage identities, not a new preregistration or experiment.",
    }
    for name in changes:
        root = "/".join(Path(name).parts[:2]) if name.startswith("running/") else Path(name).parts[0]
        report["changed_by_root"][root] = report["changed_by_root"].get(root, 0) + 1
    for name, data in changes.items():
        if rewrite(data.decode()) != data.decode():
            raise ValueError(f"Unresolved old path: {name}")
        if name.endswith(".json"):
            try:
                old = json.loads(originals[name])
            except ValueError:
                continue
            new = json.loads(data)
            if isinstance(old, dict) and "skill" in old and old["skill"] != new["skill"]:
                raise ValueError(f"Skill content changed, requires separate migration: {name}")
    if not apply:
        return report
    output.mkdir(parents=True, exist_ok=False)
    metadata = []
    # Archive exact originals (also of dirty workspace files), not reconstructed JSON.
    with tarfile.open(output / "originals.tar.gz", "w:gz") as archive:
        for name in sorted(changes):
            path = repo / name
            if path.read_bytes() != originals[name]:
                raise ValueError(f"Concurrent write before backup: {name}")
            info = archive.gettarinfo(str(path), arcname=name)
            archive.addfile(info, io.BytesIO(originals[name]))
            metadata.append({"path": name, "before_sha256": sha(originals[name]),
                             "after_sha256": sha(changes[name]), "mode": info.mode})
    (output / "files.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (output / "hash_mapping.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    (output / "aliases.json").write_text(json.dumps(moves, indent=2) + "\n")
    report["status"] = "backup_complete"
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with tarfile.open(output / "originals.tar.gz") as archive:
        for item in metadata:
            if sha(archive.extractfile(item["path"]).read()) != item["before_sha256"]:
                raise ValueError("Backup verification failed")
    # Check the whole observed scope before writing, including unchanged dependents.
    for name, data in originals.items():
        if (repo / name).read_bytes() != data:
            raise ValueError(f"Concurrent write before apply: {name}")
    written, removed = [], []
    try:
        for item in metadata:
            name = item["path"]
            if (repo / name).read_bytes() != originals[name]:
                raise ValueError(f"Concurrent write during apply: {name}")
            atomic_write(repo / name, changes[name], item["mode"])
            written.append(item)
        for item in metadata:
            if sha((repo / item["path"]).read_bytes()) != item["after_sha256"]:
                raise ValueError(f"Readback mismatch: {item['path']}")
        validate_aliases(repo, moves)
        for entry in moves:
            alias = repo / entry["source"]
            removed.append((alias, os.readlink(alias)))
            alias.unlink()
    except BaseException:
        for alias, target in removed:
            if not alias.exists() and not alias.is_symlink():
                alias.symlink_to(target)
        for item in reversed(written):
            path = repo / item["path"]
            if sha(path.read_bytes()) == item["after_sha256"]:
                atomic_write(path, originals[item["path"]], item["mode"])
        raise
    report.update(status="verified", backup_sha256=sha((output / "originals.tar.gz").read_bytes()),
                  verified_files=len(changes), removed_aliases=len(removed))
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=Path("running/migrations/phase_reorganization_20260908/moves.json"))
    parser.add_argument("--output", type=Path, default=Path("running/migrations/phase_path_rewrite_20260908"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    repo = args.repo.resolve()
    print(json.dumps(run(repo, repo / args.manifest, repo / args.output, args.apply), indent=2))


if __name__ == "__main__":
    main()
