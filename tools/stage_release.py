"""Stage the public source tree from the larger development checkout."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

DIRECTORIES = (
    "icicle", "icicle_jax", "examples", "experiments", "reproduction",
    "benchmarks", "configs", "tests", "tools/lumi", "assets",
)
FILES = (
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES",
    "CITATION.cff", "pyproject.toml", ".gitignore", "tools/stage_release.py",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=repo / "configs/icicle.json",
    )
    args = parser.parse_args()
    if args.out.exists():
        parser.error("choose a new staging directory; existing trees are not overwritten")
    config = json.loads(args.candidate_config.read_text())
    if config["run"]["num_steps"] != 80000 or config["run"]["seed"] != 0:
        parser.error("release recipe must use 80,000 updates and seed 0")
    if config["optim"]["decay_start"] + config["optim"]["decay_steps"] != 80000:
        parser.error("candidate schedule must end at T80k")
    if config["model"]["architecture"] != "icicle":
        parser.error("release configuration must use ICICLE")
    if config["model"].pop("fidelity_code", "split") != "split":
        parser.error("configuration uses an unsupported fidelity code")
    if config["model"].pop("final_norm", True) is not True:
        parser.error("configuration must enable the final norm")
    selected = set(FILES)
    eligible = set(subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        text=True,
    ).split("\0"))
    for directory in DIRECTORIES:
        for path in (repo / directory).rglob("*"):
            relative = path.relative_to(repo).as_posix()
            if (path.is_file() and relative in eligible
                    and "__pycache__" not in path.parts and path.suffix != ".pyc"):
                selected.add(relative)
    for relative in sorted(selected):
        source = repo / relative
        if source.is_symlink():
            raise ValueError(f"review symlink before staging: {relative}")
        destination = args.out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    config_path = args.out / "configs/icicle.json"
    config_path.parent.mkdir(exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    selected.add(config_path.relative_to(args.out).as_posix())
    report = {
        "status": "selected local release candidate; not published",
        "boundary": "library, examples, maintained runners and core tests",
        "candidate_config_origin_sha256": digest(args.candidate_config),
        "candidate_config_note": "portable full T80k schedule trained from initialization",
        "deferred_artifacts": [
            "public release locations",
            "paper identifiers",
        ],
        "files": {relative: digest(args.out / relative) for relative in sorted(selected)},
    }
    (args.out / "SOURCE_MANIFEST.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "files": len(selected), "status": report["status"]}))


if __name__ == "__main__":
    main()
