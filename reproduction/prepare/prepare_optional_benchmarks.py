"""Fetch pinned published ARLBench tables or Quick-Tune metadata into a fresh data folder.

Assets keep their upstream licenses. Downloads and local reuse must match the release's
SHA-256 manifest before extraction. This prepares static benchmark tables.
Normalization, crash handling and task views remain exclusively in the benchmark adapters.
"""

import argparse
import hashlib
import json
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_archive(archive, destination, prefix):
    """Extract one declared top-level directory, rejecting links and ambiguous paths."""
    with zipfile.ZipFile(archive) as source:
        seen = set()
        members = source.infolist()
        for member in members:
            name = PurePosixPath(member.filename)
            if (
                name.is_absolute()
                or ".." in name.parts
                or not name.parts
                or name.parts[0] != prefix
                or "\\" in member.filename
                or stat.S_ISLNK(member.external_attr >> 16)
                or name in seen
            ):
                raise ValueError(f"unsafe or duplicate archive member: {member.filename}")
            seen.add(name)
        for member in members:
            target = destination / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(member) as incoming, target.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)


def prepare(data_dir, suite, manifest, source_dir=None):
    """Download and check the source files in a new output directory."""
    if manifest.get("schema_version") != 1 or suite not in manifest["suites"]:
        raise ValueError("unknown suite or source manifest version")
    spec = manifest["suites"][suite]
    paths = [PurePosixPath(asset["path"]) for asset in spec["assets"]]
    if (
        not paths
        or len(set(paths)) != len(paths)
        or any(path.is_absolute() or ".." in path.parts or path.parts[0] != suite for path in paths)
    ):
        raise ValueError("source manifest contains unexpected asset paths")
    output = data_dir / suite
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for asset in spec["assets"]:
        destination = data_dir / asset["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".partial")
        if source_dir is None:
            if not asset["url"].startswith("https://"):
                raise ValueError("asset downloads require HTTPS")
            with (
                urllib.request.urlopen(asset["url"], timeout=120) as incoming,
                partial.open("xb") as outgoing,
            ):
                shutil.copyfileobj(incoming, outgoing)
        else:
            shutil.copyfile(source_dir / asset["path"], partial)
        if partial.stat().st_size != asset["bytes"] or file_digest(partial) != asset["sha256"]:
            raise ValueError(f"published asset checksum mismatch: {asset['path']}")
        partial.rename(destination)
        if "extract_prefix" in asset:
            extract_archive(destination, output, asset["extract_prefix"])
        records.append({"path": asset["path"], "sha256": asset["sha256"], "bytes": asset["bytes"]})
        print(f"PREPARED {asset['path']}", flush=True)
    report = {
        "complete": True,
        "suite": suite,
        "source": spec["source"],
        "revision": spec["revision"],
        "license": spec["license"],
        "source_manifest_sha256": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
        "assets": records,
    }
    (output / "PREPARATION.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--suite", choices=("arlbench", "qtmeta"), required=True)
    parser.add_argument(
        "--source-dir", type=Path, help="reuse pinned assets from another data root"
    )
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[2] / "benchmarks/optional_sources.json"
    prepare(args.data_dir, args.suite, json.loads(manifest.read_text()), args.source_dir)


if __name__ == "__main__":
    main()
