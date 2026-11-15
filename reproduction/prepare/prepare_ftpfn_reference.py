"""Reproduce the published FT-PFN state_dict in a separate CPU PyTorch environment.

The pinned checkpoint is hash-verified before deserialization. Two import-only changes
are applied to an isolated copy of the pinned source. Every tensor and the reference
forward are checked before writing state_dict.npz for the JAX safetensors converter.
Run reference preparation in a separate PyTorch environment.
"""

import argparse
import difflib
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "benchmarks/ftpfn_reference.json"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract_checkpoint(archive, output, spec):
    """Copy only the pinned regular member; never extract archive paths onto disk."""
    if digest(archive) != spec["archive_sha256"]:
        raise ValueError("published archive checksum differs")
    with tarfile.open(archive, "r:gz") as source:
        members = source.getmembers()
        if len(members) != 1 or members[0].name != spec["member_name"] or not members[0].isfile():
            raise ValueError("expected exactly the declared regular checkpoint member")
        with source.extractfile(members[0]) as incoming, output.open("xb") as target:
            shutil.copyfileobj(incoming, target)
    if digest(output) != spec["checkpoint_sha256"]:
        raise ValueError("published checkpoint checksum differs")


def patch_imports(package, spec):
    """Reorder initialization and import Optional from typing, with exact patch hashes."""
    records = {}
    for name, pins in spec["source_patches"].items():
        path = package / name
        original = path.read_text()
        if digest(path) != pins["original_sha256"]:
            raise ValueError(f"upstream source differs: {name}")
        if name == "ifbo/__init__.py":
            revised = original.replace("from .model import Surrogate\n", "").replace(
                "from .bar_distribution import BarDistribution\n", ""
            )
            revised = (
                revised.replace(
                    "@dataclass(unsafe_hash=True)",
                    "from .bar_distribution import BarDistribution\n\n@dataclass(unsafe_hash=True)",
                    1,
                )
                + "\n\nfrom .model import Surrogate\n"
            )
        elif name == "ifbo/layer.py":
            needle = (
                "from torch.nn.modules.transformer import _get_activation_fn, Module, Tensor, "
                "Optional, MultiheadAttention, Linear, Dropout, LayerNorm"
            )
            revised = original.replace(
                needle,
                "from typing import Optional\n" + needle.replace("Tensor, Optional,", "Tensor,"),
                1,
            )
        else:
            raise ValueError(f"unsupported source patch: {name}")
        if hashlib.sha256(revised.encode()).hexdigest() != pins["patched_sha256"]:
            raise ValueError(f"import-only patch differs: {name}")
        path.write_text(revised)
        records[name] = {
            **pins,
            "diff": "".join(
                difflib.unified_diff(
                    original.splitlines(True),
                    revised.splitlines(True),
                    fromfile=f"upstream/{name}",
                    tofile=f"isolated/{name}",
                )
            ),
        }
    return records


def copy_source(source, output, spec):
    """Copy only Git-tracked package files from the exact unmodified upstream commit."""
    command = ["git", "-C", str(source)]
    revision = subprocess.check_output([*command, "rev-parse", "HEAD"], text=True).strip()
    if revision != spec["source_commit"]:
        raise ValueError("ifBO checkout is not the pinned reference revision")
    subprocess.run(
        [
            *command,
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--quiet",
            "HEAD",
            "--",
            "ifbo",
            "LICENSE",
        ],
        check=True,
    )
    files = (
        subprocess.check_output([*command, "ls-files", "-z", "--", "ifbo", "LICENSE"])
        .decode()
        .split("\0")
    )
    hashes = {}
    for name in filter(None, files):
        path = source / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"expected regular tracked source file: {name}")
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        hashes[name] = digest(target)
    return hashes


def validate_tensors(state, spec):
    """Require all published parameters and criterion buffers, with no extra tensors."""
    expected = spec["state_tensors"]
    if set(state) != set(expected):
        raise ValueError("state_dict tensor keys differ from the published reference")
    for name, value in state.items():
        actual = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
        }
        if actual != expected[name]:
            raise ValueError(f"published tensor differs: {name}")


def prepare(output, spec, *, archive=None, source=None):
    """Create a fresh preparation directory; preserve inputs, patches and result evidence."""
    output.mkdir(parents=True, exist_ok=False)
    if archive is None:
        archive = output / "ftpfnv0.0.1.tar.gz"
        with (
            urllib.request.urlopen(spec["archive_url"], timeout=120) as incoming,
            archive.open("xb") as target,
        ):
            shutil.copyfileobj(incoming, target)
    checkpoint = output / "published-model.pt"
    extract_checkpoint(archive, checkpoint, spec)
    if source is None:
        source = output / "upstream"
        subprocess.run(["git", "init", str(source)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "fetch",
                "--depth=1",
                spec["source_url"],
                spec["source_commit"],
            ],
            check=True,
        )
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", "FETCH_HEAD"], check=True)
    isolated = output / "source"
    sources = copy_source(source, isolated, spec)
    patches = patch_imports(isolated, spec)
    (output / "source_provenance.json").write_text(
        json.dumps(
            {
                "source_commit": spec["source_commit"],
                "unmodified_files": sources,
                "patches": patches,
            },
            indent=2,
        )
        + "\n"
    )
    if "ifbo" in sys.modules:
        raise ValueError("run preparation in a fresh process without a previously imported ifbo")
    sys.path.insert(0, str(isolated))
    import numpy as np
    import torch

    model = torch.load(checkpoint, map_location="cpu", weights_only=False).eval()
    state = {
        name: value.detach().cpu().numpy().copy() for name, value in model.state_dict().items()
    }
    validate_tensors(state, spec)
    fixture = REPO / spec["golden_fixture"]
    if digest(fixture) != spec["golden_fixture_sha256"]:
        raise ValueError("reference forward fixture checksum differs")
    ref = np.load(fixture, allow_pickle=False)
    with torch.no_grad():
        logits = (
            model.double()(
                (torch.from_numpy(ref["x"]).double(), torch.from_numpy(ref["y"]).double()),
                single_eval_pos=int(ref["single_eval_pos"]),
            )
            .cpu()
            .numpy()
        )
    np.testing.assert_allclose(logits, ref["logits"], atol=1e-8, rtol=1e-8)
    np.savez(output / "state_dict.npz", **state)
    result = {
        "complete": True,
        "kind": "published FT-PFN state_dict reproduction",
        "source_commit": spec["source_commit"],
        "archive_sha256": digest(archive),
        "checkpoint_sha256": digest(checkpoint),
        "all_published_tensors_verified": True,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "forward_max_abs_error": float(np.max(np.abs(logits - ref["logits"]))),
        "expected_converted_safetensors_sha256": spec["safetensors_sha256"],
    }
    (output / "preparation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archive", type=Path, help="reuse a hash-verified published archive")
    parser.add_argument(
        "--ifbo-source", type=Path, help="reuse a pristine checkout at the pinned commit"
    )
    args = parser.parse_args()
    prepare(
        args.out.resolve(),
        json.loads(MANIFEST.read_text()),
        archive=args.archive.resolve() if args.archive else None,
        source=args.ifbo_source.resolve() if args.ifbo_source else None,
    )


if __name__ == "__main__":
    main()
