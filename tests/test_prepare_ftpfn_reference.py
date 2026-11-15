import hashlib
import io
import subprocess
import tarfile

import numpy as np
import pytest

from reproduction.prepare.prepare_ftpfn_reference import (
    copy_source,
    digest,
    extract_checkpoint,
    validate_tensors,
)


def archive_fixture(tmp_path, *, kind="regular", duplicate=False):
    path = tmp_path / "checkpoint.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for _ in range(2 if duplicate else 1):
            member = tarfile.TarInfo("model.pt")
            member.size = 7
            if kind == "link":
                member.type = tarfile.SYMTYPE
                member.linkname = "/outside"
                member.size = 0
            archive.addfile(member, io.BytesIO(b"payload") if kind == "regular" else None)
    return path, {
        "archive_sha256": digest(path),
        "checkpoint_sha256": hashlib.sha256(b"payload").hexdigest(),
        "member_name": "model.pt",
    }


def test_regular_member_is_copied_flat_and_never_overwritten(tmp_path):
    archive, spec = archive_fixture(tmp_path)
    output = tmp_path / "output.pt"
    extract_checkpoint(archive, output, spec)
    assert output.read_bytes() == b"payload"
    with pytest.raises(FileExistsError):
        extract_checkpoint(archive, output, spec)


@pytest.mark.parametrize("problem", ["checksum", "link", "duplicate", "member", "payload"])
def test_archive_corruption_rejected(tmp_path, problem):
    archive, spec = archive_fixture(
        tmp_path, kind="link" if problem == "link" else "regular", duplicate=problem == "duplicate"
    )
    if problem == "checksum":
        spec["archive_sha256"] = "0" * 64
    elif problem == "member":
        spec["member_name"] = "different.pt"
    elif problem == "payload":
        spec["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        extract_checkpoint(archive, tmp_path / "output.pt", spec)


def test_source_copy_excludes_untracked_files_and_rejects_modified_source(tmp_path):
    source = tmp_path / "upstream"
    source.mkdir()
    (source / "ifbo").mkdir()
    (source / "ifbo/core.py").write_text("value = 1\n")
    (source / "LICENSE").write_text("upstream notice\n")
    command = ["git", "-C", str(source)]
    subprocess.run([*command, "init", "-q"], check=True)
    subprocess.run([*command, "add", "ifbo", "LICENSE"], check=True)
    subprocess.run(
        [
            *command,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    revision = subprocess.check_output([*command, "rev-parse", "HEAD"], text=True).strip()
    (source / "ifbo/untracked.py").write_text("raise RuntimeError('must not be copied')\n")
    hashes = copy_source(source, tmp_path / "isolated", {"source_commit": revision})
    assert set(hashes) == {"ifbo/core.py", "LICENSE"}
    assert not (tmp_path / "isolated/ifbo/untracked.py").exists()
    (source / "ifbo/core.py").write_text("value = 2\n")
    with pytest.raises(subprocess.CalledProcessError):
        copy_source(source, tmp_path / "bad", {"source_commit": revision})


@pytest.mark.parametrize("problem", ["value", "dtype", "shape", "extra", "missing"])
def test_exact_tensor_manifest_detects_changes(problem):
    value = np.array([1, 2], dtype=np.float32)
    spec = {
        "state_tensors": {
            "weight": {
                "shape": [2],
                "dtype": "float32",
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
        }
    }
    state = {"weight": value.copy()}
    validate_tensors(state, spec)
    if problem == "value":
        state["weight"][0] = 3
    elif problem == "dtype":
        state["weight"] = value.astype(np.float64)
    elif problem == "shape":
        state["weight"] = value.reshape(1, 2)
    elif problem == "extra":
        state["extra"] = value
    else:
        state.clear()
    with pytest.raises(ValueError):
        validate_tensors(state, spec)
