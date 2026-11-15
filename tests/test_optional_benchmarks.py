import hashlib
import json
import stat
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from reproduction.prepare import verify_benchmarks
from reproduction.prepare.prepare_optional_benchmarks import extract_archive, prepare


def test_pinned_local_asset_preparation_preserves_bytes_and_refuses_overwrite(tmp_path):
    source = tmp_path / "source"
    asset = source / "arlbench/landscapes/task.csv"
    asset.parent.mkdir(parents=True)
    content = b"config_id,performance\n0,NaN\n"
    asset.write_bytes(content)
    manifest = {
        "schema_version": 1,
        "suites": {
            "arlbench": {
                "source": "https://example.org/data",
                "revision": "fixed",
                "license": "upstream",
                "assets": [
                    {
                        "path": "arlbench/landscapes/task.csv",
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ],
            }
        },
    }
    output = tmp_path / "prepared"
    report = prepare(output, "arlbench", manifest, source)
    assert (output / "arlbench/landscapes/task.csv").read_bytes() == content
    assert json.loads((output / "arlbench/PREPARATION.json").read_text()) == report
    with pytest.raises(FileExistsError):
        prepare(output, "arlbench", manifest, source)
    asset.write_bytes(b"config_id,performance\n0,0.0\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare(tmp_path / "changed", "arlbench", manifest, source)


@pytest.mark.parametrize("name", ["../outside", "/outside", "micro/../outside", "other/file"])
def test_archive_rejects_escape_and_unexpected_prefix(tmp_path, name):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("micro/valid", b"valid")
        target.writestr(name, b"invalid")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="unsafe"):
        extract_archive(archive, output, "micro")
    assert not output.exists()


def test_archive_rejects_symlink_before_writing_any_members(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as target:
        link = zipfile.ZipInfo("micro/link")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        target.writestr(link, "../outside")
    with pytest.raises(ValueError, match="unsafe"):
        extract_archive(archive, tmp_path / "output", "micro")


def test_archive_extracts_published_directory_layout(tmp_path):
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("micro/", b"")
        target.writestr("micro/args.json", b'{"curve":{}}')
    extract_archive(archive, tmp_path / "output", "micro")
    assert (tmp_path / "output/micro/args.json").read_bytes() == b'{"curve":{}}'


@pytest.mark.parametrize("field", ["hps", "curves", "fidelities", "cat_card", "costs"])
def test_verifier_detects_changed_inputs(monkeypatch, tmp_path, field):
    arrays = {
        "hps": np.array([[0.25], [0.75]], dtype=np.float32),
        "curves": np.array([[0.2, 0.5], [0.3, 0.7]], dtype=np.float32),
        "fidelities": np.array([0.5, 1.0], dtype=np.float32),
        "cat_card": np.array([2], dtype=np.int32),
        "costs": np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64),
    }
    task = SimpleNamespace(name="task", **arrays)
    fields = list(arrays)
    checks = {"fields": fields, "tasks": {"task": verify_benchmarks.input_record(task, fields)}}
    (tmp_path / "checks.json").write_text(json.dumps(checks))
    manifest = tmp_path / "inputs.json"
    manifest.write_text(json.dumps({"suite": {"tasks": 1, "inputs": "checks.json"}}))
    monkeypatch.setattr(verify_benchmarks, "benchmark_tasks", lambda *_: [("task", lambda: task)])
    assert verify_benchmarks.verify(None, manifest, ["suite"]) == {"suite": 1}
    setattr(task, field, getattr(task, field) + 1)
    with pytest.raises(ValueError, match="prepared inputs differ"):
        verify_benchmarks.verify(None, manifest, ["suite"])
