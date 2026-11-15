"""Round trip: nnx model -> safetensors export -> numpy reload, byte-exact."""

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
from flax import nnx
from safetensors.numpy import load_file

from icicle.model_format import expected_tensors, validate
from reproduction.jax.eval import evaluate
from icicle_jax.loading import load_eval_model
from icicle_jax.export import export_model, export_run, model_tensors
from icicle_jax.loading import build_model
from tests.reproduction.jax.test_eval import MODEL_CFG, PROTO, make_task


def roundtrip(model_cfg, tmp_path):
    model = build_model(model_cfg, nnx.Rngs(0))
    out = export_model(model, dataclasses.asdict(model_cfg), 123, tmp_path / "small.safetensors")
    tensors = load_file(out)
    manifest = json.loads(out.with_suffix(".json").read_text())
    return model, tensors, manifest


def test_export_roundtrip_byte_exact(tmp_path):
    model, tensors, manifest = roundtrip(MODEL_CFG, tmp_path)
    validate(manifest, tensors)
    assert manifest["step"] == 123
    assert manifest["model"]["num_bars"] == MODEL_CFG.num_bars
    live = model_tensors(model)
    assert tensors.keys() == live.keys()
    for name, a in live.items():
        np.testing.assert_array_equal(tensors[name], a)


def test_expected_set_matches_real_model(tmp_path):
    variant = dataclasses.replace(MODEL_CFG, qk_norm=False)
    for cfg in (MODEL_CFG, variant):
        model = build_model(cfg, nnx.Rngs(0))
        live = model_tensors(model)
        expected = expected_tensors(dataclasses.asdict(cfg))
        assert live.keys() == expected.keys()
        for name, shape in expected.items():
            assert live[name].shape == tuple(shape), name


def portable_fixture(tmp_path):
    cfg = MODEL_CFG
    model = build_model(cfg, nnx.Rngs(19))
    out = export_model(model, dataclasses.asdict(cfg), 123, tmp_path / "portable.safetensors")
    config = {
        "model": dataclasses.asdict(cfg),
        "run": {
            "checkpoint_dir": "/private/ckpt",
        },
        "prior": {"extensions": "quantize"},
    }
    out.with_suffix(".config.json").write_text(json.dumps(config))
    return model, out


def test_portable_restore_preserves_parameters_and_predictions_without_optimizer(
    tmp_path, monkeypatch
):
    model, out = portable_fixture(tmp_path)

    def forbid_optimizer(*args, **kwargs):
        raise AssertionError("portable inference must not rebuild an optimizer")

    monkeypatch.setattr(nnx, "Optimizer", forbid_optimizer)
    restored, criterion, cfg, step = load_eval_model(out)
    assert step == 123 and cfg["prior"]["extensions"] == "quantize"
    for name, tensor in model_tensors(model).items():
        np.testing.assert_array_equal(model_tensors(restored)[name], tensor)
    task = make_task()
    first = evaluate(model, criterion, [("t", task)], PROTO, config_dim=4, feature_keys=1)
    second = evaluate(restored, criterion, [("t", task)], PROTO, config_dim=4, feature_keys=1)
    pd.testing.assert_frame_equal(first, second)
    copied = export_run(out, out=tmp_path / "copied.safetensors")
    copied_cfg = json.loads(copied.with_suffix(".config.json").read_text())
    assert all(
        copied_cfg["run"].get(key) is None for key in ("checkpoint_dir",)
    )


def test_portable_metadata_must_agree_and_source_config_is_preserved(tmp_path):
    _, out = portable_fixture(tmp_path)
    original = out.with_suffix(".config.json").read_bytes()
    with pytest.raises(ValueError, match="separate file"):
        export_run(out, out=out)
    assert out.with_suffix(".config.json").read_bytes() == original
    with pytest.raises(ValueError, match="requested step"):
        load_eval_model(out, step=122)
    cfg = json.loads(original)
    cfg["model"]["config_dim"] += 1
    out.with_suffix(".config.json").write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="different models"):
        load_eval_model(out)


def test_curve_identity_checkpoint_is_rejected():
    from icicle_jax.loading import strip_retired_model_keys

    config = dataclasses.asdict(MODEL_CFG)
    config["curve_identity"] = "learned"
    with pytest.raises(ValueError, match="curve identities"):
        strip_retired_model_keys(config)


def test_direct_export_loads_without_training_sidecar(tmp_path):
    model = build_model(MODEL_CFG, nnx.Rngs(0))
    out = export_model(model, dataclasses.asdict(MODEL_CFG), 0, tmp_path / "direct.safetensors")
    restored, _, config, step = load_eval_model(out)
    assert step == 0
    assert config["model"] == dataclasses.asdict(MODEL_CFG)
    for name, expected in model_tensors(model).items():
        np.testing.assert_array_equal(model_tensors(restored)[name], expected)
    copied = export_run(out, out=tmp_path / "copy.safetensors")
    assert load_eval_model(copied)[3] == 0
