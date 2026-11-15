"""ONNX graph vs the jax forward: same export, same inputs, matching logits."""

import dataclasses

import jax
import numpy as np
import pytest
from flax import nnx

ort = pytest.importorskip("onnxruntime")

from icicle.model_format import make_manifest  # noqa: E402
from icicle.onnx_export import build_model_graph  # noqa: E402
from icicle_jax.export import model_tensors  # noqa: E402
from icicle_jax.nn.config_encoder import feature_identity  # noqa: E402
from icicle_jax.loading import build_model
from tests.reproduction.jax.test_eval import MODEL_CFG  # noqa: E402


def _f32_mode():
    # the ONNX side is f32; run the jax reference in plain f32 too
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    return old


def run_both(model_cfg, seed=0, n_cfg=5, n_pts=12, sep=7, fused=False, nonfinite=False):
    old = _f32_mode()
    try:
        model = build_model(model_cfg, nnx.Rngs(seed))
        tensors = model_tensors(model)
        onnx_model = build_model_graph(
            make_manifest(dataclasses.asdict(model_cfg), 0), tensors, fused=fused
        )

        rng = np.random.default_rng(seed)
        h = model_cfg.config_dim
        num_params = h - 1  # one padded column, so the validity mask is exercised
        hps = rng.uniform(size=(n_cfg, h)).astype(np.float32)
        hps[:, num_params:] = 0.0
        idx = rng.integers(0, n_cfg, size=n_pts).astype(np.int64)
        fid = rng.uniform(0.05, 1.0, size=n_pts).astype(np.float32)
        y = rng.uniform(size=n_pts).astype(np.float32)
        if nonfinite:
            y[:3] = [np.nan, np.inf, -np.inf]
            y[sep:sep + 3] = [np.nan, np.inf, -np.inf]
        fkey = jax.random.key(42)
        frame = np.asarray(feature_identity(fkey, 1, h, model_cfg.emsize)[0], dtype=np.float32)
        feeds = {
            "config_hps": hps,
            "feature_identity": frame,
            "num_params": np.array(num_params, np.int64),
            "point_config_idx": idx,
            "fidelities": fid,
            "y": y,
            "single_eval_pos": np.array(sep, np.int64),
        }

        ref = np.asarray(
            model(
                hps[None],
                num_params,
                idx[None].astype(np.int32),
                fid[None],
                y[None],
                sep,
                feature_key=fkey,
            )[0]
        )
        sess = ort.InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        (out,) = sess.run(["logits"], feeds)
        return ref, out
    finally:
        jax.config.update("jax_enable_x64", old)


def test_onnx_matches_jax_default_config():
    ref, out = run_both(MODEL_CFG)
    np.testing.assert_allclose(out, ref, rtol=2e-4, atol=2e-4)


def test_onnx_matches_jax_conditional_variant():
    variant = dataclasses.replace(MODEL_CFG, qk_norm=False)
    ref, out = run_both(variant, seed=3)
    np.testing.assert_allclose(out, ref, rtol=2e-4, atol=2e-4)




def test_export_onnx_writes_a_servable_artifact(tmp_path):
    from icicle.onnx_export import export_onnx
    from icicle_jax.export import export_model

    old = _f32_mode()
    try:
        model = build_model(MODEL_CFG, nnx.Rngs(0))
        out = export_model(model, dataclasses.asdict(MODEL_CFG), 7, tmp_path / "small.safetensors")
        onnx_path = export_onnx(out)
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        assert [o.name for o in sess.get_outputs()] == ["logits"]
    finally:
        jax.config.update("jax_enable_x64", old)


def test_onnx_query_independence():
    old = _f32_mode()
    try:
        model = build_model(MODEL_CFG, nnx.Rngs(0))
        tensors = model_tensors(model)
        onnx_model = build_model_graph(make_manifest(dataclasses.asdict(MODEL_CFG), 0), tensors)
        sess = ort.InferenceSession(
            onnx_model.SerializeToString(), providers=["CPUExecutionProvider"]
        )

        rng = np.random.default_rng(1)
        h = MODEL_CFG.config_dim
        hps = rng.uniform(size=(4, h)).astype(np.float32)
        fkey = jax.random.key(42)
        frame = np.asarray(feature_identity(fkey, 1, h, MODEL_CFG.emsize)[0], dtype=np.float32)

        base_idx = (np.arange(10) % 4).astype(np.int64)
        base_fid = np.linspace(0.1, 1.0, 10).astype(np.float32)
        base_y = np.linspace(0.2, 0.8, 10).astype(np.float32)

        def run(n_pts):
            idx, fid, y = base_idx[:n_pts], base_fid[:n_pts], base_y[:n_pts]
            (out,) = sess.run(
                ["logits"],
                {
                    "config_hps": hps,
                    "feature_identity": frame,
                    "num_params": np.array(h, np.int64),
                    "point_config_idx": idx,
                    "fidelities": fid,
                    "y": y,
                    "single_eval_pos": np.array(5, np.int64),
                },
            )
            return out

        a, b = run(9), run(10)  # same 5 context points; one extra query in b
        np.testing.assert_allclose(b[5:9], a[5:9], rtol=1e-5, atol=1e-6)
    finally:
        jax.config.update("jax_enable_x64", old)


@pytest.mark.parametrize("fused", [False, True])
def test_onnx_matches_jax_with_nonfinite_values(fused):
    reference, actual = run_both(MODEL_CFG, fused=fused, nonfinite=True)
    assert np.isfinite(reference).all()
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual, reference, rtol=2e-4, atol=2e-4)
