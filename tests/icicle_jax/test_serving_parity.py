"""Compare ONNX and JAX predictions using the same context and feature frame."""

import jax
import numpy as np
import pytest
from flax import nnx

ort = pytest.importorskip("onnxruntime")

from icicle.serving import OnnxSurrogate  # noqa: E402
from icicle_jax.bar_distribution import BarDistribution  # noqa: E402
from icicle_jax.export import export_model  # noqa: E402
from icicle_jax.hpo import make_raw_predict_fn  # noqa: E402
from icicle_jax.nn.config_encoder import feature_identity  # noqa: E402
from icicle_jax.loading import build_model

from icicle.onnx_export import export_onnx  # noqa: E402  isort: skip
from tests.reproduction.jax.test_eval import MODEL_CFG  # noqa: E402


def test_onnx_surrogate_matches_raw_predict_fn(tmp_path):
    old = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", False)
    try:
        import dataclasses
        import json

        model = build_model(MODEL_CFG, nnx.Rngs(0))
        weights_path = export_model(model, dataclasses.asdict(MODEL_CFG), 0, tmp_path / "small.safetensors")
        onnx_path = export_onnx(weights_path)
        assert onnx_path.exists() and json.loads(weights_path.with_suffix(".json").read_text())

        h, e = MODEL_CFG.config_dim, MODEL_CFG.emsize
        criterion = BarDistribution(np.linspace(0.0, 1.0, MODEL_CFG.num_bars + 1))
        jax_fn = make_raw_predict_fn(
            model, criterion, config_dim=h, ctx_pad=16, qry_pad=8, cfg_pad=16
        )
        fkey = jax.random.fold_in(jax.random.key(0), 0)
        frame = np.asarray(feature_identity(fkey, 1, h, e)[0], np.float32)
        surrogate = OnnxSurrogate(onnx_path, frame=frame)

        rng = np.random.default_rng(0)
        ctx_hps = rng.uniform(size=(9, h)).astype(np.float32)
        ctx_fid = rng.uniform(0.1, 1.0, size=9).astype(np.float32)
        ctx_y = rng.uniform(size=9).astype(np.float32)
        qry_hps = np.concatenate([ctx_hps[:2], rng.uniform(size=(3, h)).astype(np.float32)])
        qry_fid = np.full(5, 1.0, np.float32)

        ref = jax_fn(ctx_hps, ctx_fid, ctx_y, qry_hps, qry_fid)
        out = surrogate(ctx_hps, ctx_fid, ctx_y, qry_hps, qry_fid)
        np.testing.assert_allclose(out, ref, rtol=2e-4, atol=2e-4)
    finally:
        jax.config.update("jax_enable_x64", old)
