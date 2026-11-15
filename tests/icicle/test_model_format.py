"""Check model-export tensor names, shapes and dtypes."""

import numpy as np
import pytest

from icicle.model_format import expected_tensors, make_manifest, validate

BASE = {
    "config_dim": 10,
    "emsize": 64,
    "num_heads": 4,
    "hidden_dim": 128,
    "n_feat_layers": 1,
    "n_icl_layers": 2,
    "num_bars": 100,
    "qk_norm": True,
}


def test_expected_tensors_shapes_by_hand():
    t = expected_tensors(BASE)
    assert t["icl_layers.0.attn.q.kernel"] == (64, 64)
    assert t["icl_layers.1.w_in.kernel"] == (64, 128)
    assert t["icl_layers.0.attn.q_norm.scale"] == (16,)
    assert t["config_encoder.pool.seed"] == (64,)
    assert t["config_encoder.feat_layers.0.attn.q.kernel"] == (64, 64)
    assert t["decoder_linear2.kernel"] == (128, 100)


def test_expected_tensors_qk_norm_conditional():
    t = expected_tensors(BASE | {"qk_norm": False})
    assert "icl_layers.0.attn.q_norm.scale" not in t
    assert "config_encoder.pool.attn.k_norm.scale" not in t


def test_final_norm_always_demanded():
    assert expected_tensors(BASE)["final_norm.scale"] == (64,)
    tensors = {n: np.zeros(s, dtype=np.float32) for n, s in expected_tensors(BASE).items()}
    validate(make_manifest(BASE, 0), tensors)
    with pytest.raises(ValueError, match="missing"):
        # a pre-final-norm export (no final_norm.scale tensor) is invalid by construction
        validate(
            make_manifest(BASE, 0), {k: v for k, v in tensors.items() if k != "final_norm.scale"}
        )


def test_validate_rejects_bad_sets_and_shapes():
    tensors = {n: np.zeros(s, dtype=np.float32) for n, s in expected_tensors(BASE).items()}
    manifest = make_manifest(BASE, 100)
    validate(manifest, tensors)  # the true set passes

    with pytest.raises(ValueError, match="missing"):
        validate(manifest, {k: v for k, v in tensors.items() if k != "config_encoder.pool.seed"})
    with pytest.raises(ValueError, match="extra"):
        validate(manifest, tensors | {"bogus": np.zeros(3, dtype=np.float32)})
    with pytest.raises(ValueError, match="shape"):
        validate(manifest, tensors | {"config_encoder.pool.seed": np.zeros(7, dtype=np.float32)})
    with pytest.raises(ValueError, match="float32"):
        validate(manifest, tensors | {"config_encoder.pool.seed": np.zeros(64, dtype=np.float64)})
    with pytest.raises(ValueError, match="format"):
        validate(manifest | {"format": "other"}, tensors)
