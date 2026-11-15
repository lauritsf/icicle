"""OnnxSurrogate over a synthetic export -- jax-free (numpy + onnx + onnxruntime)."""

import json

import numpy as np
import pytest

ort = pytest.importorskip("onnxruntime")
onnx = pytest.importorskip("onnx")

from icicle.model_format import expected_tensors, make_manifest  # noqa: E402
from icicle.onnx_export import build_model_graph  # noqa: E402
from icicle.serving import (  # noqa: E402
    OnnxSurrogate,
    orthonormal_frame,
)

CFG = {
    "config_dim": 6,
    "emsize": 16,
    "num_heads": 4,
    "hidden_dim": 32,
    "n_feat_layers": 1,
    "n_icl_layers": 2,
    "num_bars": 10,
    "qk_norm": True,
}


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("export")
    config = dict(CFG)
    rng = np.random.default_rng(0)
    tensors = {
        n: (rng.standard_normal(s) * 0.1).astype(np.float32)
        for n, s in expected_tensors(config).items()
    }
    manifest = make_manifest(config, step=0)
    (d / "icicle.json").write_text(json.dumps(manifest))
    (d / "icicle.onnx").write_bytes(build_model_graph(manifest, tensors).SerializeToString())
    (d / "icicle_fused.onnx").write_bytes(
        build_model_graph(manifest, tensors, fused=True).SerializeToString()
    )
    (d / "icicle_fused.json").write_text(json.dumps(manifest))
    return d / "icicle.onnx"


def _some_context(rng, n=8, q=3, h=6):
    return (
        rng.uniform(size=(n, h)).astype(np.float32),
        rng.uniform(0.1, 1.0, size=n).astype(np.float32),
        rng.uniform(size=n).astype(np.float32),
        rng.uniform(size=(q, h)).astype(np.float32),
        np.full(q, 1.0, np.float32),
    )


def test_probs_shape_and_normalization(model_path):
    s = OnnxSurrogate(model_path, seed=1, intra_op_num_threads=1)
    probs = s(*_some_context(np.random.default_rng(2)))
    assert probs.shape == (3, CFG["num_bars"])
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, rtol=1e-5)
    assert (probs >= 0).all()


def test_fixed_draws_make_calls_deterministic(model_path):
    args = _some_context(np.random.default_rng(3))
    s = OnnxSurrogate(model_path, seed=1, intra_op_num_threads=1)
    np.testing.assert_array_equal(s(*args), s(*args))
    np.testing.assert_array_equal(
        s(*args), OnnxSurrogate(model_path, seed=1, intra_op_num_threads=1)(*args)
    )
    assert not np.array_equal(
        s(*args), OnnxSurrogate(model_path, seed=9, intra_op_num_threads=1)(*args)
    )


def test_context_actually_conditions_the_prediction(model_path):
    s = OnnxSurrogate(model_path, seed=1, intra_op_num_threads=1)
    ctx_hps, ctx_fid, ctx_y, qry_hps, qry_fid = _some_context(np.random.default_rng(4))
    a = s(ctx_hps, ctx_fid, ctx_y, qry_hps, qry_fid)
    b = s(ctx_hps, ctx_fid, 1.0 - ctx_y, qry_hps, qry_fid)
    assert not np.array_equal(a, b)


def test_seen_predictions_are_independent_of_query_additions_and_removals(model_path):
    surrogate = OnnxSurrogate(model_path, seed=1, intra_op_num_threads=1)
    context = np.full((1, 6), 0.5, np.float32)
    query = np.full((1, 6), 0.7, np.float32)
    initial = surrogate(context, [0.1], [0.4], query, [0.8])
    expanded = np.concatenate([np.full((1, 6), 0.1, np.float32), query])
    together = surrogate(context, [0.1], [0.4], expanded, [0.8, 0.8])
    np.testing.assert_allclose(together[1:], initial, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(
        surrogate(context, [0.1], [0.4], query, [0.8]), initial, rtol=1e-6, atol=1e-7
    )




def test_guards(model_path):
    s = OnnxSurrogate(model_path, seed=1, intra_op_num_threads=1)
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError, match="context"):
        s(np.zeros((0, 6)), [], [], rng.uniform(size=(1, 6)), [1.0])
    with pytest.raises(ValueError, match="frame"):
        wide = rng.uniform(size=(2, 12)).astype(np.float32)
        s(wide, [0.5, 0.5], [0.1, 0.2], wide[:1], [1.0])


def test_orthonormal_frame_properties():
    f = orthonormal_frame(np.random.default_rng(0), 6, 16)
    gram = (f / np.sqrt(16 / 2.0)) @ (f / np.sqrt(16 / 2.0)).T
    np.testing.assert_allclose(gram, np.eye(6), atol=1e-5)


def test_injected_frame_may_cover_more_rows_than_the_declared_width(model_path):
    frame = orthonormal_frame(np.random.default_rng(11), 8, CFG["emsize"])
    surrogate = OnnxSurrogate(
        model_path,
        config_dim=6,
        frame=frame,
        intra_op_num_threads=1,
    )
    assert surrogate._frame.shape == (8, CFG["emsize"])


def test_pickle_round_trip(model_path):
    import pickle

    s = OnnxSurrogate(model_path, seed=1, intra_op_num_threads=1)
    args = _some_context(np.random.default_rng(6))
    ref = s(*args)
    s2 = pickle.loads(pickle.dumps(s))
    np.testing.assert_array_equal(s2(*args), ref)
    assert s.session.get_session_options().intra_op_num_threads == 1
    assert s2.session.get_session_options().intra_op_num_threads == 1


def test_fused_graph_serves(model_path):
    """The fused graph returns valid distributions without a JAX dependency.

    Logit agreement is tested separately in test_onnx_parity.py. This fixture
    uses random weights, so its sharp softmax is unsuitable for probability
    comparisons across differently fused graphs.
    """
    args = _some_context(np.random.default_rng(7))
    s = OnnxSurrogate(model_path.with_name("icicle_fused.onnx"), seed=1, intra_op_num_threads=1)
    probs = s(*args)
    assert probs.shape == (3, CFG["num_bars"])
    np.testing.assert_allclose(probs.sum(axis=-1), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(probs, s(*args))


def test_named_models_share_a_directory(model_path):
    import shutil

    small = model_path.with_name("icicle-small.onnx")
    shutil.copyfile(model_path, small)
    metadata = json.loads(model_path.with_suffix(".json").read_text())
    metadata["model"]["config_dim"] = 3
    small.with_suffix(".json").write_text(json.dumps(metadata))

    default = OnnxSurrogate(model_path, intra_op_num_threads=1)
    named = OnnxSurrogate(small, intra_op_num_threads=1)
    context = _some_context(np.random.default_rng(0), h=4)
    assert default(*context).shape == (3, CFG["num_bars"])
    with pytest.raises(ValueError, match="HP count"):
        named(*context)
    assert named(*_some_context(np.random.default_rng(1), h=3)).shape == (3, CFG["num_bars"])


@pytest.mark.parametrize("providers", [[], [("CPUExecutionProvider", {})]])
def test_provider_arguments(model_path, providers):
    from icicle.serving import make_session

    session = make_session(model_path, providers, intra_op_num_threads=1)
    assert session.get_providers()[0] == "CPUExecutionProvider"


def test_provider_options_override_defaults(monkeypatch):
    from types import SimpleNamespace
    from icicle.serving import make_session

    received = []

    def create_session(path, *, sess_options, providers):
        received.extend(providers)
        return SimpleNamespace(get_providers=lambda: ["CUDAExecutionProvider"])

    monkeypatch.setattr(ort, "InferenceSession", create_session)
    make_session("unused.onnx", [("CUDAExecutionProvider", {"device_id": "2"})])
    assert received == [("CUDAExecutionProvider", {"use_tf32": "0", "device_id": "2"})]
    received.clear()
    make_session("unused.onnx", [("CUDAExecutionProvider", {"use_tf32": "1"})])
    assert received == [("CUDAExecutionProvider", {"use_tf32": "1"})]
    with pytest.raises(RuntimeError, match="unavailable"):
        make_session("unused.onnx", ["ROCMExecutionProvider"])


@pytest.mark.parametrize("config_dim", [0, -1])
def test_invalid_config_dim(model_path, config_dim):
    with pytest.raises(ValueError, match="config_dim must be positive"):
        OnnxSurrogate(model_path, config_dim=config_dim, intra_op_num_threads=1)


def test_relative_model_path_survives_restore_in_another_directory(model_path, tmp_path, monkeypatch):
    import pickle

    monkeypatch.chdir(model_path.parent)
    model = OnnxSurrogate(model_path.name, intra_op_num_threads=1)
    inputs = _some_context(np.random.default_rng(4))
    expected = model(*inputs)
    serialized = pickle.dumps(model)
    monkeypatch.chdir(tmp_path)
    restored = pickle.loads(serialized)
    np.testing.assert_array_equal(restored(*inputs), expected)
