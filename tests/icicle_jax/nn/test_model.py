"""Tests for the full ICICLE forward -- the PFN in-context invariants at the model level."""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from icicle_jax.bar_distribution import BarDistribution
from icicle_jax.nn.model import Surrogate

jax.config.update("jax_enable_x64", True)

FK = jax.random.key(1)


def _model(**kw):
    # Pin xla: these are exact numerical-parity / invariance checks (CPU / float64), independent
    # of the attention kernel; the flash path is covered in test_flash.py.
    return Surrogate(
        emsize=16,
        num_heads=4,
        hidden_dim=32,
        n_feat_layers=1,
        n_icl_layers=2,
        num_bars=8,
        icl_backend="xla",
        param_dtype=jnp.float64,
        rngs=nnx.Rngs(0),
        **kw,
    )


def _inputs(rng, b=2, c=4, h=3, p=10):
    return {
        "config_hps": jnp.asarray(rng.uniform(size=(b, c, h))),
        "num_params": jnp.array(h),
        "point_config_idx": jnp.asarray(rng.integers(0, c, size=(b, p))),
        "fidelities": jnp.asarray(rng.uniform(size=(b, p))),
        "y": jnp.asarray(rng.uniform(size=(b, p))),
    }


def test_forward_shape():
    rng = np.random.default_rng(0)
    out = _model()(**_inputs(rng), single_eval_pos=6, feature_key=FK)
    assert out.shape == (2, 10, 8)
    assert np.all(np.isfinite(np.asarray(out)))


def test_context_permutation_invariance():
    rng = np.random.default_rng(1)
    m = _model()
    inp, sep = _inputs(rng), 6
    out = m(**inp, single_eval_pos=sep, feature_key=FK)

    perm = rng.permutation(sep)
    inp2 = dict(inp)
    for k in ("point_config_idx", "fidelities", "y"):
        inp2[k] = inp[k].at[:, :sep].set(inp[k][:, :sep][:, perm])
    out2 = m(**inp2, single_eval_pos=sep, feature_key=FK)

    # invariance is structural; residual is GPU reduction-order noise from reordering the sum
    np.testing.assert_allclose(
        np.asarray(out2[:, sep:]), np.asarray(out[:, sep:]), rtol=1e-5, atol=1e-6
    )


def test_no_query_to_query_leakage():
    rng = np.random.default_rng(2)
    m = _model()
    inp, sep, j = _inputs(rng), 6, 8
    out = m(**inp, single_eval_pos=sep, feature_key=FK)

    inp2 = dict(inp)
    inp2["fidelities"] = inp["fidelities"].at[:, j].set(0.123)
    out2 = m(**inp2, single_eval_pos=sep, feature_key=FK)

    keep = jnp.arange(10) != j
    np.testing.assert_allclose(np.asarray(out2[:, keep]), np.asarray(out[:, keep]), atol=1e-9)
    assert not np.allclose(np.asarray(out2[:, j]), np.asarray(out[:, j]))


def test_nll_runs():
    rng = np.random.default_rng(3)
    inp, sep = _inputs(rng), 6
    logits = _model()(**inp, single_eval_pos=sep, feature_key=FK)
    bar = BarDistribution(jnp.array([0.0, 0.02, 0.08, 0.2, 0.4, 0.5, 0.7, 0.9, 1.0]))
    nll = bar.nll(logits[:, sep:], inp["y"][:, sep:])
    assert nll.shape == (2, 4)
    assert np.all(np.isfinite(np.asarray(nll)))










def test_remat_and_feat_chunk_match_plain():
    """Compare rematerialized and chunked evaluation with the unchunked model.

    A final partial chunk exercises padding and gradient propagation. Attention
    softmax uses float32 even in this float64 test, so chunk-dependent reduction
    order requires a looser tolerance than rematerialization alone.
    """
    rng = np.random.default_rng(4)
    inp, sep = _inputs(rng), 6
    plain = _model()
    remat_only = _model(remat=True)
    config_remat_only = _model(remat=False, config_remat=True)
    levered = _model(remat=True, feat_chunk=3)

    out_p = plain(**inp, single_eval_pos=sep, feature_key=FK)
    out_r = remat_only(**inp, single_eval_pos=sep, feature_key=FK)
    out_cr = config_remat_only(**inp, single_eval_pos=sep, feature_key=FK)
    out_l = levered(**inp, single_eval_pos=sep, feature_key=FK)
    np.testing.assert_array_equal(np.asarray(out_r), np.asarray(out_p))  # remat: bit-exact
    np.testing.assert_array_equal(np.asarray(out_cr), np.asarray(out_p))
    np.testing.assert_allclose(np.asarray(out_l), np.asarray(out_p), rtol=1e-4, atol=1e-5)

    def loss_fn(m):
        return (m(**inp, single_eval_pos=sep, feature_key=FK) ** 2).sum()

    # Chunking's forward fp32-softmax difference is amplified through the ICL stack's softmax
    # Jacobians. Its bound is a sanity check, while remat gets a much tighter float64 bound.
    g_p = jax.tree.leaves(nnx.grad(loss_fn)(plain))
    g_cr = jax.tree.leaves(nnx.grad(loss_fn)(config_remat_only))
    g_l = jax.tree.leaves(nnx.grad(loss_fn)(levered))
    for a, b in zip(g_p, g_cr, strict=True):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-12, atol=1e-12)
    for a, b in zip(g_p, g_l, strict=True):
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-4, atol=1e-4)
