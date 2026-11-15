"""Tests for the config-encoding subsystem: tokenizer (cells) + encoder (e_c)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from icicle_jax.nn.config_encoder import ConfigEncoder, ConfigTokenizer, feature_identity

jax.config.update("jax_enable_x64", True)

KEY = jax.random.key(1)


# --- tokenizer -------------------------------------------------------------------------------


def test_tokenizer_shapes_and_mask():
    rng = np.random.default_rng(0)
    hps = jnp.asarray(rng.uniform(size=(2, 3, 5)))
    num_params = jnp.array(3)
    cells, mask = ConfigTokenizer(16)(hps, num_params, feature_key=KEY)

    assert cells.shape == (2, 3, 5, 16)
    assert mask.shape == (2, 3, 5)
    assert bool(mask[:, :, :3].all()) and not bool(mask[:, :, 3:].any())
    with pytest.raises(ValueError, match="shared scalar"):
        ConfigTokenizer(16)(hps, jnp.array([3, 5]), feature_key=KEY)


def test_tokenizer_feature_binding_is_an_invertible_sketch():
    rng = np.random.default_rng(0)
    hps = jnp.asarray(rng.uniform(size=(1, 1, 4)))
    cells, _ = ConfigTokenizer(16)(hps, jnp.array(4), feature_key=KEY)
    u = feature_identity(KEY, 1, 4, 16)[0]  # (4, 16)
    pooled = np.asarray(cells[0, 0].sum(0))
    recovered = pooled @ np.asarray(u).T / (16 / 2)  # project onto each slot's direction
    np.testing.assert_allclose(recovered, np.asarray((hps[0, 0] - 0.5) * jnp.sqrt(12.0)), atol=1e-9)

    centred = jnp.full((1, 1, 1), 0.5)
    cells, _ = ConfigTokenizer(16)(centred, jnp.array(1), feature_key=KEY)
    np.testing.assert_allclose(np.asarray(cells), 0.0, atol=1e-12)


def test_tokenizer_feature_binding_flat_matches_dense():
    rng = np.random.default_rng(1)
    hps = jnp.asarray(rng.uniform(size=(2, 3, 5)))
    tok = ConfigTokenizer(16)
    dense, _ = tok(hps, jnp.array(5), feature_key=KEY)
    flat, _ = tok.tokenize_flat(
        hps.reshape(6, 5), jnp.repeat(jnp.arange(2), 3), jnp.array(5), 2, feature_key=KEY
    )
    np.testing.assert_array_equal(np.asarray(dense.reshape(6, 5, 16)), np.asarray(flat))


def test_feature_identity_orthonormal_and_per_task():
    u = feature_identity(KEY, 3, 5, 16)  # (3 tasks, 5 slots, dim 16)
    assert u.shape == (3, 5, 16)
    gram = np.asarray(u[0] @ u[0].T)  # (5, 5)
    off = gram - np.diag(np.diag(gram))
    assert np.abs(off).max() < 1e-8  # orthogonal within a task
    np.testing.assert_allclose(np.diag(gram), 16 / 2, rtol=1e-6)  # equal norm^2 = dim/2
    assert not np.allclose(np.asarray(u[0]), np.asarray(u[1]))  # tasks differ


# --- encoder ---------------------------------------------------------------------------------


def _enc(**kw):
    return ConfigEncoder(
        16, num_heads=4, hidden_dim=32, param_dtype=jnp.float64, rngs=nnx.Rngs(0), **kw
    )


def test_encoder_shapes_and_output_norm():
    rng = np.random.default_rng(0)
    cells = jnp.asarray(rng.normal(size=(2, 3, 5, 16)))
    mask = jnp.ones((2, 3, 5), dtype=bool)
    e_c = _enc()(cells, mask)
    assert e_c.shape == (2, 3, 16)
    # RMSNorm -> norm sqrt(emsize) up to the normalization eps
    np.testing.assert_allclose(np.asarray(jnp.linalg.norm(e_c, axis=-1)), np.sqrt(16), rtol=1e-3)


def test_encoder_permutation_invariant_over_complete_tokenized_cells():
    """Jointly reordering tokenizer-produced complete cells and masks preserves e_c.

    Keep each value paired with its feature identity and use the same feature frame.
    """
    rng = np.random.default_rng(1)
    enc = _enc()
    tokenizer = ConfigTokenizer(16)
    for layout_width, active_width in ((4, 4), (7, 5), (12, 8), (16, 13)):
        hps = jnp.asarray(rng.uniform(size=(1, 3, layout_width)), dtype=jnp.float64)
        cells, mask = tokenizer(hps, active_width, feature_key=KEY)
        reference = enc(cells, mask)
        for _ in range(6):
            perm = rng.permutation(layout_width)
            reordered = enc(cells[:, :, perm], mask[:, :, perm])
            # The attention softmax stays f32 in the x64 harness, so this contract uses the
            # same 1e-6 ceiling as the production f32 path.
            np.testing.assert_allclose(
                np.asarray(reordered), np.asarray(reference), rtol=0.0, atol=1e-6
            )


def test_encoder_validity_mask_ignores_padded_cells():
    rng = np.random.default_rng(2)
    cells = jnp.asarray(rng.normal(size=(1, 1, 5, 16)))
    mask = jnp.asarray([[[True, True, True, False, False]]])  # last two invalid
    enc = _enc()
    e_c = enc(cells, mask)
    cells2 = cells.at[0, 0, 3].set(jnp.asarray(rng.normal(size=(16,))))
    e_c2 = enc(cells2, mask)
    np.testing.assert_allclose(np.asarray(e_c2), np.asarray(e_c), rtol=1e-6, atol=1e-8)
