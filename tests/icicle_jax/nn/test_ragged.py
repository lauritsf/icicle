"""Tests for the ragged config axis: dense-parity + the packing mechanics."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from icicle_jax.nn.model import Surrogate
from icicle_jax.nn.ragged import (
    count_referenced_configs,
    next_pow2,
    pack_referenced_configs,
)
from icicle_jax.prior.bnn import build_output_sorted
from icicle_jax.prior.task import sample_task

jax.config.update("jax_enable_x64", True)

SEQ_LEN, CONFIG_DIM, SEP, NUM_PARAMS = 20, 4, 12, 3


def _batch(bs=3):
    table = build_output_sorted(
        jax.random.key(0),
        config_dim=CONFIG_DIM,
        layers=(2, 4),
        hidden=(8, 16),
        num_networks=64,
        configs_per_network=16,
    )
    sample = partial(
        sample_task,
        seq_len=SEQ_LEN,
        config_dim=CONFIG_DIM,
        num_params=jnp.array(NUM_PARAMS),
        n_steps=jnp.array(6),
        single_eval_pos=jnp.array(SEP),
        table=table,
        layers=(2, 4),
        hidden=(8, 16),
        n_steps_max=8,
    )
    return jax.vmap(sample)(jax.random.split(jax.random.key(1), bs))


def _model(*, feat_chunk=0, config_remat=False, dtype=None):
    # Use XLA to compare dense and ragged packing. Softmax runs in float32,
    # so differently shaped reductions require a 1e-6 tolerance.
    return Surrogate(
        emsize=16,
        num_heads=4,
        hidden_dim=32,
        n_feat_layers=1,
        n_icl_layers=2,
        num_bars=8,
        icl_backend="xla",
        feat_chunk=feat_chunk,
        config_remat=config_remat,
        dtype=dtype,
        param_dtype=jnp.float64,
        rngs=nnx.Rngs(0),
    )


def test_next_pow2():
    assert [next_pow2(x) for x in (1, 2, 3, 5, 100)] == [1, 2, 4, 8, 128]


def test_ragged_encodes_fewer_configs():
    batch = _batch()
    total = count_referenced_configs(batch.point_config_idx, SEQ_LEN)
    dense = batch.point_config_idx.size  # B*seq_len: what the dense path encodes
    assert 0 < total < dense  # fails if every point referenced a distinct config (no recurrence)


def test_pack_point_slot_roundtrip():
    config_hps = jnp.arange(2 * 5 * 3, dtype=jnp.float64).reshape(2, 5, 3)
    point_config_idx = jnp.array([[0, 0, 2, 2, 4, 4], [1, 3, 1, 3, 1, 3]])  # 3 + 2 distinct
    n_pad = next_pow2(count_referenced_configs(point_config_idx, 5))
    flat_hps, flat_task, point_slot, valid = pack_referenced_configs(
        config_hps, point_config_idx, n_pad
    )
    gathered = flat_hps[point_slot]
    expected = jnp.take_along_axis(config_hps, point_config_idx[..., None], axis=1)
    np.testing.assert_array_equal(np.asarray(gathered), np.asarray(expected))
    assert int(valid.sum()) == count_referenced_configs(point_config_idx, 5)


def test_ragged_matches_dense():
    batch, model, fk = _batch(), _model(), jax.random.key(7)
    num_params = jnp.array(NUM_PARAMS)
    dense = model(
        batch.config_hps,
        num_params,
        batch.point_config_idx,
        batch.fidelities,
        batch.y,
        SEP,
        feature_key=fk,
    )
    n_pad = next_pow2(count_referenced_configs(batch.point_config_idx, SEQ_LEN))
    ragged = model.forward_ragged(
        batch.config_hps,
        num_params,
        batch.point_config_idx,
        batch.fidelities,
        batch.y,
        SEP,
        n_pad,
        feature_key=fk,
    )
    np.testing.assert_allclose(np.asarray(ragged), np.asarray(dense), rtol=1e-6, atol=1e-6)


def test_ragged_grouped_matches_dense():
    batch, model, fk = _batch(bs=4), _model(feat_chunk=2), jax.random.key(7)
    num_params = jnp.array(NUM_PARAMS)
    dense = model(
        batch.config_hps,
        num_params,
        batch.point_config_idx,
        batch.fidelities,
        batch.y,
        SEP,
        feature_key=fk,
    )
    n_pad = next_pow2(count_referenced_configs(batch.point_config_idx, SEQ_LEN, groups=2))
    grouped = model.forward_ragged(
        batch.config_hps,
        num_params,
        batch.point_config_idx,
        batch.fidelities,
        batch.y,
        SEP,
        n_pad,
        feature_key=fk,
        groups=2,
    )
    np.testing.assert_allclose(np.asarray(grouped), np.asarray(dense), rtol=1e-6, atol=1e-6)


def test_empty_chunk_dispatch_supports_bfloat_remat_gradients():
    batch, model, fk = (
        _batch(bs=4),
        _model(feat_chunk=2, config_remat=True, dtype=jnp.bfloat16),
        jax.random.key(7),
    )
    n_valid = count_referenced_configs(batch.point_config_idx, SEQ_LEN)
    n_pad = next_pow2(n_valid) * 2
    assert n_valid <= n_pad - model.config_encoder._chunk_width(n_pad)

    @nnx.jit
    def loss_and_grad(current):
        return nnx.value_and_grad(
            lambda candidate: candidate.forward_ragged(
                batch.config_hps,
                jnp.array(NUM_PARAMS),
                batch.point_config_idx,
                batch.fidelities,
                batch.y,
                SEP,
                n_pad,
                feature_key=fk,
            ).sum()
        )(current)

    loss, gradients = loss_and_grad(model)
    assert np.isfinite(np.asarray(loss)).all()
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(gradients))


def test_padding_chunks_match_dense_encoding():
    batch, model, fk = _batch(bs=4), _model(feat_chunk=2), jax.random.key(7)
    n_valid = count_referenced_configs(batch.point_config_idx, SEQ_LEN)
    n_pad = next_pow2(n_valid) * 2
    args = (
        batch.config_hps,
        jnp.array(NUM_PARAMS),
        batch.point_config_idx,
        batch.fidelities,
        batch.y,
        SEP,
        n_pad,
    )
    skipped = model.forward_ragged(*args, feature_key=fk)
    dense_chunks = model(*args[:-1], feature_key=fk)
    np.testing.assert_allclose(np.asarray(skipped), np.asarray(dense_chunks), rtol=1e-6, atol=1e-6)
