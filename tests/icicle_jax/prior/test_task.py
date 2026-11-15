"""Tests for factored task assembly (shapes/ranges, batching, and feeding the model)."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from icicle_jax.nn.model import Surrogate
from icicle_jax.prior.bnn import build_output_sorted
from icicle_jax.prior.task import sample_task

jax.config.update("jax_enable_x64", True)

LAYERS, HIDDEN = (2, 4), (8, 16)
TABLE = build_output_sorted(
    jax.random.key(0),
    config_dim=4,
    layers=LAYERS,
    hidden=HIDDEN,
    num_networks=128,
    configs_per_network=32,
)
KW = dict(
    seq_len=12,
    config_dim=4,
    num_params=jnp.array(3),
    n_steps=jnp.array(6),
    single_eval_pos=jnp.array(8),
    table=TABLE,
    layers=LAYERS,
    hidden=HIDDEN,
    n_steps_max=8,
)


def test_sample_task_shapes_and_ranges():
    task = sample_task(jax.random.key(1), **KW)
    assert task.config_hps.shape == (12, 4)
    assert task.point_config_idx.shape == (12,)
    assert task.fidelities.shape == (12,) and task.y.shape == (12,)
    y, f, idx = np.asarray(task.y), np.asarray(task.fidelities), np.asarray(task.point_config_idx)
    assert np.all((y >= 0.0) & (y <= 1.0))  # observations clipped to [0,1]
    assert np.all((f > 0.0) & (f <= 1.0))  # fidelities in (0,1]
    assert np.all((idx >= 0) & (idx < 12))  # valid config indices
    assert np.all(np.asarray(task.config_hps)[:, 3] == 0.0)  # unused column (>= num_params) zeroed


def test_batched_tasks():
    batch = jax.vmap(partial(sample_task, **KW))(jax.random.split(jax.random.key(2), 4))
    assert batch.config_hps.shape == (4, 12, 4)
    assert batch.point_config_idx.shape == (4, 12)
    assert batch.fidelities.shape == (4, 12) and batch.y.shape == (4, 12)


def test_feeds_surrogate():
    batch = jax.vmap(partial(sample_task, **KW))(jax.random.split(jax.random.key(3), 2))
    model = Surrogate(
        emsize=16,
        num_heads=4,
        hidden_dim=32,
        n_feat_layers=1,
        n_icl_layers=2,
        num_bars=8,
        icl_backend="xla",
        param_dtype=jnp.float64,
        rngs=nnx.Rngs(0),
    )
    logits = model(
        batch.config_hps,
        jnp.array(3),
        batch.point_config_idx,
        batch.fidelities,
        batch.y,
        single_eval_pos=8,
        feature_key=jax.random.key(4),
    )
    assert logits.shape == (2, 12, 8)
    assert np.all(np.isfinite(np.asarray(logits)))


def test_recalled_queries_reuse_observed_values():
    from icicle_jax.prior.observations import ObservationConfig

    sample = jax.jit(
        lambda key: sample_task(
            key,
            **KW,
            recall_probability=1.0,
            observation_config=ObservationConfig(quantize=True, noise_tails=True, divergence=True),
        )
    )
    recalled = 0
    for seed in range(4):
        task = sample(jax.random.key(seed))
        configs, fidelities, values = map(
            np.asarray, (task.point_config_idx, task.fidelities, task.y)
        )
        for query in range(8, 12):
            matches = (configs[:8] == configs[query]) & (fidelities[:8] == fidelities[query])
            if matches.any():
                np.testing.assert_array_equal(values[:8][matches], values[query])
                recalled += 1
    assert recalled > 0
