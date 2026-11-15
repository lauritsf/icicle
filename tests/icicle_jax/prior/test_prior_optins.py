"""Typed hyperparameters and observation noise in the synthetic prior."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from icicle_jax.prior.bnn import build_output_sorted
from icicle_jax.prior.observations import parse_observations
from icicle_jax.prior.task import (
    CONTINUOUS,
    NOMINAL,
    ORDINAL,
    _sample_configs,
    _sample_hp_types,
    sample_task,
    truncated_poisson,
)

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
TRIO = parse_observations("divergence,noise_tails,quantize")


def test_truncated_poisson_support_and_mean():
    d = np.asarray(truncated_poisson(jax.random.key(3), jnp.array(2.0), 22, shape=(20000,)))
    assert d.min() >= 0 and d.max() <= 22
    assert abs(d.mean() - 2.0) < 0.1  # truncation at 22 is negligible for lam=2
    hi = np.asarray(truncated_poisson(jax.random.key(4), jnp.array(20.0), 9, shape=(20000,)))
    assert hi.max() <= 9  # hard truncation honoured even when lam >> max_part


def test_typed_hp_level_values_by_kind():
    types = _sample_hp_types(jax.random.key(5), 64)
    kind = np.asarray(types.kind)
    n_levels = np.asarray(types.n_levels)
    values = np.asarray(types.level_values)
    assert set(np.unique(kind)) <= {CONTINUOUS, ORDINAL, NOMINAL}
    assert n_levels.min() >= 2 and n_levels.max() <= 24
    for j in np.where(kind == ORDINAL)[0]:
        k = n_levels[j]
        np.testing.assert_allclose(values[j, :k], np.linspace(0.0, 1.0, k), atol=1e-12)
    # nominal rows are IID uniforms -- astronomically unlikely to be sorted for k >= 4
    nom = [j for j in np.where(kind == NOMINAL)[0] if n_levels[j] >= 4]
    if nom:
        assert any(not np.all(np.diff(values[j, : n_levels[j]]) >= 0) for j in nom)


def test_typed_hp_configs_snap_to_level_values():
    types = _sample_hp_types(jax.random.key(6), 8)
    cfg = np.asarray(_sample_configs(jax.random.key(7), types, 500, 8))
    kind, n_levels = np.asarray(types.kind), np.asarray(types.n_levels)
    values = np.asarray(types.level_values)
    for j in range(8):
        col = np.unique(cfg[:, j])
        if kind[j] == CONTINUOUS:
            assert len(col) > 400  # fresh uniforms
        else:
            assert len(col) <= n_levels[j]
            assert np.isin(col, values[j, : n_levels[j]]).all()


def test_sample_task_with_typed_hyperparameters_and_extensions():
    config = dataclasses.replace(TRIO, tail_scale=(0.01, 8.0))
    t = sample_task(
        jax.random.key(11),
        seq_len=24,
        config_dim=4,
        num_params=jnp.array(3),
        n_steps=jnp.array(6),
        single_eval_pos=jnp.array(12),
        table=TABLE,
        layers=LAYERS,
        hidden=HIDDEN,
        n_steps_max=8,
        observation_config=config,
        typed_hyperparameters=True,
    )
    assert np.isfinite(np.asarray(t.y)).all()
    assert (np.asarray(t.y) >= 0).all() and (np.asarray(t.y) <= 1).all()
