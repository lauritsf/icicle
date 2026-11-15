"""Extension flags: default-off parity, per-transform behavior, bounds, jit compilation."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from icicle_jax.prior.bnn import build_output_sorted
from icicle_jax.prior.observations import (
    ObservationConfig,
    ObservationNoise,
    apply_alive_window,
    observe,
)
from icicle_jax.prior.task import _sample_task_params, sample_task

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
ALL_ON = ObservationConfig(
    quantize=True,
    noise_tails=True,
    divergence=True,
)


def noise(**kv):
    base = dict(n_levels=1.0, df=10.0, scale=1.0)
    base.update(kv)
    keys = ("n_levels", "df", "scale")
    return ObservationNoise(*(jnp.array(base[k]) for k in keys))


def test_default_off_is_identity():
    a = sample_task(jax.random.key(1), **KW)
    b = sample_task(jax.random.key(1), observation_config=ObservationConfig(), **KW)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def test_flags_only_touch_y():
    off = sample_task(jax.random.key(2), **KW)
    on = sample_task(jax.random.key(2), observation_config=ALL_ON, **KW)
    np.testing.assert_array_equal(np.asarray(off.config_hps), np.asarray(on.config_hps))
    np.testing.assert_array_equal(np.asarray(off.point_config_idx), np.asarray(on.point_config_idx))
    np.testing.assert_array_equal(np.asarray(off.fidelities), np.asarray(on.fidelities))
    y = np.asarray(on.y)
    assert not np.array_equal(np.asarray(off.y), y)
    assert np.all((y >= 0.0) & (y <= 1.0))


def test_quantize_puts_values_on_the_grid():
    config = ObservationConfig(quantize=True)
    y = observe(
        jnp.linspace(0.05, 0.95, 256),
        jax.random.key(3),
        jnp.array(0.01),
        config,
        noise(n_levels=20),
    )
    y = np.asarray(y)
    np.testing.assert_allclose(y * 20, np.round(y * 20), atol=1e-9)
    assert np.all((y >= 0.0) & (y <= 1.0))


def test_noise_tails_heavy_but_unit_scale():
    config = ObservationConfig(noise_tails=True)
    n, sigma = 200000, 0.005
    y = observe(
        jnp.full(n, 0.5), jax.random.key(6), sigma, config, noise(df=5.0, scale=1.0)
    )
    r = np.asarray(y) - 0.5
    assert 0.9 * sigma < r.std() < 1.1 * sigma  # standardized draw keeps the scale
    excess_kurtosis = float(np.mean((r / r.std()) ** 4) - 3.0)
    assert excess_kurtosis > 2.0  # t5's is 6; Gaussian's is 0


def test_bounds_under_adversarial_noise():
    y = observe(
        jnp.linspace(0.0, 1.0, 512),
        jax.random.key(7),
        jnp.array(0.05),
        ALL_ON,
        noise(n_levels=3, df=2.2, scale=4.0),
    )
    y = np.asarray(y)
    assert np.all((y >= 0.0) & (y <= 1.0))
    np.testing.assert_allclose(y * 3, np.round(y * 3), atol=1e-9)


def test_apply_alive_window_clamp_and_noise_kill():
    fid = jnp.linspace(0.1, 1.0, 10)
    y = jnp.full(10, 0.8)
    # dead start that revives at 0.35 and crashes back at 0.75
    out, keep = apply_alive_window(
        y, fid, jnp.array(0.3), jnp.array(0.35), jnp.array(0.75), jnp.array(0.5)
    )
    out, keep = np.asarray(out), np.asarray(keep)
    alive = (np.asarray(fid) >= 0.35) & (np.asarray(fid) < 0.75)
    np.testing.assert_allclose(out[alive], 0.8)
    np.testing.assert_allclose(out[~alive], 0.15)  # v_frac * y0, exactly constant
    np.testing.assert_allclose(keep, np.where(alive, 1.0, 0.0))
    # non-diverged config = full window: exact identity
    out2, keep2 = apply_alive_window(
        y, fid, jnp.array(0.3), jnp.array(-1.0), jnp.array(2.0), jnp.array(0.0)
    )
    np.testing.assert_array_equal(np.asarray(out2), 0.8)
    np.testing.assert_allclose(np.asarray(keep2), 1.0)
    # empty window = dead forever
    out3, _ = apply_alive_window(
        y, fid, jnp.array(0.3), jnp.array(1.1), jnp.array(2.0), jnp.array(0.0)
    )
    np.testing.assert_array_equal(np.asarray(out3), 0.0)


def test_task_quantize_end_to_end():
    task = sample_task(
        jax.random.key(8), **KW,
        observation_config=ObservationConfig(quantize=True, quantize_levels=(50.0, 50.0)),
    )
    y = np.asarray(task.y)
    np.testing.assert_allclose(y * 50, np.round(y * 50), atol=1e-9)


def test_task_divergence_zero_floor():
    sample = partial(
        sample_task, **KW,
        observation_config=ObservationConfig(divergence=True, div_prob=(1.0, 1.0), div_zero=1.0),
    )
    batch = jax.vmap(sample)(jax.random.split(jax.random.key(13), 8))
    assert np.count_nonzero(np.asarray(batch.y) == 0.0) > 0


def test_divergence_participation_bnn_driven():
    config = ObservationConfig(divergence=True, div_prob=(0.3, 0.3))
    fracs = []
    for seed in range(24):
        tp = _sample_task_params(
            jax.random.key(seed),
            config_dim=4,
            num_params=jnp.array(3),
            n_configs=64,
            table=TABLE,
            layers=LAYERS,
            hidden=HIDDEN,
            observation_config=config,
        )
        p = tp.curves
        is_div = (np.asarray(p.div_alive) > -1.0) | (np.asarray(p.div_die) < 2.0)
        fracs.append(float(is_div.mean()))
    assert 0.15 < np.mean(fracs) < 0.45  # pooled participation ~ div_prob (pinned 0.3)
    assert np.std(fracs) > 0.03  # per-task clustering from the BNN's shared weights


def test_jit_vmap_compiles_with_all_flags():
    sample = jax.jit(jax.vmap(partial(sample_task, observation_config=ALL_ON, **KW)))
    batch = sample(jax.random.split(jax.random.key(9), 4))
    assert batch.y.shape == (4, 12)
    assert np.all(np.isfinite(np.asarray(batch.y)))
