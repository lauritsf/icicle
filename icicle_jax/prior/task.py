"""Sample hyperparameters, curve parameters and observations for a training task."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from icicle_jax.prior.bnn import bnn_forward, sample_bnn, standardize_uniform
from icicle_jax.prior.curves import (
    CurveParams,
    extract_curve_params,
    learning_curve,
    num_bnn_outputs,
)
from icicle_jax.prior.observations import (
    DEFAULT_OBSERVATIONS,
    ObservationConfig,
    ObservationNoise,
    apply_alive_window,
    observe,
    sample_config_indices,
    sample_fidelities,
)


CONTINUOUS, ORDINAL, NOMINAL = 0, 1, 2

# Log-uniform ranges for HP-type rates and discrete cardinality.
_CONTINUOUS_RATE = (1.0, 20.0)
_ORDINAL_RATE = (0.05, 10.0)
_NOMINAL_RATE = (0.05, 4.0)
_CARDINALITY_RATE = (0.3, 8.0)
_MAX_LEVELS = 24


class _HPTypes(NamedTuple):
    """Types and discrete levels for each hyperparameter."""

    kind: jax.Array  # (config_dim,) int32 in {CONTINUOUS, ORDINAL, NOMINAL}
    n_levels: jax.Array  # (config_dim,) int32; meaningless where kind == CONTINUOUS
    level_values: jax.Array  # (config_dim, max_levels) the [0,1] value each level encodes to


class Task(NamedTuple):
    """Configuration table and observations for one task."""

    config_hps: jax.Array  # (seq_len, config_dim) raw HPs in [0,1]; unused columns zeroed
    point_config_idx: jax.Array  # (seq_len,) configuration row for each observation
    fidelities: jax.Array  # (seq_len,) fidelity of each token, in (0, 1]
    y: jax.Array  # (seq_len,) observed curve value, in [0, 1]


class _TaskParams(NamedTuple):
    """Sampled configurations and curve parameters for one task."""

    configs: jax.Array  # (n_configs, config_dim) raw HPs; inactive columns pinned to 0.5
    active: jax.Array  # (config_dim,) which columns are real HPs
    y0: jax.Array  # () dataset floor
    curves: CurveParams  # curve parameters and noise scales


def _log_uniform(key, span):
    lo, hi = span
    return jnp.exp(jax.random.uniform(key, minval=jnp.log(lo), maxval=jnp.log(hi)))


def truncated_poisson(key, lam, max_part, shape=()):
    """Sample Poisson(lam) conditioned on values from zero through max_part.

    max_part and shape are fixed during tracing; lam may vary by task.
    """
    support = jnp.arange(max_part + 1, dtype=jnp.float32)
    logits = support * jnp.log(lam) - jax.scipy.special.gammaln(support + 1.0)
    return jax.random.categorical(key, logits, shape=shape)


def _sample_observation_noise(key, config: ObservationConfig) -> ObservationNoise:
    """Sample the noise and quantization parameters enabled for this task."""
    k_levels, k_df, k_scale = jax.random.split(key, 3)
    return ObservationNoise(
        jnp.round(_log_uniform(k_levels, config.quantize_levels))
        if config.quantize
        else jnp.array(1.0),
        _log_uniform(k_df, config.tail_df) if config.noise_tails else jnp.array(10.0),
        _log_uniform(k_scale, config.tail_scale) if config.noise_tails else jnp.array(1.0),
    )


def _sample_hp_types(key, config_dim: int) -> _HPTypes:
    """Draw slot types, discrete cardinalities and encoded level values.

    Types are categorical under normalized sampled rates. Cardinalities use a
    renormalized truncated Poisson distribution. Ordinal levels are ordered;
    nominal encoded values are independent. Inactive level columns are unused.
    """
    k_rates, k_kind, k_lam_k, k_lev, k_val = jax.random.split(key, 5)
    kr = jax.random.split(k_rates, 3)
    rates = jnp.stack(
        [
            _log_uniform(kr[0], _CONTINUOUS_RATE),
            _log_uniform(kr[1], _ORDINAL_RATE),
            _log_uniform(kr[2], _NOMINAL_RATE),
        ]
    )
    kind = jax.random.categorical(k_kind, jnp.log(rates), shape=(config_dim,))

    lam_k = _log_uniform(k_lam_k, _CARDINALITY_RATE)
    n_levels = 2 + truncated_poisson(k_lev, lam_k, _MAX_LEVELS - 2, shape=(config_dim,))

    idx = jnp.arange(_MAX_LEVELS)[None, :]
    ordered = idx / jnp.maximum(n_levels[:, None] - 1, 1)
    shuffled = jax.random.uniform(k_val, (config_dim, _MAX_LEVELS))
    level_values = jnp.where((kind == NOMINAL)[:, None], shuffled, ordered)
    return _HPTypes(kind=kind, n_levels=n_levels.astype(jnp.int32), level_values=level_values)


def _sample_configs(key, types: _HPTypes, n_configs: int, config_dim: int) -> jax.Array:
    """Sample configurations with the assigned continuous and discrete HP types."""
    u = jax.random.uniform(key, (n_configs, config_dim))
    # which level each draw falls in; clipped so u == 1.0 does not index past the last level
    level = jnp.minimum(
        (u * types.n_levels[None, :]).astype(jnp.int32), types.n_levels[None, :] - 1
    )
    discrete = jnp.take_along_axis(types.level_values.T, level, axis=0)
    return jnp.where(types.kind[None, :] == CONTINUOUS, u, discrete)


def _sample_task_params(
    key,
    *,
    config_dim,
    num_params,
    n_configs,
    table,
    layers,
    hidden,
    observation_config=DEFAULT_OBSERVATIONS,
    typed_hyperparameters=False,
) -> _TaskParams:
    """Sample configurations and their curve parameters for one task."""
    k_bnn, k_arch, k_y, k_cfg, k_fwd, k_div_frac = jax.random.split(key, 6)
    depth = jax.random.randint(jax.random.fold_in(k_arch, 0), (), layers[0], layers[1] + 1)
    width = jax.random.randint(jax.random.fold_in(k_arch, 1), (), hidden[0], hidden[1] + 1)
    bnn = sample_bnn(
        k_bnn,
        num_inputs=config_dim,
        num_outputs=num_bnn_outputs(observation_config),
        num_hidden=width,
        num_layers=depth,
        max_hidden=hidden[1],
        max_layers=layers[1],
    )

    # A quarter of tasks have an asymptote ceiling below one.
    u = jax.random.uniform(k_y, (3,))
    y0 = jnp.minimum(u[0], u[1])
    y_max = jnp.where(u[2] < 0.25, jnp.maximum(u[0], u[1]), 1.0)

    active = jnp.arange(config_dim) < num_params

    if not typed_hyperparameters:
        configs = jax.random.uniform(k_cfg, (n_configs, config_dim))
    else:
        k_types, k_vals = jax.random.split(k_cfg)
        types = _sample_hp_types(k_types, config_dim)
        configs = _sample_configs(k_vals, types, n_configs, config_dim)
    # Inactive columns become zero when standardized for the BNN.
    configs = jnp.where(active, configs, 0.5)
    bnn_out = bnn_forward(bnn, standardize_uniform(configs), k_fwd)
    divergence_fraction = None
    if observation_config.divergence:
        divergence_fraction = _log_uniform(k_div_frac, observation_config.div_prob)
    return _TaskParams(
        configs,
        active,
        y0,
        extract_curve_params(bnn_out, y0, y_max, table, observation_config, divergence_fraction),
    )


def sample_task(
    key,
    *,
    seq_len,
    config_dim,
    num_params,
    n_steps,
    single_eval_pos,
    table,
    layers,
    hidden,
    n_steps_max,
    observation_config=DEFAULT_OBSERVATIONS,
    recall_probability=0.0,
    typed_hyperparameters=False,
):
    """Sample one task with a fixed padded shape.

    seq_len, config_dim, n_steps_max, layers and hidden are static. num_params,
    n_steps and single_eval_pos can vary within a compiled sampler.
    Typed sampling mixes continuous, ordinal and nominal hyperparameters.
    """
    # Configuration and schedule draws are independent of observation options.
    k_params, k_order, k_fid, k_noise, k_observation_noise = jax.random.split(key, 5)
    task_params = _sample_task_params(
        k_params,
        config_dim=config_dim,
        num_params=num_params,
        n_configs=seq_len,
        table=table,
        layers=layers,
        hidden=hidden,
        observation_config=observation_config,
        typed_hyperparameters=typed_hyperparameters,
    )

    noise = _sample_observation_noise(k_observation_noise, observation_config)
    ordering = sample_config_indices(k_order, seq_len, n_steps)
    fidelities = sample_fidelities(
        k_fid, ordering, single_eval_pos, n_steps, n_steps_max, recall_probability
    )

    curves = task_params.curves

    def observe_token(pos):
        config_idx = ordering[pos]
        fidelity = fidelities[pos]
        y_clean = learning_curve(
            fidelity,
            task_params.y0,
            curves.y_inf[config_idx],
            curves.precision[:, config_idx],
            curves.sat_fidelity[:, config_idx],
            curves.alpha[:, config_idx],
            curves.post_sat_rate[:, config_idx],
            curves.weights[:, config_idx],
        )
        noise_scale = curves.sigma[config_idx]
        if observation_config.divergence:
            y_clean, noise_keep = apply_alive_window(
                y_clean, fidelity, task_params.y0,
                curves.div_alive[config_idx], curves.div_die[config_idx],
                curves.div_level[config_idx],
            )
            noise_scale *= noise_keep

        # Repeated observations of a configuration at the same step share their noise.
        step_idx = jnp.round(fidelity * n_steps).astype(jnp.int32) - 1
        noise_key = jax.random.fold_in(k_noise, config_idx * n_steps_max + step_idx)
        return observe(y_clean, noise_key, noise_scale, observation_config, noise)

    y = jax.vmap(observe_token)(jnp.arange(seq_len))
    config_hps = jnp.where(task_params.active, task_params.configs, 0.0)
    return Task(config_hps, ordering, fidelities, y)
