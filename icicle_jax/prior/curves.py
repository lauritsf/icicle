"""Sample curve parameters and evaluate their mixture of four learning curves."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import ndtri


def _empirical_quantile(values, table):
    """Map values to ranks, keeping the tail quantiles finite."""
    size = table.shape[0]
    rank = jnp.searchsorted(table, values, side="left")
    return jnp.clip(rank / size, 0.5 / size, 1.0 - 0.5 / size)


def _lognormal_quantile(u, mean, std):
    return jnp.exp(mean + std * ndtri(u))


class CurveParams(NamedTuple):
    """Curve parameters and observation-noise scales for each configuration."""

    y_inf: jax.Array  # asymptote, (C,)
    weights: jax.Array  # mixture weights, (4, C), sum to 1 over the 4 bases
    alpha: jax.Array  # shape exponents, (4, C)
    sat_fidelity: jax.Array  # saturation fidelities, (4, C)
    precision: jax.Array  # gap-reduction factors, (4, C)
    post_sat_rate: jax.Array  # post-saturation slopes, (4, C)
    sigma: jax.Array  # observation-noise std, (C,)
    # Active interval; full range when divergence is disabled.
    div_alive: jax.Array  # window start, (C,); <= 0 = alive from the start
    div_die: jax.Array  # window end, (C,); > 1 = never dies
    div_level: jax.Array  # clamp value as a fraction of the task floor y0, (C,)


NUM_BASES = 4  # the four basis curves (power, exponential, logarithmic, Hill)

# Number of BNN outputs used by each parameter.
_LAYOUT = (
    ("y_inf", 1),
    ("weights", NUM_BASES),
    ("alpha", NUM_BASES),
    ("sat_max", 1),  # combined with sat_rel to set saturation fidelities
    ("sat_rel", NUM_BASES),
    ("precision", NUM_BASES),
    ("post_sat_rate", NUM_BASES),
    ("sigma", 1),
)
# Additional outputs for configurations that crash or recover.
_LAYOUT_DIVERGENCE = (
    ("div_gate", 1),  # thresholded against the per-task fraction: which configs diverge
    ("div_alive", 1),  # initial inactivity and recovery time
    ("div_die", 1),  # failure time
    ("div_level", 1),  # clamp value (with the div_zero atom at exactly 0)
)


def _slot_offsets(layout):
    """Map each name to its ``(start, count)`` in the output vector; also return the total width."""
    offsets, start = {}, 0
    for name, count in layout:
        offsets[name] = (start, count)
        start += count
    return offsets, start


_BASE_SLOTS, NUM_BNN_OUTPUTS = _slot_offsets(_LAYOUT)
_DIVERGENCE_SLOTS, _DIVERGENCE_OUTPUTS = _slot_offsets(_LAYOUT + _LAYOUT_DIVERGENCE)


def num_bnn_outputs(observation_config) -> int:
    """Number of network outputs needed for the selected observation settings."""
    if observation_config is not None and observation_config.divergence:
        return _DIVERGENCE_OUTPUTS
    return NUM_BNN_OUTPUTS


def extract_curve_params(
    bnn_output, y0, y_max, output_sorted, observation_config=None, divergence_fraction=None
) -> CurveParams:
    """Map network outputs to one set of curve parameters per configuration.

    output_sorted combines calibration outputs from sampled networks.
    y0 and y_max bound the asymptote. When failures are enabled,
    divergence_fraction determines how many configurations participate.
    """
    divergence = observation_config is not None and observation_config.divergence
    slots = _DIVERGENCE_SLOTS if divergence else _BASE_SLOTS
    uniforms = _empirical_quantile(bnn_output, output_sorted).T  # (num_outputs, C)

    def slot(name):
        start, count = slots[name]
        return uniforms[start : start + count]

    y_inf = y0 + (y_max - y0) * slot("y_inf")[0]

    weights = -jnp.log1p(-slot("weights"))  # Gamma(1) draws
    weights = weights / weights.sum(0, keepdims=True)  # Dirichlet mixture weights

    a = slot("alpha")
    alpha = jnp.stack(
        [
            _lognormal_quantile(a[0], 1.0, 1.0),
            _lognormal_quantile(a[1], 0.0, 1.0),
            1.0 + _lognormal_quantile(a[2], -4.0, 1.0),
            _lognormal_quantile(a[3], 0.5, 0.5),
        ]
    )

    sat_max = 10.0 ** ndtri(slot("sat_max")[0])
    sat_rel = -jnp.log1p(-slot("sat_rel"))
    sat_fidelity = sat_max * sat_rel / sat_rel.max(0, keepdims=True)

    precision = 1.0 / 10.0 ** (-3.0 + 3.0 * slot("precision"))
    post_sat_rate = 1.0 + jnp.log1p(-slot("post_sat_rate"))
    sigma = _lognormal_quantile(slot("sigma")[0], -5.0, 1.0)

    if divergence:
        is_div = slot("div_gate")[0] > 1.0 - divergence_fraction
        u_alive, u_die = slot("div_alive")[0], slot("div_die")[0]
        # Half the configurations start active and may fail at t_die. The others
        # become active at t_alive and may fail later. Bounds outside [0, 1]
        # allow configurations that stay at the floor throughout the curve.
        alive_early = u_alive < 0.5
        t_alive_late = (u_alive - 0.5) * 2.0 * observation_config.div_alive_max
        t_die_early = -observation_config.div_start_frac + (1.0 + observation_config.div_start_frac) * u_die
        t_die_late = jnp.where(
            u_die < 0.5, 2.0, t_alive_late + (1.0 - t_alive_late) * (u_die - 0.5) * 2.0
        )
        t_alive = jnp.where(alive_early, 0.0, t_alive_late)
        t_die = jnp.where(alive_early, t_die_early, t_die_late)
        # clamp value: atom at exactly 0 (TaskSet's loss clamp), else a fraction of the floor
        u_level = slot("div_level")[0]
        z = observation_config.div_zero
        level = jnp.where(u_level < z, 0.0, (u_level - z) / (1.0 - z))
        div_alive = jnp.where(is_div, t_alive, -1.0)
        div_die = jnp.where(is_div, t_die, 2.0)
        div_level = jnp.where(is_div, level, 0.0)
    else:
        div_alive = jnp.full_like(sigma, -1.0)
        div_die = jnp.full_like(sigma, 2.0)
        div_level = jnp.zeros_like(sigma)
    return CurveParams(
        y_inf,
        weights,
        alpha,
        sat_fidelity,
        precision,
        post_sat_rate,
        sigma,
        div_alive,
        div_die,
        div_level,
    )


_EPS = 1e-9


def _saturation_break(x: jax.Array, sat_fidelity: jax.Array, post_sat_rate: jax.Array) -> jax.Array:
    """Rescale progress after saturation; negative rates reverse its direction."""
    return jnp.where(x < sat_fidelity, x, post_sat_rate * (x - sat_fidelity) + sat_fidelity)


def learning_curve(
    x: jax.Array,  # fidelity
    y0: jax.Array,  # initial value
    y_inf: jax.Array,  # asymptote
    precision: jax.Array,  # per-basis gap reduction at saturation
    sat_fidelity: jax.Array,  # saturation fidelity
    alpha: jax.Array,  # shape exponent
    post_sat_rate: jax.Array,  # progress rate after saturation
    weights: jax.Array,  # mixture weights
) -> jax.Array:
    """Evaluate the mixture of power, exponential, logarithmic and Hill curves.

    This follows ifBO's comb function. Per-basis parameters have a leading
    axis of length four. Post-saturation rates can reverse curve progress.
    """
    near_zero = jnp.array([_EPS, 2 * _EPS])  # two points near 0+, for the initial-slope estimate

    # Each basis gives the fraction of the gap to y_inf that remains.
    def power(t, prec, a, sat):
        return ((prec ** (1 / a) - 1) / sat * t + 1) ** (-a)

    def exponential(t, prec, a, sat):
        return prec ** (-((t / sat) ** a))

    def logarithmic(t, prec, a, sat):
        return jnp.log(a) / jnp.log((a**prec - a) * t / sat + a)

    def hill(t, prec, a, sat):
        return 1.0 / ((t / sat) ** a * (prec - 1) + 1)

    def basis(sat, post_sat, prec, a, remaining_gap):
        # Extend to negative adjusted fidelities using the slope near zero.
        broken_x = _saturation_break(x, sat, post_sat)
        near = remaining_gap(near_zero, prec, a, sat)
        slope_at_zero = -(y_inf - y0) * (near[1] - near[0]) / _EPS
        on_curve = y_inf - (y_inf - y0) * remaining_gap(broken_x, prec, a, sat)
        below_zero = y0 * jnp.exp(broken_x * (slope_at_zero + _EPS) / y0)
        return jnp.where(broken_x > 0, on_curve, below_zero)

    bases = (power, exponential, logarithmic, hill)
    return sum(
        weights[i] * basis(sat_fidelity[i], post_sat_rate[i], precision[i], alpha[i], fn)
        for i, fn in enumerate(bases)
    )
