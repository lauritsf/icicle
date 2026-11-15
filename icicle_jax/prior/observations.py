"""Choose observation points and apply noise, quantization and failure windows."""

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class ObservationConfig:
    """Settings for quantization, heavy-tailed noise and curve failures.

    Tuple ranges are sampled log-uniformly once per task.
    """

    quantize: bool = False
    noise_tails: bool = False
    quantize_levels: tuple[float, float] = (1e2, 3e4)
    tail_df: tuple[float, float] = (2.1, 8.0)  # Student-t degrees of freedom
    tail_scale: tuple[float, float] = (0.25, 4.0)  # noise multiplier
    divergence: bool = False
    div_prob: tuple[float, float] = (0.02, 0.6)  # fraction of configurations affected
    div_start_frac: float = 0.25  # allows failure before the first observation
    div_alive_max: float = 1.2  # recovery times above one leave the curve inactive
    div_zero: float = 0.5  # probability of clamping to zero instead of a fraction of y0


DEFAULT_OBSERVATIONS = ObservationConfig()

FLAG_NAMES = (
    "quantize",
    "noise_tails",
    "divergence",
)


def parse_observations(spec: str) -> ObservationConfig:
    """Read the enabled observation settings."""
    if spec == "none":
        return ObservationConfig()
    if spec == "all":
        return ObservationConfig(**dict.fromkeys(FLAG_NAMES, True))
    names = sorted(set(spec.split(",")))
    unknown = [n for n in names if n not in FLAG_NAMES]
    if unknown:
        raise ValueError(f"unknown observation option(s) {unknown}; valid: {list(FLAG_NAMES)}")
    return ObservationConfig(**dict.fromkeys(names, True))


class ObservationNoise(NamedTuple):
    """Sampled quantization levels and noise parameters for one task."""

    n_levels: jax.Array  # () quantization grid size N
    df: jax.Array  # () Student-t degrees of freedom
    scale: jax.Array  # () per-task sigma multiplier


def apply_alive_window(y_clean, fidelity, y0, t_alive, t_die, v_frac):
    """Clamp values outside [t_alive, t_die) and disable noise there.

    A positive start allows delayed recovery; an end below 1 models a crash.
    A full interval leaves the curve unchanged. Returns (y, noise_keep), with
    elementwise broadcasting.
    """
    alive = (fidelity >= t_alive) & (fidelity < t_die)
    y = jnp.where(alive, y_clean, v_frac * y0)
    return y, jnp.where(alive, 1.0, 0.0)


def observe(y_clean, key, sigma, config: ObservationConfig, noise: ObservationNoise):
    """Add noise, clip to [0, 1], and optionally round to the quantization grid."""
    shape = jnp.broadcast_shapes(jnp.shape(y_clean), jnp.shape(sigma))
    sigma_eff = sigma
    if config.noise_tails:
        sigma_eff = sigma_eff * noise.scale
        # Standardize Student-t noise to unit variance (df > 2).
        eps = jax.random.t(key, noise.df, shape) / jnp.sqrt(noise.df / (noise.df - 2.0))
    else:
        eps = jax.random.normal(key, shape)
    y = jnp.clip(y_clean + sigma_eff * eps, 0.0, 1.0)
    if config.quantize:
        y = jnp.round(y * noise.n_levels) / noise.n_levels
    return y


def sample_config_indices(key, seq_len, n_steps):
    """Assign points to configurations, with at most n_steps points per configuration."""
    k_conc, k_weight, k_draw = jax.random.split(key, 3)

    # Small concentrations concentrate observations on a few configurations.
    concentration = 10.0 ** jax.random.uniform(k_conc, minval=-4.0, maxval=-1.0)
    weights = jax.random.gamma(k_weight, concentration, (seq_len,)) + 1e-9
    draw_keys = jax.random.split(k_draw, seq_len)

    def draw(remaining, draw_key):
        # Selection probability is proportional to weight times remaining steps.
        mass = weights * remaining
        logits = jnp.where(remaining > 0, jnp.log(mass), -jnp.inf)
        config_idx = jax.random.categorical(draw_key, logits)
        return remaining.at[config_idx].add(-1), config_idx

    _, ordering = jax.lax.scan(draw, jnp.full(seq_len, n_steps, jnp.int32), draw_keys)
    return ordering


def sample_fidelities(
    key, ordering, single_eval_pos, n_steps, n_steps_max, recall_probability=0.0
):
    """Assign context and query fidelities to each point.

    Each observed step becomes eligible with recall_probability. Queries sample
    from the combined pool of eligible observed steps and unseen steps.
    n_steps_max fixes the buffer shape; n_steps can vary between sampled tasks.
    """
    seq_len = ordering.shape[0]
    context = (jnp.arange(seq_len) < single_eval_pos).astype(jnp.int32)
    n_observed = jnp.zeros(seq_len, jnp.int32).at[ordering].add(context)
    step = jnp.arange(n_steps_max)
    step_fidelity = (step + 1).astype(jnp.float32) / n_steps

    def curve_schedule(curve_key, observed):
        eligible = (step >= observed) & (step < n_steps)
        if recall_probability > 0.0:
            k_rank, k_recall = jax.random.split(curve_key)
            recall = jax.random.uniform(k_recall, (n_steps_max,)) < recall_probability
            eligible |= recall & (step < observed) & (step < n_steps)
            curve_key = k_rank

        # Sort eligible steps randomly, with padding and excluded steps last.
        rank = jnp.where(eligible, jax.random.uniform(curve_key, (n_steps_max,)), 2.0)
        query_fidelity = step_fidelity[jnp.argsort(rank)]
        query_idx = jnp.maximum(step - observed, 0)
        return jnp.where(step < observed, step_fidelity, query_fidelity[query_idx])

    schedules = jax.vmap(curve_schedule)(jax.random.split(key, seq_len), n_observed)

    def take_next_fidelity(counters, config_idx):
        fidelity = schedules[config_idx, counters[config_idx]]
        return counters.at[config_idx].add(1), fidelity

    _, fidelities = jax.lax.scan(
        take_next_fidelity, jnp.zeros(seq_len, jnp.int32), ordering
    )
    return fidelities
