"""Sample random configuration-to-curve networks and calibrate their outputs."""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from icicle_jax.prior.curves import NUM_BNN_OUTPUTS

_WEIGHT_STD = (0.089, 0.193)
_PREACT_NOISE = (0.0003, 0.0014)
_OUTPUT_NOISE = (0.0004, 0.0013)
_DROP_PROBABILITY = 0.145


class BNN(NamedTuple):
    """A sampled network with fixed-size weight arrays."""

    w_in: jax.Array  # (max_hidden, num_inputs)
    b_in: jax.Array  # (max_hidden,)
    w_hidden: jax.Array  # (max_layers - 2, max_hidden, max_hidden), stacked for the scan
    b_hidden: jax.Array  # (max_layers - 2, max_hidden)
    w_out: jax.Array  # (num_outputs, max_hidden)
    b_out: jax.Array  # (num_outputs,)
    preact_noise: jax.Array  # sampled pre-activation noise std
    output_noise: jax.Array  # sampled output noise std
    num_layers: jax.Array  # sampled depth (traced); the scan runs num_layers - 2 hidden steps


def _sample_uniform(key, span):
    low, high = span
    return jax.random.uniform(key, minval=low, maxval=high)


def _sparsify(key, w_hidden):
    """Drop hidden weights and rescale to preserve their variance."""
    keep = jax.random.bernoulli(key, 1.0 - _DROP_PROBABILITY, w_hidden.shape).astype(w_hidden.dtype)
    return w_hidden / (1.0 - _DROP_PROBABILITY) ** 0.5 * keep


def _mask_width(w_in, w_hidden, w_out, num_hidden):
    """Mask connections to unused hidden units."""
    active = jnp.arange(w_in.shape[0]) < num_hidden
    return (
        w_in * active[:, None],
        w_hidden * active[None, :, None] * active[None, None, :],
        w_out * active[None, :],
    )


def sample_bnn(
    key, *, num_inputs, num_outputs, num_hidden, num_layers, max_hidden, max_layers
):
    """Sample weights and noise scales, masking to the chosen width and depth."""
    k_scale, k_weight, k_sparse = jax.random.split(key, 3)

    k_wstd, k_pre, k_out = jax.random.split(k_scale, 3)
    weight_std = _sample_uniform(k_wstd, _WEIGHT_STD)
    preact_noise = _sample_uniform(k_pre, _PREACT_NOISE)
    output_noise = _sample_uniform(k_out, _OUTPUT_NOISE)

    def layer(k, shape):
        kw, kb = jax.random.split(k)
        w = weight_std * jax.random.normal(kw, shape)
        b = weight_std * jax.random.normal(kb, shape[:1])
        return w, b

    k_in, k_hidden, k_outw = jax.random.split(k_weight, 3)
    w_in, b_in = layer(k_in, (max_hidden, num_inputs))
    w_hidden, b_hidden = jax.vmap(lambda k: layer(k, (max_hidden, max_hidden)))(
        jax.random.split(k_hidden, max_layers - 2)
    )
    w_out, b_out = layer(k_outw, (num_outputs, max_hidden))

    w_hidden = _sparsify(k_sparse, w_hidden)
    w_in, w_hidden, w_out = _mask_width(w_in, w_hidden, w_out, num_hidden)
    return BNN(w_in, b_in, w_hidden, b_hidden, w_out, b_out, preact_noise, output_noise, num_layers)


def bnn_forward(bnn, x, key):
    """Evaluate the sampled network with pre-activation and output noise.

    The scan applies num_layers - 2 hidden layers; remaining steps pass the
    state through unchanged.
    """
    k_hidden, k_out = jax.random.split(key)
    h = x @ bnn.w_in.T + bnn.b_in
    h = jnp.tanh(h + bnn.preact_noise * jax.random.normal(k_hidden, h.shape))

    hidden_keys = jax.random.split(k_hidden, bnn.w_hidden.shape[0])
    steps = jnp.arange(bnn.w_hidden.shape[0])

    def run_layer(h, layer):
        w, b, k, step = layer
        h_next = jnp.tanh(h @ w.T + b + bnn.preact_noise * jax.random.normal(k, h.shape))
        return jnp.where(step < bnn.num_layers - 2, h_next, h), None

    h, _ = jax.lax.scan(run_layer, h, (bnn.w_hidden, bnn.b_hidden, hidden_keys, steps))
    out = h @ bnn.w_out.T + bnn.b_out
    return out + bnn.output_noise * jax.random.normal(k_out, out.shape)


def standardize_uniform(configs: jax.Array) -> jax.Array:
    """Standardize uniform [0, 1] inputs to zero mean and unit variance."""
    return (configs - 0.5) * jnp.sqrt(12.0)


def build_output_sorted(
    key, *, config_dim, layers, hidden, num_networks=1000, configs_per_network=64
) -> jax.Array:
    """Build a sorted calibration table from sampled networks and configurations.

    Sample num_networks networks with depths and widths in the inclusive
    layers/hidden ranges. Evaluate configs_per_network configurations using all hyperparameters
    for each network, then sort the combined outputs.
    """
    max_layers, max_hidden = layers[1], hidden[1]
    k_net, k_depth, k_width, k_cfg, k_fwd = jax.random.split(key, 5)
    net_keys = jax.random.split(k_net, num_networks)
    depths = jax.random.randint(k_depth, (num_networks,), layers[0], layers[1] + 1)
    widths = jax.random.randint(k_width, (num_networks,), hidden[0], hidden[1] + 1)
    cfg_keys = jax.random.split(k_cfg, num_networks)
    fwd_keys = jax.random.split(k_fwd, num_networks)

    def outputs_of(net_key, depth, width, cfg_key, fwd_key):
        bnn = sample_bnn(
            net_key,
            num_inputs=config_dim,
            num_outputs=NUM_BNN_OUTPUTS,
            num_hidden=width,
            num_layers=depth,
            max_hidden=max_hidden,
            max_layers=max_layers,
        )
        configs = jax.random.uniform(cfg_key, (configs_per_network, config_dim))
        return bnn_forward(bnn, standardize_uniform(configs), fwd_key)

    outputs = jax.vmap(outputs_of)(net_keys, depths, widths, cfg_keys, fwd_keys)
    return jnp.sort(outputs.reshape(-1))
