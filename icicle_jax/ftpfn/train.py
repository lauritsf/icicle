"""FT-PFN training wrapper and paper-recipe command.

    python -m icicle_jax.ftpfn.train --run.checkpoint-dir runs/ftpfn-paper/ckpt

The common trainer also uses TrainableFTPFN for architecture/prior ablations.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx

from icicle_jax.ftpfn.model import FTPFN


def pack_tokens(
    config_hps,
    num_params,
    point_config_idx,
    fidelities,
    sep,
    *,
    config_dim,
    max_curves,
    curve_offsets=None,
    slot_permutations=None,
):
    """Convert configuration tables to FT-PFN tokens.

    Scale active HPs by config_dim / num_params and zero the remaining columns.
    A query gets ID 0 unless its configuration has context observations.
    curve_offsets rotates the embedding assignment so every embedding row can train.
    slot_permutations optionally moves active HPs to randomly chosen input positions.
    """
    b, c, h = config_hps.shape
    if h > config_dim or c > max_curves:
        raise ValueError(
            f"FT-PFN training expects H≤{config_dim} and C≤{max_curves}; got H={h}, C={c}"
        )
    active = jnp.arange(h) < num_params
    hps = jnp.take_along_axis(config_hps, point_config_idx[..., None], axis=1)  # (B, P, H)
    hps = jnp.where(active, hps * (config_dim / num_params), 0.0)
    hps = jnp.pad(hps, ((0, 0), (0, 0), (0, config_dim - h)))  # (B, P, H_cfg)
    if slot_permutations is not None:
        if slot_permutations.shape != (b, config_dim):
            raise ValueError(
                f"slot_permutations must have shape {(b, config_dim)}, "
                f"got {slot_permutations.shape}"
            )
        inverse = jnp.argsort(slot_permutations, axis=-1)
        hps = jnp.take_along_axis(hps, inverse[:, None, :], axis=-1)
    context = jnp.arange(point_config_idx.shape[1]) < sep
    observed = (
        jnp.zeros((b, c), dtype=bool)
        .at[jnp.arange(b)[:, None], point_config_idx]
        .max(jnp.broadcast_to(context, point_config_idx.shape))
    )
    seen = jnp.take_along_axis(observed, point_config_idx, axis=1)  # (B, P)
    if curve_offsets is None:
        curve_offsets = jnp.zeros((b,), dtype=point_config_idx.dtype)
    row_ids = (point_config_idx + curve_offsets[:, None]) % max_curves + 1
    ids = jnp.where(seen, row_ids, 0)
    return jnp.concatenate([ids[..., None], fidelities[..., None], hps], axis=-1)


def _run_layer(layer, src, sep):
    return layer(src, sep)


def presentation_randomness(feature_key, batch_size, config_dim, random_slots):
    """Draw per-task keys and, when requested, random HP-slot permutations."""
    if feature_key.ndim:
        task_keys = feature_key
    else:
        task_keys = jax.vmap(lambda i: jax.random.fold_in(feature_key, i))(jnp.arange(batch_size))
    if not random_slots:
        return task_keys, None
    slot_keys = jax.vmap(lambda key: jax.random.fold_in(key, 1))(task_keys)
    permutations = jax.vmap(lambda key: jax.random.permutation(key, config_dim))(slot_keys)
    return task_keys, permutations


def initialize_like_pytorch(model, key):
    """Use the initialization distributions in the pinned ifBO source."""
    def uniform(variable, bound):
        nonlocal key
        key, draw = jax.random.split(key)
        variable[...] = jax.random.uniform(
            draw, variable.shape, variable.dtype, minval=-bound, maxval=bound
        )

    def linear(layer):
        bound = 1 / math.sqrt(layer.kernel.shape[0])
        uniform(layer.kernel, bound)
        if layer.bias is not None:
            uniform(layer.bias, bound)

    for layer in (model.epoch_enc, model.config_enc, model.y_enc,
                  model.decoder_linear1, model.decoder_linear2):
        linear(layer)
    key, draw = jax.random.split(key)
    embedding = model.idcurve_enc.embedding
    embedding[...] = jax.random.normal(draw, embedding.shape, embedding.dtype)
    for layer in model.layers:
        # PyTorch initializes the combined (3E, E) QKV matrix with Xavier uniform.
        embed = layer.linear1.kernel.shape[0]
        for projection in (layer.attn.query, layer.attn.key, layer.attn.value):
            uniform(projection.kernel, math.sqrt(6 / (4 * embed)))
            projection.bias[...] = jnp.zeros_like(projection.bias[...])
        linear(layer.linear1)
        for output in (layer.attn.out, layer.linear2):
            output.kernel[...] = jnp.zeros_like(output.kernel[...])
            output.bias[...] = jnp.zeros_like(output.bias[...])
        for norm in (layer.norm1, layer.norm2):
            norm.scale[...] = jnp.ones_like(norm.scale[...])
            norm.bias[...] = jnp.zeros_like(norm.bias[...])


class TrainableFTPFN(nnx.Module):
    """Train FT-PFN on batches from the synthetic prior.

    Converts configuration tables and curve observations to FT-PFN tokens.
    The underlying FT-PFN model is available as ``model`` for inference.
    """

    def __init__(
        self,
        *,
        emsize,
        num_heads,
        hidden_dim,
        num_layers,
        num_bars,
        config_dim,
        max_curves,
        dtype,
        remat,
        random_slots,
        rngs,
        initialization="flax",
    ):
        self.model = FTPFN(
            config_dim=config_dim,
            emsize=emsize,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_bars=num_bars,
            max_curves=max_curves,
            dtype=dtype,
            param_dtype=jnp.float32,
            rngs=rngs,
        )
        if initialization == "pytorch":
            initialize_like_pytorch(self.model, rngs.params())
        elif initialization != "flax":
            raise ValueError(f"unknown FT-PFN initialization: {initialization!r}")
        self.config_dim = config_dim
        self.max_curves = max_curves
        self.remat = remat
        self.random_slots = random_slots

    def forward_tokens(self, x, y, sep):
        """Evaluate tokens that are already padded and scaled for FT-PFN."""
        src = self.model._encode(x, y, sep)
        run_layer = nnx.remat(_run_layer) if self.remat else _run_layer
        for layer in self.model.layers:
            src = run_layer(layer, src, sep)
        return self.model._decode(src)

    def __call__(
        self,
        config_hps,
        num_params,
        point_config_idx,
        fidelities,
        y,
        single_eval_pos,
        *,
        feature_key,
        slot_permutations=None,
    ):
        b = config_hps.shape[0]
        curve_keys, sampled_slot_permutations = presentation_randomness(
            feature_key, b, self.config_dim, self.random_slots
        )
        if slot_permutations is None:
            slot_permutations = sampled_slot_permutations
        elif sampled_slot_permutations is not None:
            raise ValueError("explicit slot permutations require random_slots=False")
        curve_offsets = jax.vmap(lambda key: jax.random.randint(key, (), 0, self.max_curves))(
            curve_keys
        )
        x = pack_tokens(
            config_hps,
            num_params,
            point_config_idx,
            fidelities,
            single_eval_pos,
            config_dim=self.config_dim,
            max_curves=self.max_curves,
            curve_offsets=curve_offsets,
            slot_permutations=slot_permutations,
        )
        return self.forward_tokens(x, y, single_eval_pos)


def paper_schedule(config, num_steps):
    import optax

    return optax.warmup_cosine_decay_schedule(
        0.0, config.peak_lr, config.warmup_steps, num_steps
    )


def paper_batch_parameters(key, model_cfg, data_cfg, _prior_cfg):
    k_arity, _, k_split, _ = jax.random.split(key, 4)
    num_params = jax.random.randint(k_arity, (), 1, model_cfg.config_dim + 1)
    sep = jax.random.randint(k_split, (), 0, data_cfg.seq_len)
    return num_params, None, sep  # horizons are drawn separately for each task


def paper_samplers(model_cfg, prior_cfg, data_cfg, *, table, observation_config):
    from icicle_jax.prior.task import sample_task
    from icicle_jax.train import VAL_CHUNK

    def make_sampler(batch_size):
        @jax.jit
        def sample(step_key, num_params, _n_steps, sep):
            keys = jax.random.split(jax.random.fold_in(step_key, 0), batch_size)

            def draw_task(key):
                exponent = jax.random.uniform(
                    jax.random.fold_in(key, 1), (),
                    minval=0.0, maxval=math.log10(data_cfg.n_steps_max),
                )
                n_steps = jnp.clip(
                    jnp.round(10.0**exponent).astype(jnp.int32), 1, data_cfg.n_steps_max
                )
                return sample_task(
                    key, seq_len=data_cfg.seq_len, config_dim=model_cfg.config_dim,
                    num_params=num_params, n_steps=n_steps, single_eval_pos=sep,
                    table=table, layers=prior_cfg.layers,
                    hidden=prior_cfg.hidden, n_steps_max=data_cfg.n_steps_max,
                    observation_config=observation_config,
                )

            return jax.vmap(draw_task)(keys)

        return sample

    return make_sampler(data_cfg.batch_size), make_sampler(VAL_CHUNK)


def train_paper(model_cfg, optim_cfg, prior_cfg, data_cfg, run_cfg, *, log=print):
    from icicle_jax.train import fit

    if model_cfg.architecture != "ftpfn":
        raise ValueError("this command trains the FT-PFN architecture")
    if (prior_cfg.extensions != "none" or prior_cfg.recall_probability
            or prior_cfg.hp_types_poisson or prior_cfg.arity_trunc_poisson):
        raise ValueError("the paper recipe uses the base prior without recall or typed HPs")
    if optim_cfg.decay_start is not None:
        raise ValueError("the paper recipe uses cosine decay; leave decay_start unset")
    return fit(
        model_cfg, optim_cfg, prior_cfg, data_cfg, run_cfg,
        schedule=paper_schedule(optim_cfg, run_cfg.num_steps),
        sampler_factory=paper_samplers, batch_parameters=paper_batch_parameters,
        trainer="ftpfn-paper", log=log,
    )


def main():
    import dataclasses
    import os
    from pathlib import Path

    import tyro
    from icicle_jax.train import Experiment, load_recipe

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    config = Path(__file__).resolve().parents[2] / "configs/ftpfn_paper.json"
    recipe = tyro.cli(Experiment, default=load_recipe(config))
    for field in dataclasses.fields(recipe):
        print(f"{field.name}: {getattr(recipe, field.name)}", flush=True)
    train_paper(
        recipe.model, recipe.optim, recipe.prior, recipe.data, recipe.run,
        log=lambda message: print(message, flush=True),
    )


if __name__ == "__main__":
    main()
