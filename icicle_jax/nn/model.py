"""Predict learning curves from configuration vectors and observed values.

Encode each configuration once and gather its embedding for each point.
Add value and fidelity encodings, then predict a bar distribution per point.
"""

import math

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike

from icicle_jax.nn.attention import incontext_key_mask
from icicle_jax.nn.config_encoder import ConfigEncoder, ConfigTokenizer
from icicle_jax.nn.encoder_layer import EncoderLayer
from icicle_jax.nn.ragged import pack_referenced_configs
from icicle_jax.nn.value_embedding import sincos


def _run_icl_layer(layer, x, key_mask):
    """Apply one ICL layer (standalone so ``nnx.remat`` can wrap it)."""
    return layer(x, key_mask=key_mask)


class Surrogate(nnx.Module):
    """Configuration encoder, in-context transformer and bar-distribution head."""

    def __init__(
        self,
        emsize: int,
        num_heads: int,
        hidden_dim: int,
        n_feat_layers: int,
        n_icl_layers: int,
        num_bars: int,
        *,
        qk_norm: bool = False,
        icl_backend: str = "auto",
        remat: bool = False,
        config_remat: bool | None = None,
        feat_chunk: int = 0,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        # Value and fidelity encodings each need an even-width half of the embedding.
        if emsize % 4:
            raise ValueError(f"emsize {emsize} must be divisible by 4 (split y/fidelity codes)")
        self.emsize = emsize
        self.remat = remat
        config_remat = remat if config_remat is None else config_remat
        kw = {"dtype": dtype, "param_dtype": param_dtype, "rngs": rngs}
        self.tokenizer = ConfigTokenizer(emsize)
        # Feature attention runs independently for each configuration. It shares
        # the selected attention backend; PMA cross-attention uses XLA.
        self.config_encoder = ConfigEncoder(
            emsize,
            num_heads,
            hidden_dim,
            n_feat_layers,
            qk_norm=qk_norm,
            remat=config_remat,
            feat_chunk=feat_chunk,
            backend=icl_backend,
            **kw,
        )
        self.icl_layers = nnx.List(
            [
                EncoderLayer(
                    emsize,
                    num_heads,
                    hidden_dim,
                    qk_norm=qk_norm,
                    backend=icl_backend,
                    **kw,
                )
                for _ in range(n_icl_layers)
            ]
        )
        # Normalize the residual stream before the bar-distribution head.
        self.final_norm = nnx.RMSNorm(emsize, **kw)
        self.decoder_linear1 = nnx.Linear(emsize, hidden_dim, **kw)
        self.decoder_linear2 = nnx.Linear(hidden_dim, num_bars, **kw)

    def __call__(
        self,
        config_hps: jax.Array,
        num_params,
        point_config_idx: jax.Array,
        fidelities: jax.Array,
        y: jax.Array,
        single_eval_pos: int | jax.Array,
        *,
        feature_key: jax.Array,
    ) -> jax.Array:
        cells, cell_mask = self.tokenizer(config_hps, num_params, feature_key=feature_key)
        config_embeddings = self.config_encoder(cells, cell_mask)
        point_embeddings = jnp.take_along_axis(
            config_embeddings, point_config_idx[..., None], axis=1
        )
        return self._decode(point_embeddings, fidelities, y, single_eval_pos)

    def forward_ragged(
        self,
        config_hps: jax.Array,
        num_params,
        point_config_idx: jax.Array,
        fidelities: jax.Array,
        y: jax.Array,
        single_eval_pos: int | jax.Array,
        n_pad: int,
        *,
        feature_key: jax.Array,
        groups: int = 1,
    ) -> jax.Array:
        """Encode only configurations referenced by the point indices.

        n_pad must fit every group's distinct configurations. It sets a static buffer
        size, so changing it compiles another program. groups divides the batch into
        equal slices whose packing and gathers stay within their device shard.
        """
        batch_size = config_hps.shape[0]
        tasks_per_group = batch_size // groups
        grouped_hps = config_hps.reshape(groups, tasks_per_group, *config_hps.shape[1:])
        grouped_indices = point_config_idx.reshape(groups, tasks_per_group, -1)
        flat_hps, task_indices, point_slots, valid = jax.vmap(
            lambda configs, indices: pack_referenced_configs(configs, indices, n_pad)
        )(grouped_hps, grouped_indices)

        # Convert group-local task indices to batch indices for the feature identities.
        task_offsets = jnp.arange(groups, dtype=task_indices.dtype) * tasks_per_group
        task_indices = task_indices + task_offsets[:, None]
        n_hyperparameters = flat_hps.shape[-1]
        cells, mask = self.tokenizer.tokenize_flat(
            flat_hps.reshape(groups * n_pad, n_hyperparameters),
            task_indices.reshape(-1),
            num_params,
            batch_size,
            feature_key=feature_key,
        )
        config_embeddings = self.config_encoder(
            cells.reshape(groups, n_pad, n_hyperparameters, self.emsize),
            mask.reshape(groups, n_pad, n_hyperparameters),
            groups=groups,
            active_counts=valid.sum(axis=1),
        )  # (G, n_pad, emsize)
        n_points = point_config_idx.shape[1]
        point_embeddings = jnp.take_along_axis(
            config_embeddings,
            point_slots.reshape(groups, tasks_per_group * n_points)[..., None],
            axis=1,
        ).reshape(batch_size, n_points, self.emsize)
        return self._decode(point_embeddings, fidelities, y, single_eval_pos)

    def _decode(self, point_embeddings, fidelities, y, single_eval_pos):
        """Add observed values and fidelities, run attention, and predict bucket logits."""
        run_layer = nnx.remat(_run_icl_layer) if self.remat else _run_icl_layer
        n_points = fidelities.shape[1]
        is_context = jnp.arange(n_points) < single_eval_pos
        keep_y = is_context[None, :] & jnp.isfinite(y)

        # Value and fidelity codes occupy separate halves, scaled to the full-width norm.
        half = self.emsize // 2
        values = jnp.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        value_features = sincos(values, half) * math.sqrt(2.0)
        fidelity_features = sincos(fidelities, half) * math.sqrt(2.0)
        value_code = jnp.concatenate([value_features, jnp.zeros_like(value_features)], axis=-1)
        fidelity_code = jnp.concatenate(
            [jnp.zeros_like(fidelity_features), fidelity_features], axis=-1
        )
        point_embeddings = point_embeddings + (value_code * keep_y[..., None]).astype(
            point_embeddings.dtype
        )
        point_embeddings = point_embeddings + fidelity_code.astype(point_embeddings.dtype)

        # Queries attend only to context points.
        key_mask = incontext_key_mask(single_eval_pos, n_points)
        for layer in self.icl_layers:
            point_embeddings = run_layer(layer, point_embeddings, key_mask)

        point_embeddings = self.final_norm(point_embeddings)
        hidden = jax.nn.gelu(self.decoder_linear1(point_embeddings), approximate=False)
        return self.decoder_linear2(hidden)  # (B, P, num_bars)
