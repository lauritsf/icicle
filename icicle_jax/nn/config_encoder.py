"""Encode each configuration as one vector.

Bind values to feature identities, apply attention across features, then pool.
Parameters are shared across features, so their shapes do not depend on HP count.
"""

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import DTypeLike

from icicle_jax.nn.attention import Attention
from icicle_jax.nn.encoder_layer import EncoderLayer


def _run_feat_layer(layer, x, key_mask):
    """Apply one feature-attention layer (standalone so ``nnx.remat`` can wrap it)."""
    return layer(x, key_mask=key_mask)


def feature_identity(key: jax.Array, num_tasks: int, num_features: int, dim: int) -> jax.Array:
    """Draw per-task orthonormal feature vectors with shape (B, H, E).

    Requires H <= E. Gaussian QR with the Mezzadri (2007) column-sign correction
    gives a Haar-uniform frame. Each vector has norm sqrt(E/2).

    A scalar key is split into one key per task. A key array of shape (B,) is
    used directly. Training supplies fresh keys per step; evaluation supplies
    explicit per-task keys so moving a task between batch lanes does not change
    its feature frame.
    """
    if num_features > dim:
        raise ValueError(f"orthonormal identities need num_features {num_features} <= dim {dim}")

    def per_task(k):
        g = jax.random.normal(k, (dim, num_features))
        q, r = jnp.linalg.qr(g)  # (dim, num_features), orthonormal columns
        q = q * jnp.sign(jnp.diag(r))  # Mezzadri sign fix -> Haar-uniform (unbias the slot index)
        return q.T * jnp.sqrt(dim / 2)  # (num_features, dim), matched to |sincos|

    keys = key if key.ndim else jax.random.split(key, num_tasks)
    if keys.shape != (num_tasks,):
        raise ValueError(f"feature key array {keys.shape} does not match num_tasks {num_tasks}")
    return jax.vmap(per_task)(keys)


class ConfigTokenizer(nnx.Module):
    """Map (B, C, H) hyperparameters to (B, C, H, emsize) cells and a validity mask.

    Multiply each feature identity by its standardized value,
    (v - 0.5) * sqrt(12). A value of 0.5 therefore produces a zero cell.
    """

    def __init__(self, emsize: int):
        self.emsize = emsize

    def _cells(self, hps: jax.Array, identities: jax.Array) -> jax.Array:
        return ((hps - 0.5) * jnp.sqrt(12.0))[..., None] * identities

    def __call__(
        self, config_hps: jax.Array, num_params, *, feature_key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Tokenize (B, C, H) values using one scalar HP count for the batch."""
        if jnp.ndim(num_params) != 0:
            raise ValueError("num_params must be a shared scalar HP count")
        b, c, h = config_hps.shape
        u = feature_identity(feature_key, b, h, self.emsize)  # (B, H, emsize) per-task
        cells = self._cells(config_hps, u[:, None])  # broadcast the task's identities over configs
        mask = jnp.broadcast_to(jnp.arange(h) < num_params, (b, c, h))
        return cells, mask

    def tokenize_flat(
        self,
        flat_hps: jax.Array,
        flat_task: jax.Array,
        num_params,
        num_tasks: int,
        *,
        feature_key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Tokenize packed configuration rows using their task's feature identities.

        flat_task selects the task for each row. num_params is shared across the batch.
        """
        if jnp.ndim(num_params) != 0:
            raise ValueError("num_params must be a shared scalar HP count")
        n, h = flat_hps.shape
        u = feature_identity(feature_key, num_tasks, h, self.emsize)  # (num_tasks, H, emsize)
        cells = self._cells(flat_hps, u[flat_task])  # (N, H, emsize)
        mask = jnp.broadcast_to(jnp.arange(h) < num_params, (n, h))
        return cells, mask


class PMAPool(nnx.Module):
    """Pooling by multi-head attention (Set Transformer, Lee et al. 2019).

    A learned seed query attends to the valid cells to produce one vector per set.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        qk_norm: bool = False,
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.embed_dim = embed_dim
        self.seed = nnx.Param(
            jax.nn.initializers.normal(0.02)(rngs.params(), (embed_dim,), param_dtype)
        )
        self.attn = Attention(
            embed_dim, num_heads, qk_norm=qk_norm, dtype=dtype, param_dtype=param_dtype, rngs=rngs
        )

    def __call__(self, cells: jax.Array, key_mask: jax.Array) -> jax.Array:
        """``cells`` ``(N, K, e)``, ``key_mask`` ``(N, 1, 1, K)`` -> ``(N, e)``."""
        n = cells.shape[0]
        q = jnp.broadcast_to(self.seed[...], (n, 1, self.embed_dim)).astype(cells.dtype)
        return self.attn(q, cells, key_mask=key_mask)[:, 0]


class ConfigEncoder(nnx.Module):
    """Encode feature cells with attention, PMA pooling and output RMSNorm."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        n_layers: int = 1,
        *,
        qk_norm: bool = False,
        remat: bool = False,
        feat_chunk: int = 0,
        backend: str = "xla",
        dtype: DTypeLike | None = None,
        param_dtype: DTypeLike = jnp.float32,
        rngs: nnx.Rngs,
    ):
        self.remat = remat
        self.feat_chunk = feat_chunk
        self.dtype = dtype
        kw = {"dtype": dtype, "param_dtype": param_dtype, "rngs": rngs}
        # Active feature cells form a shared prefix, allowing fused attention to
        # avoid the quadratic feature-attention matrix. PMA uses cross-attention
        # and remains on the XLA path.
        self.feat_layers = nnx.List(
            [
                EncoderLayer(
                    embed_dim,
                    num_heads,
                    hidden_dim,
                    qk_norm=qk_norm,
                    backend=backend,
                    **kw,
                )
                for _ in range(n_layers)
            ]
        )
        self.pool = PMAPool(embed_dim, num_heads, qk_norm=qk_norm, **kw)
        self.norm = nnx.RMSNorm(embed_dim, **kw)

    def __call__(
        self,
        cells: jax.Array,
        mask: jax.Array,
        *,
        groups: int = 1,
        active_counts: jax.Array | None = None,
    ) -> jax.Array:
        """``cells`` ``(B, C, H, e)``, ``mask`` ``(B, C, H)`` bool -> ``e_c`` ``(B, C, e)``."""
        b, c, h, e = cells.shape
        x = cells.reshape(b * c, h, e)
        key_mask = mask.reshape(b * c, 1, 1, h)  # ignore padded HP cells
        return self._encode(
            x,
            key_mask,
            groups=groups,
            active_counts=active_counts,
        ).reshape(b, c, e)

    def _encode_rows(
        self,
        x: jax.Array,
        key_mask: jax.Array,
    ) -> jax.Array:
        """``x`` ``(N, H, e)`` config rows + ``key_mask`` ``(N, 1, 1, H)`` -> ``e_c`` ``(N, e)``."""
        run_feat = nnx.remat(_run_feat_layer) if self.remat else _run_feat_layer
        for layer in self.feat_layers:
            x = run_feat(layer, x, key_mask)
        return self.norm(self.pool(x, key_mask))  # (N, e)

    def _encode(
        self,
        x: jax.Array,
        key_mask: jax.Array,
        *,
        groups: int = 1,
        active_counts: jax.Array | None = None,
    ) -> jax.Array:
        """Encode independent configuration rows in feat_chunk-sized scan blocks.

        Increase small requested chunks when needed to limit the number of blocks.
        Pass parameters through the scan carry to respect NNX trace levels.
        Ragged callers can supply active row counts to skip complete trailing
        padding chunks; dense callers process every chunk.
        """
        n, h, e = x.shape
        if groups < 1 or n % groups:
            raise ValueError(f"config rows {n} must divide evenly into {groups} groups")
        per_group = n // groups
        cw = self._chunk_width(per_group)
        if cw >= per_group:
            return self._encode_rows(x, key_mask)
        n_chunks = -(-per_group // cw)
        pad = n_chunks * cw - per_group
        x = x.reshape(groups, per_group, h, e)
        key_mask = key_mask.reshape(groups, per_group, *key_mask.shape[1:])
        if pad:
            x = jnp.pad(x, ((0, 0), (0, pad), (0, 0), (0, 0)))
            key_mask = jnp.pad(
                key_mask,
                ((0, 0), (0, pad), (0, 0), (0, 0), (0, 0)),
                mode="edge",
            )
        graphdef, state = nnx.split(self)

        def chunks(value):
            value = value.reshape(groups, n_chunks, cw, *value.shape[2:])
            return jnp.swapaxes(value, 0, 1)

        x_chunks = chunks(x)
        mask_chunks = chunks(key_mask)

        def dense_body(state, chunk):
            xc, mc = chunk
            enc = nnx.merge(graphdef, state)
            encoded = enc._encode_rows(
                xc.reshape(groups * cw, *xc.shape[2:]),
                mc.reshape(groups * cw, *mc.shape[2:]),
            )
            return nnx.state(enc), encoded.reshape(groups, cw, -1)

        if active_counts is not None:
            active_counts = jnp.asarray(active_counts).reshape(groups)

            def body(state, chunk):
                xc, mc, chunk_index = chunk

                def encode(values):
                    branch_state, branch_x, branch_mask = values
                    return dense_body(branch_state, (branch_x, branch_mask))

                def skip(values):
                    branch_state, branch_x, _branch_mask = values
                    output_dtype = self.dtype or branch_x.dtype
                    encoded = jnp.zeros((groups, cw, branch_x.shape[-1]), dtype=output_dtype)
                    return branch_state, encoded

                return jax.lax.cond(
                    jnp.any(chunk_index * cw < active_counts),
                    encode,
                    skip,
                    (state, xc, mc),
                )

            chunk_indices = jnp.arange(n_chunks, dtype=jnp.int32)

            def sparse_scan(values):
                initial, chunks_x, chunks_mask, indices = values
                return jax.lax.scan(body, initial, (chunks_x, chunks_mask, indices))

            def dense_scan(values):
                initial, chunks_x, chunks_mask, _indices = values
                return jax.lax.scan(dense_body, initial, (chunks_x, chunks_mask))

            _, es = jax.lax.cond(
                jnp.max(active_counts) <= per_group - cw,
                sparse_scan,
                dense_scan,
                (state, x_chunks, mask_chunks, chunk_indices),
            )
        else:
            _, es = jax.lax.scan(dense_body, state, (x_chunks, mask_chunks))
        es = jnp.swapaxes(es, 0, 1).reshape(groups, n_chunks * cw, e)
        return es[:, :per_group].reshape(n, e)

    def _chunk_width(self, rows: int) -> int:
        """Resolve the scan block width for one independent device group."""
        if not self.feat_chunk:
            return rows
        return max(self.feat_chunk, -(-rows // 16))
