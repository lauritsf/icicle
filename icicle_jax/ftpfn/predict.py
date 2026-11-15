"""Adapt published FT-PFN inference to freeze-thaw predictors and score functions.

Inputs use the reference tokenization: first-appearance curve IDs, configurations
matched by exact equality, and ID 0 for queries without an observed configuration.
Static padding allows repeated BO steps to reuse compiled shapes.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from icicle_jax import acquisition


def assign_curve_ids(ctx_hps: np.ndarray, qry_hps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The official id assignment: ``(context ids, query ids)``.

    First-appearance ordinals (1-based) over context rows then query rows, exact
    vector equality; queries of configs never observed in context get id 0.
    """
    n_ctx = len(ctx_hps)
    all_hps = np.concatenate([ctx_hps, qry_hps]) if n_ctx else np.asarray(qry_hps)
    _, first, inv = np.unique(all_hps, axis=0, return_index=True, return_inverse=True)
    ids = (np.argsort(np.argsort(first)) + 1)[inv]
    observed = np.zeros(ids.max() + 1, dtype=bool)
    observed[ids[:n_ctx]] = True
    observed[0] = False
    return ids[:n_ctx], np.where(observed[ids[n_ctx:]], ids[n_ctx:], 0)


def scan_curve_ids(point_cfg, ctx_keep, n_cfg: int):
    """Assign curve IDs to the scan runner's configuration-major context.

    The runner identifies configurations by table row and sorts context by
    (row, fidelity). Cumulative counts therefore give first-appearance IDs.
    This assumes distinct rows represent distinct configurations; the reference
    vector-based assignment instead merges identical HP vectors.

    ctx_keep excludes padding. Queries for unobserved rows receive ID 0.
    """
    b = point_cfg.shape[0]
    ctx_cfg = point_cfg[:, : ctx_keep.shape[1]]
    observed = (
        jnp.zeros((b, n_cfg), jnp.int32)
        .at[jnp.arange(b)[:, None], ctx_cfg]
        .max(ctx_keep.astype(jnp.int32))
    )
    ordinal = jnp.cumsum(observed, axis=1) * observed  # 1-based rank, 0 where unobserved
    return jnp.take_along_axis(ordinal, point_cfg, axis=1)


def make_ftpfn_scan_score_fn(
    model,
    criterion,
    *,
    borders,
    n_hps: int,
    max_distinct_curves: int,
    query_chunk_size: int | None = None,
    prefix_attention: bool = False,
):
    """Build a published FT-PFN scorer for run_freeze_thaw_scan.

    Context and queries use the reference token layout and a traced context
    split. Check max_distinct_curves before compilation because the scan cannot
    raise a Python exception when the embedding capacity is exceeded.
    """
    max_curves = model.idcurve_enc.embedding.shape[0] - 1
    if max_distinct_curves > max_curves:
        raise ValueError(f"{max_distinct_curves} distinct curves possible > {max_curves}")
    if query_chunk_size is not None and query_chunk_size < 1:
        raise ValueError("query_chunk_size must be positive")
    graphdef, state = nnx.split(model)
    borders = jnp.asarray(borders)

    def score_fn(_state, t, point_cfg, point_fid, y, sep, params, it):
        del _state, it  # The reference weights are captured by the closure.
        model = nnx.merge(graphdef, state)
        b, p = point_cfg.shape
        n_cfg = t["hps"].shape[1]
        ctx_pad = p - n_cfg  # queries are exactly one row per config, appended after context
        pos = jnp.broadcast_to(jnp.arange(p)[None, :], (b, p))
        # context padding sits past sep: the official packing parks it at id 0, t = 1, hps 0,
        # where it reads as an extra query whose output is sliced away
        keep = jnp.where(pos < ctx_pad, pos < sep, True)
        ids = scan_curve_ids(point_cfg, keep[:, :ctx_pad], n_cfg)

        def chunk_score(indices):
            chunk_cfg = jnp.concatenate(
                [point_cfg[:, :ctx_pad], point_cfg[:, ctx_pad + indices]], axis=1
            )
            chunk_fid = jnp.concatenate(
                [point_fid[:, :ctx_pad], point_fid[:, ctx_pad + indices]], axis=1
            )
            chunk_y = jnp.concatenate([y[:, :ctx_pad], y[:, ctx_pad + indices]], axis=1)
            chunk_keep = jnp.concatenate(
                [keep[:, :ctx_pad], jnp.ones((b, len(indices)), dtype=bool)], axis=1
            )
            chunk_ids = jnp.concatenate([ids[:, :ctx_pad], ids[:, ctx_pad + indices]], axis=1)
            hps = jnp.take_along_axis(t["hps"][:, :, :n_hps], chunk_cfg[..., None], axis=1)
            x = jnp.concatenate(
                [
                    jnp.where(chunk_keep, chunk_ids, 0).astype(jnp.float32)[..., None],
                    jnp.where(chunk_keep, chunk_fid, 1.0)[..., None],
                    jnp.where(chunk_keep[..., None], hps, 0.0),
                ],
                axis=-1,
            )
            packed_y = jnp.where(chunk_keep, chunk_y, 0.0)
            logits = (
                model.forward_padded_prefix(x, packed_y, sep, ctx_pad)
                if prefix_attention
                else model(x, packed_y, sep)
            ).astype(jnp.float32)
            probs = criterion.probs(logits[:, ctx_pad:])
            return acquisition.exceedance_prob(probs, borders, params)

        if query_chunk_size is None or query_chunk_size >= n_cfg:
            return chunk_score(jnp.arange(n_cfg))
        n_chunks = math.ceil(n_cfg / query_chunk_size)
        indices = jnp.minimum(
            jnp.arange(n_chunks * query_chunk_size).reshape(n_chunks, query_chunk_size),
            n_cfg - 1,
        )
        chunked = jax.lax.map(chunk_score, indices)  # (K, B, Q_chunk)
        return jnp.swapaxes(chunked, 0, 1).reshape(b, -1)[:, :n_cfg]

    return score_fn


def make_ftpfn_predict_fn(model, criterion, *, ctx_pad: int, qry_pad: int):
    """Bind the vendored FT-PFN into ``predict_fn(ctx_hps, ctx_fid, ctx_y, qry_hps, qry_fid)``."""
    ctx_pad, qry_pad = int(ctx_pad), int(qry_pad)
    max_curves = model.idcurve_enc.embedding.shape[0] - 1

    @nnx.jit
    def forward(model, x, y, sep):
        logits = model(x[None], y[None], sep).astype(jnp.float32)
        return criterion.probs(logits[0])

    def predict_fn(ctx_hps, ctx_fid, ctx_y, qry_hps, qry_fid):
        ctx_hps = np.atleast_2d(np.asarray(ctx_hps, dtype=np.float32))
        qry_hps = np.atleast_2d(np.asarray(qry_hps, dtype=np.float32))
        n_ctx, n_qry = len(ctx_hps), len(qry_hps)
        if n_ctx > ctx_pad or n_qry > qry_pad:
            raise ValueError(f"context {n_ctx} > {ctx_pad} or queries {n_qry} > {qry_pad}")

        ctx_ids, qry_ids = assign_curve_ids(ctx_hps, qry_hps)
        if n_ctx and int(ctx_ids.max()) > max_curves:
            raise ValueError(f"{int(ctx_ids.max())} distinct context curves > {max_curves}")

        p = ctx_pad + qry_pad
        x = np.zeros((p, 2 + qry_hps.shape[1]), dtype=np.float32)
        x[:, 1] = 1.0  # Padding follows the context and is excluded from the output.
        y = np.zeros(p, dtype=np.float32)
        x[:n_ctx, 0] = ctx_ids
        x[:n_ctx, 1] = ctx_fid
        x[:n_ctx, 2:] = ctx_hps
        y[:n_ctx] = ctx_y
        x[ctx_pad : ctx_pad + n_qry, 0] = qry_ids
        x[ctx_pad : ctx_pad + n_qry, 1] = qry_fid
        x[ctx_pad : ctx_pad + n_qry, 2:] = qry_hps

        probs = forward(model, jnp.asarray(x), jnp.asarray(y), n_ctx)
        return np.asarray(probs[ctx_pad : ctx_pad + n_qry])

    return predict_fn
