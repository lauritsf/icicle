"""JAX prediction from configuration vectors and learning-curve observations.

Fixed buffer capacities let successive calls reuse compiled shapes.
Benchmark optimization loops live in reproduction.jax.hpo_scan.
"""

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

def make_raw_predict_fn(
    model, criterion, *, config_dim: int, ctx_pad: int, qry_pad: int, cfg_pad: int | None = None,
    feature_seed: int = 0
):
    """Build a predictor accepting raw configuration vectors and fidelity values.

    Deduplicate configurations referenced by context and queries on each call.
    cfg_pad bounds the resulting table; ctx_pad and qry_pad fix the point-array
    capacities for compilation reuse. By default cfg_pad accommodates distinct
    configurations at every context and query position.
    """
    ctx_pad, qry_pad = int(ctx_pad), int(qry_pad)
    cfg_pad = ctx_pad + qry_pad if cfg_pad is None else int(cfg_pad)
    feature_key = jax.random.fold_in(jax.random.key(feature_seed), 0)

    @nnx.jit(static_argnums=(2,))
    def forward(model, hps, n_hps, point_cfg, point_fid, y, sep):
        logits = model(
            hps, n_hps, point_cfg[None], point_fid[None], y[None], sep, feature_key=feature_key
        ).astype(jnp.float32)
        return criterion.probs(logits[0])

    def predict_fn(ctx_hps, ctx_fid, ctx_y, qry_hps, qry_fid):
        ctx_hps = np.atleast_2d(np.asarray(ctx_hps, dtype=np.float32))
        qry_hps = np.atleast_2d(np.asarray(qry_hps, dtype=np.float32))
        n_ctx, n_qry, n_hps = len(ctx_hps), len(qry_hps), qry_hps.shape[1]
        if n_ctx > ctx_pad or n_qry > qry_pad:
            raise ValueError(f"context {n_ctx} > {ctx_pad} or queries {n_qry} > {qry_pad}")
        if n_hps > config_dim:
            raise ValueError(f"{n_hps} HPs > model config_dim {config_dim}")

        all_hps = np.concatenate([ctx_hps, qry_hps])
        uniq, inv = np.unique(all_hps, axis=0, return_inverse=True)
        if len(uniq) > cfg_pad:
            raise ValueError(f"{len(uniq)} distinct rows > cfg_pad {cfg_pad}")
        hps = np.zeros((1, cfg_pad, config_dim), dtype=np.float32)
        hps[0, : len(uniq), :n_hps] = uniq

        p = ctx_pad + qry_pad
        point_cfg = np.zeros(p, dtype=np.int32)
        point_fid = np.ones(p, dtype=np.float32)
        y = np.zeros(p, dtype=np.float32)
        point_cfg[:n_ctx] = inv[:n_ctx]
        point_fid[:n_ctx] = ctx_fid
        y[:n_ctx] = ctx_y
        point_cfg[ctx_pad : ctx_pad + n_qry] = inv[n_ctx:]
        point_fid[ctx_pad : ctx_pad + n_qry] = qry_fid
        probs = forward(
            model,
            jnp.asarray(hps),
            n_hps,
            jnp.asarray(point_cfg),
            jnp.asarray(point_fid),
            jnp.asarray(y),
            n_ctx,
        )
        return np.asarray(probs[ctx_pad : ctx_pad + n_qry])

    return predict_fn
