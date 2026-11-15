"""Predictive NLL and MSE on benchmark scenarios.

ICICLE rows describe one task, scenario and feature-identity draw. Published
FT-PFN produces one row per scenario. Inputs are
padded to fixed point/configuration shapes so compatible scenarios share a JIT.
Checkpoint loading is provided by icicle_jax.loading.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx

from reproduction.bench.scenario import EvalProtocol, Scenario, iter_scenarios
from reproduction.bench.task import BenchmarkTask, valid_lengths


def _gather(task: BenchmarkTask, s: Scenario, p_pad: int, config_dim: int, *, published_ftpfn=False):
    """A Scenario's index arrays -> the surrogate's dense input layout + host-side masks."""
    n_hps = task.hps.shape[1]
    if n_hps > config_dim:
        raise ValueError(f"{task.name}: {n_hps} HPs exceed the model's config_dim {config_dim}")
    n_eff, n_cfg_used = s.n_total_eff, len(s.config_index)
    cfg_global = s.config_index[s.point_config_idx]

    config_hps = point_config_idx = None
    if not published_ftpfn:
        config_hps = np.zeros((1, p_pad, config_dim), dtype=np.float32)
        config_hps[0, :n_cfg_used, :n_hps] = task.hps[s.config_index]
        point_config_idx = np.zeros((1, p_pad), dtype=np.int32)
        point_config_idx[0, :n_eff] = s.point_config_idx
    # task.fidelities is (n_steps,) shared or (n_configs, n_steps) per-config (walltime
    # views); the sampler guarantees level_index stays within each config's observed support,
    # so a per-config gather never touches the NaN pad.
    fid2d = task.fidelities.ndim == 2
    fid_pts = (
        task.fidelities[cfg_global, s.level_index] if fid2d else task.fidelities[s.level_index]
    )
    fidelities = np.ones((1, p_pad), dtype=np.float32)
    fidelities[0, :n_eff] = fid_pts

    values = task.curves[cfg_global, s.level_index]
    y = np.zeros((1, p_pad), dtype=np.float32)
    y[0, : s.n_context] = values[: s.n_context]
    # Full targets for the metric kernel; NaN outside the real points (nll zeroes NaN targets,
    # and the host aggregation masks to real queries anyway).
    y_true = np.full((1, p_pad), np.nan, dtype=np.float32)
    y_true[0, :n_eff] = values

    # Per-query extrapolation distance: query fidelity minus its curve's cutoff fidelity
    # (0 for from-scratch queries -- curves with no context).
    cuts = np.bincount(s.point_config_idx[: s.n_context], minlength=n_cfg_used)
    cut_levels = np.maximum(cuts - 1, 0)
    cutoff_fid = np.where(
        cuts > 0,
        task.fidelities[s.config_index, cut_levels] if fid2d else task.fidelities[cut_levels],
        0.0,
    )
    query = np.zeros(p_pad, dtype=bool)
    query[s.n_context : n_eff] = True
    gap = np.zeros(p_pad, dtype=np.float32)
    gap[s.n_context : n_eff] = (fid_pts - cutoff_fid[s.point_config_idx])[s.n_context :]
    # "Final" = the curve's last observed level (== the global last column when rectangular).
    is_final = np.zeros(p_pad, dtype=bool)
    is_final[:n_eff] = s.level_index == valid_lengths(task.curves)[cfg_global] - 1
    return (config_hps, point_config_idx, fidelities, y, y_true), values, query, gap, is_final


def evaluate(
    model,
    criterion,
    tasks,
    protocol: EvalProtocol,
    *,
    config_dim: int,
    seed: int = 0,
    feature_keys: int = 8,
    feature_seed: int = 0,
    p_pad: int | None = None,
    log=lambda *_: None,
    on_task=None,
    published_ftpfn: bool = False,
) -> pd.DataFrame:
    """Score (benchmark, task) pairs under the protocol; one row per (task, scenario, key).

    ``tasks`` yields ``(benchmark_label, BenchmarkTask)``. All arrays are padded to
    ``p_pad`` (default ``protocol.n_total``) so every forward shares one compiled shape.
    ``on_task``, if given, receives the accumulated DataFrame after every task -- the hook
    for continual persistence, so a killed job keeps everything scored so far.
    """
    p_pad = p_pad or protocol.n_total
    if published_ftpfn:
        feature_keys = 1

    def frame(rows):
        df = pd.DataFrame(rows)
        if not df.empty:
            df["protocol_mode"] = protocol.mode
            df["seed"] = seed
            df["feature_seed"] = feature_seed
        return df

    # Categorical HPs use their normalized coordinates; no separate category metadata
    # is passed to the model.
    @nnx.jit
    def eval_step(model, config_hps, num_params, point_config_idx, fidelities, y, sep, y_true, fk):
        logits = model(
            config_hps, num_params, point_config_idx, fidelities, y, sep, feature_key=fk
        ).astype(jnp.float32)
        return criterion.nll(logits, y_true), criterion.mean(logits)

    @nnx.jit
    def ftpfn_step(model, x, y, sep, y_true):
        logits = model(x, y, sep).astype(jnp.float32)
        return criterion.nll(logits, y_true), criterion.mean(logits)

    # Batch feature-identity draws in one forward pass. Split each scalar key
    # before batching to match the scalar-key feature_identity convention.
    fkeys = None if published_ftpfn else jnp.stack(
        [
            jax.random.split(jax.random.fold_in(jax.random.key(feature_seed), k), 1)[0]
            for k in range(feature_keys)
        ]
    )
    rows = []
    for benchmark, task in tasks:
        n_scen = 0
        for idx, s in iter_scenarios(task, protocol, seed):
            arrays, values, query, gap, is_final = _gather(
                task, s, p_pad, config_dim, published_ftpfn=published_ftpfn
            )
            n_scen += 1
            qvals = values[s.n_context :]
            qgap, qfinal = gap[query], is_final[query]
            near = qgap <= np.median(qgap)
            if published_ftpfn:
                from icicle_jax.ftpfn.predict import assign_curve_ids

                hps = task.hps[s.config_index[s.point_config_idx]]
                ctx_ids, qry_ids = assign_curve_ids(hps[:s.n_context], hps[s.n_context:])
                ids = np.concatenate([ctx_ids, qry_ids])
                if ids.max() >= model.idcurve_enc.embedding.shape[0]:
                    raise ValueError(f"{task.name}: scenario exceeds FT-PFN curve capacity")
                x = np.zeros((1, p_pad, 2 + hps.shape[1]), np.float32)
                x[:, :, 1] = 1.0
                x[0, :s.n_total_eff, 0] = ids
                x[0, :s.n_total_eff, 1] = arrays[2][0, :s.n_total_eff]
                x[0, :s.n_total_eff, 2:] = hps
                nll, mean = ftpfn_step(
                    model, jnp.asarray(x), jnp.asarray(arrays[3]), s.n_context, jnp.asarray(arrays[4])
                )
            else:
                config_hps, point_config_idx, fidelities, y, y_true = (
                    jnp.repeat(jnp.asarray(a), feature_keys, axis=0) for a in arrays
                )
                nll, mean = eval_step(
                    model,
                    config_hps,
                    task.hps.shape[1],
                    point_config_idx,
                    fidelities,
                    y,
                    s.n_context,
                    y_true,
                    fkeys,
                )
            nll, mean = np.asarray(nll), np.asarray(mean)
            for k in range(feature_keys):
                qnll = nll[k][query]
                qse = (mean[k][query] - qvals) ** 2
                rows.append(
                    {
                        "benchmark": benchmark,
                        "task": task.name,
                        "scenario_idx": idx,
                        "feature_key_idx": k,
                        "context_fraction": s.context_fraction,
                        "context_size": s.n_context,
                        "alpha": s.alpha,
                        "n_total": protocol.n_total,
                        "n_total_eff": s.n_total_eff,
                        "n_query": int(query.sum()),
                        "ll": -float(qnll.mean()),
                        "mse": float(qse.mean()),
                        "ll_final": -float(qnll[qfinal].mean()) if qfinal.any() else np.nan,
                        "mse_final": float(qse[qfinal].mean()) if qfinal.any() else np.nan,
                        "ll_near": -float(qnll[near].mean()),
                        "ll_far": -float(qnll[~near].mean()) if (~near).any() else np.nan,
                    }
                )
        log(f"{task.name}: {n_scen} scenarios x {feature_keys} keys")
        if on_task is not None:
            on_task(frame(rows))
    return frame(rows)
