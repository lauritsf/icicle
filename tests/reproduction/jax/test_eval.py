"""Tests for the benchmark eval runner (tiny model, CPU)."""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from flax import nnx

from reproduction.bench import BenchmarkTask, EvalProtocol
from icicle_jax import checkpoint
from icicle_jax.bar_distribution import BarDistribution
from reproduction.jax.eval import _gather, evaluate
from icicle_jax.loading import load_eval_model
from icicle_jax.loading import ModelConfig, build_model
from icicle_jax.train import (
    DataConfig,
    OptimConfig,
    PriorConfig,
    RunConfig,
    make_optimizer,
    wsd_schedule,
)

MODEL_CFG = ModelConfig(
    config_dim=4,
    emsize=16,
    num_heads=4,
    hidden_dim=32,
    n_feat_layers=1,
    n_icl_layers=2,
    num_bars=8,
    icl_backend="xla",
    dtype="float32",
    remat=False,
)
PROTO = EvalProtocol(n_total=40, n_scenarios=4)


def make_task(n_cfg=10, b_max=8, n_hps=3, name="test/task", seed=0):
    rng = np.random.default_rng(seed)
    return BenchmarkTask(
        name=name,
        hps=rng.uniform(size=(n_cfg, n_hps)).astype(np.float32),
        curves=rng.uniform(size=(n_cfg, b_max)).astype(np.float32),
        fidelities=((np.arange(b_max) + 1) / b_max).astype(np.float32),
    )


def make_model():
    model = build_model(MODEL_CFG, nnx.Rngs(0))
    criterion = BarDistribution(borders=jnp.linspace(0.0, 1.0, MODEL_CFG.num_bars + 1))
    return model, criterion


def test_gather_masks_query_values():
    task = make_task()
    from reproduction.bench import iter_scenarios

    _, s = next(iter(iter_scenarios(task, PROTO, seed=0)))
    (config_hps, _pci, _fid, y, y_true), values, query, _gap, _final = _gather(
        task, s, p_pad=64, config_dim=4
    )
    np.testing.assert_array_equal(y[0, s.n_context :], 0.0)
    np.testing.assert_array_equal(y[0, : s.n_context], values[: s.n_context])
    assert query.sum() == s.n_total_eff - s.n_context
    assert np.isnan(y_true[0, s.n_total_eff :]).all()
    # the 4th HP column is padding (task has 3 HPs)
    np.testing.assert_array_equal(config_hps[0, :, 3], 0.0)


def test_gather_ragged_per_config_fidelities():
    from reproduction.bench.scenario import Scenario

    rng = np.random.default_rng(2)
    lengths, b_max = (4, 1, 3), 4
    curves = np.full((3, b_max), np.nan, dtype=np.float32)
    fid = np.full((3, b_max), np.nan, dtype=np.float32)
    for i, n in enumerate(lengths):
        curves[i, :n] = rng.uniform(size=n)
        fid[i, :n] = np.sort(rng.uniform(size=n))
    task = BenchmarkTask(
        name="t/ragged",
        hps=rng.uniform(size=(3, 3)).astype(np.float32),
        curves=curves,
        fidelities=fid,
    )
    # context: cfg0 prefix [0,1], cfg2 prefix [0]; queries: cfg0 {2,3}, cfg1 {0}, cfg2 {2}
    pts = [(0, 0), (0, 1), (2, 0), (0, 2), (0, 3), (1, 0), (2, 2)]
    s = Scenario(
        config_index=np.array([0, 1, 2], dtype=np.int32),
        point_config_idx=np.array([c for c, _ in pts], dtype=np.int32),
        level_index=np.array([e for _, e in pts], dtype=np.int32),
        n_context=3,
        n_total_eff=len(pts),
        alpha=1.0,
        context_fraction=0.5,
    )
    (_cfg, _pci, fids, _y, _yt), _values, _query, gap, is_final = _gather(
        task, s, p_pad=8, config_dim=3
    )
    np.testing.assert_array_equal(fids[0, :7], [fid[c, e] for c, e in pts])
    # cutoffs: cfg0 -> fid[0,1]; cfg1 -> 0 (no context); cfg2 -> fid[2,0]
    expect_gap = [fid[0, 2] - fid[0, 1], fid[0, 3] - fid[0, 1], fid[1, 0], fid[2, 2] - fid[2, 0]]
    np.testing.assert_allclose(gap[3:7], expect_gap, rtol=1e-6)
    np.testing.assert_array_equal(gap[:3], 0.0)
    # last observed level per config: 3, 0, 2
    np.testing.assert_array_equal(is_final[:7], [False, False, False, False, True, True, True])


def test_padding_invariance():
    model, criterion = make_model()
    task = make_task()
    kw = dict(config_dim=4, seed=0, feature_keys=1)
    a = evaluate(model, criterion, [("t", task)], PROTO, p_pad=40, **kw)
    b = evaluate(model, criterion, [("t", task)], PROTO, p_pad=96, **kw)
    for col in ("ll", "mse", "ll_final", "ll_near", "ll_far"):
        # Log-likelihood near zero keeps the float32 log-softmax rounding floor.
        np.testing.assert_allclose(a[col], b[col], rtol=1e-5, atol=1e-6)


def test_metric_parity_hand_rolled():
    model, criterion = make_model()
    task = make_task()
    df = evaluate(model, criterion, [("t", task)], PROTO, config_dim=4, seed=0, feature_keys=1)

    from reproduction.bench import iter_scenarios

    idx, s = next(iter(iter_scenarios(task, PROTO, seed=0)))
    arrays, values, query, _gap, _final = _gather(task, s, p_pad=PROTO.n_total, config_dim=4)
    config_hps, pci, fid, y, _y_true = (jnp.asarray(a) for a in arrays)
    fk = jax.random.fold_in(jax.random.key(0), 0)
    logits = model(config_hps, task.hps.shape[1], pci, fid, y, s.n_context, feature_key=fk)

    qlogits = np.asarray(logits[0])[query]
    qvals = values[s.n_context :]
    ll = -float(np.mean(np.asarray(criterion.nll(jnp.asarray(qlogits), jnp.asarray(qvals)))))
    mse = float(np.mean((np.asarray(criterion.mean(jnp.asarray(qlogits))) - qvals) ** 2))

    row = df[(df.scenario_idx == idx)].iloc[0]
    np.testing.assert_allclose(row.ll, ll, rtol=1e-6)
    np.testing.assert_allclose(row.mse, mse, rtol=1e-6)


def test_feature_key_rows_and_determinism():
    model, criterion = make_model()
    task = make_task()
    kw = dict(config_dim=4, seed=0, feature_keys=2)
    df = evaluate(model, criterion, [("t", task)], PROTO, **kw)
    assert len(df) == PROTO.n_scenarios * 2
    assert set(df.feature_key_idx) == {0, 1}
    # the identity draw is a real noise source: different keys score differently
    piv = df.pivot(index="scenario_idx", columns="feature_key_idx", values="ll")
    assert not np.allclose(piv[0], piv[1])
    df2 = evaluate(model, criterion, [("t", task)], PROTO, **kw)
    pd.testing.assert_frame_equal(df, df2)


def test_on_task_continual_persistence():
    model, criterion = make_model()
    tasks = [("t", make_task(name="test/a", seed=1)), ("t", make_task(name="test/b", seed=2))]
    seen = []
    df = evaluate(
        model,
        criterion,
        tasks,
        PROTO,
        config_dim=4,
        seed=0,
        feature_keys=1,
        on_task=lambda cur: seen.append(cur),
    )
    assert len(seen) == 2
    assert len(seen[0]) == PROTO.n_scenarios  # first task's rows already persisted
    pd.testing.assert_frame_equal(seen[-1], df)


def test_load_eval_model_roundtrip(tmp_path):
    model, criterion = make_model()
    model.decoder_linear2.bias[...] += 0.25
    optim_cfg = OptimConfig()
    schedule = wsd_schedule(
        optim_cfg.peak_lr, optim_cfg.warmup_steps, optim_cfg.decay_start, optim_cfg.decay_steps
    )
    opt = nnx.Optimizer(
        model, make_optimizer(schedule, optim_cfg.grad_clip, optim_cfg.beta2), wrt=nnx.Param
    )
    ckpt_dir = tmp_path / "ckpt"
    manager = checkpoint.make_manager(str(ckpt_dir))
    checkpoint.save(manager, 7, model, opt)
    manager.wait_until_finished()
    manager.close()

    cfgs = {
        "model": MODEL_CFG,
        "optim": optim_cfg,
        "prior": PriorConfig(),
        "data": DataConfig(),
        "run": RunConfig(checkpoint_dir=str(ckpt_dir)),
    }
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                **{k: dataclasses.asdict(v) for k, v in cfgs.items()},
            }
        )
    )

    model2, criterion2, cfg, step = load_eval_model(tmp_path)
    assert step == 7
    assert cfg["prior"]["extensions"] == "none"
    np.testing.assert_array_equal(criterion2.borders, criterion.borders)

    task = make_task()
    kw = dict(config_dim=4, seed=0, feature_keys=1)
    a = evaluate(model, criterion, [("t", task)], PROTO, **kw)
    b = evaluate(model2, criterion2, [("t", task)], PROTO, **kw)
    pd.testing.assert_frame_equal(a, b)

    from icicle_jax.export import export_run

    portable = export_run(tmp_path, out=tmp_path / "portable.safetensors")
    model3, criterion3, cfg3, step3 = load_eval_model(portable)
    assert step3 == step and cfg3["run"]["checkpoint_dir"] is None
    c = evaluate(model3, criterion3, [("t", task)], PROTO, **kw)
    pd.testing.assert_frame_equal(a, c)


def test_feature_seed_controls_and_identifies_predictions():
    model, criterion = make_model()
    task = make_task()
    first = evaluate(model, criterion, [("t", task)], PROTO,
                     config_dim=4, feature_keys=1, feature_seed=0)
    second = evaluate(model, criterion, [("t", task)], PROTO,
                      config_dim=4, feature_keys=1, feature_seed=1)
    assert first.feature_seed.eq(0).all()
    assert second.feature_seed.eq(1).all()
    np.testing.assert_array_equal(first.scenario_idx, second.scenario_idx)
    assert not np.allclose(first.ll, second.ll)
