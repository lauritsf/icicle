"""Tests for the training core (WSD schedule + query loss)."""

import json

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from icicle_jax.bar_distribution import BarDistribution
from icicle_jax.loading import ModelConfig
from icicle_jax.train import (
    DataConfig,
    OptimConfig,
    PriorConfig,
    RunConfig,
    query_nll,
    ragged_bucket_bounds,
    train,
    wsd_schedule,
)

jax.config.update("jax_enable_x64", True)


def test_wsd_schedule():
    sched = wsd_schedule(1e-3, warmup_steps=10, decay_start=30, decay_steps=10)
    assert float(sched(0)) == 0.0  # starts at 0
    assert sched(5) < sched(10)  # warms up
    np.testing.assert_allclose(float(sched(10)), 1e-3)  # peak reached at end of warmup
    np.testing.assert_allclose(float(sched(30)), 1e-3)  # still peak at the (absolute) decay start
    assert float(sched(40)) < 1e-9  # fully decayed decay_steps later


def test_ragged_bounds_include_one_regular_small_bound():
    assert ragged_bucket_bounds(128_000) == (
        16_384,
        32_000,
        64_000,
        96_000,
        128_000,
    )
    assert ragged_bucket_bounds(64_000) == (
        8_192,
        16_000,
        32_000,
        48_000,
        64_000,
    )


def test_wsd_schedule_open_ended_holds_at_peak():
    sched = wsd_schedule(1e-3, warmup_steps=10, decay_start=None, decay_steps=10)
    np.testing.assert_allclose(float(sched(10)), 1e-3)
    np.testing.assert_allclose(float(sched(10_000)), 1e-3)  # never decays on its own


def test_query_nll_masks_context_and_nan():
    crit = BarDistribution(borders=jnp.linspace(0.0, 1.0, 9))  # 8 equal bars over [0,1]
    logits = jnp.zeros((2, 6, 8))  # uniform -> density 1 -> nll 0 everywhere
    y = jnp.full((2, 6), 0.5)
    np.testing.assert_allclose(float(query_nll(crit, logits, y, single_eval_pos=3)), 0.0, atol=1e-6)
    # a NaN query target is ignored, loss stays finite
    y_nan = y.at[0, 4].set(jnp.nan)
    assert np.isfinite(float(query_nll(crit, logits, y_nan, 3)))


def test_train_checkpoints_and_resumes(tmp_path):
    ck = str(tmp_path / "ck")
    tiny = dict(
        model_cfg=ModelConfig(
            emsize=16, num_heads=1, hidden_dim=32, n_feat_layers=1, n_icl_layers=2, num_bars=32
        ),
        optim_cfg=OptimConfig(warmup_steps=1, decay_start=None, decay_steps=1),
        prior_cfg=PriorConfig(layers=(2, 4), hidden=(8, 16), table_networks=16, table_configs=8),
        data_cfg=DataConfig(seq_len=32, batch_size=2, n_steps_max=8),
    )
    run = dict(print_every=10, val_every=10, val_batches=1, checkpoint_dir=ck, checkpoint_every=2)
    (tmp_path / "metrics.csv").write_text("step,metric,value\n999,loss,123\n")
    train(**tiny, run_cfg=RunConfig(num_steps=3, **run), log=lambda _m: None)  # saves at step 2

    assert "999,loss,123" not in (tmp_path / "metrics.csv").read_text()
    original = (tmp_path / "config.json").read_bytes()
    logs = []
    train(**tiny, run_cfg=RunConfig(num_steps=5, **run), log=logs.append)  # resumes at 3
    assert any("resuming from step 3" in m for m in logs)
    assert (tmp_path / "config.json").read_bytes() == original
    with pytest.raises(ValueError, match="resume changes run"):
        train(**tiny, run_cfg=RunConfig(num_steps=6, seed=1, **run), log=lambda _: None)
    assert (tmp_path / "config.json").read_bytes() == original


def test_query_nll_ignores_context_positions():
    crit = BarDistribution(borders=jnp.linspace(0.0, 1.0, 9))
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(size=(2, 6, 8)))
    y = jnp.asarray(rng.uniform(size=(2, 6)))
    base = query_nll(crit, logits, y, single_eval_pos=3)
    y_ctx_changed = y.at[:, :3].set(0.99)  # only context targets changed
    np.testing.assert_allclose(float(query_nll(crit, logits, y_ctx_changed, 3)), float(base))


def test_resume_recipe_checks_cosine_duration_and_scientific_settings():
    from icicle_jax.train import check_resume_config

    config = {"model": {}, "optim": {}, "prior": {}, "data": {},
              "run": {"num_steps": 10, "seed": 0}, "trainer": "ftpfn-paper"}
    requested = json.loads(json.dumps(config))
    requested["run"]["num_steps"] = 20
    with pytest.raises(ValueError, match="run"):
        check_resume_config(config, requested)
    requested["run"]["num_steps"] = 10
    requested["run"]["print_every"] = 5
    check_resume_config(config, requested)
    for field in ("model", "optim", "prior", "data"):
        changed = {**requested, field: {"changed": True}}
        with pytest.raises(ValueError, match=field):
            check_resume_config(config, changed)
