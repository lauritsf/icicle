"""Run the evaluation commands on a small model and prepared benchmark table."""

import dataclasses
import sys

import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest
from flax import nnx

from experiments import eval, hpo_regret
from icicle_jax.bar_distribution import BarDistribution
from icicle_jax.loading import ModelConfig, build_model
from reproduction.bench._hp_spaces import LCBENCH_SPACE


@pytest.fixture
def tiny_run(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    table_dir = data_dir / "lcbench-tabular"
    table_dir.mkdir(parents=True)
    n_configs, n_epochs = 4, 6
    frame = pd.DataFrame({
        "val_balanced_accuracy": np.linspace(0.1, 0.9, n_configs * n_epochs),
        **{name: lower for name, lower, _, _ in LCBENCH_SPACE},
    }, index=pd.MultiIndex.from_product(
        [range(n_configs), range(n_epochs)], names=["id", "epoch"]
    ))
    frame.to_parquet(table_dir / "toy.parquet")
    run_dir = tmp_path / "model"
    run_dir.mkdir()
    model_config = ModelConfig(
        config_dim=len(LCBENCH_SPACE), emsize=16, num_heads=4, hidden_dim=32,
        n_icl_layers=1, num_bars=8, dtype="float32", icl_backend="xla", remat=False,
    )
    model = build_model(model_config, nnx.Rngs(0))
    criterion = BarDistribution(borders=jnp.linspace(0.0, 1.0, 9))
    config = {
        "model": dataclasses.asdict(model_config),
        "data": {"seq_len": 16},
        "prior": {"extensions": "none"},
    }
    for command in (eval, hpo_regret):
        monkeypatch.setattr(command, "load_eval_model", lambda *a, **kw: (
            model, criterion, config, 7
        ))
    return run_dir, data_dir


@pytest.mark.parametrize(
    "method", ["icicle", "ftpfn", "random_search", "asha", "asha_neps", "hyperband", "gp_ft", "gp_blackbox"]
)
def test_regret_runs_only_selected_method(tiny_run, tmp_path, monkeypatch, method):
    from experiments import hpo_regret as command

    run_dir, data_dir = tiny_run
    output = tmp_path / f"{method}.parquet"
    argv = [
        "hpo_regret", "--method", method, "--benchmark", "lcbench_ifbo",
        "--data-dir", str(data_dir), "--out", str(output),
        "--iters", "4", "--seeds", "0", "1", "--chain-batch", "1",
    ]
    if method == "icicle":
        argv += ["--model", str(run_dir)]
    elif method == "ftpfn":
        from icicle_jax.ftpfn.model import FTPFN

        model = FTPFN(
            config_dim=10, emsize=16, num_heads=2, hidden_dim=32,
            num_layers=1, num_bars=1000, max_curves=16, rngs=nnx.Rngs(0),
        )
        monkeypatch.setattr(command, "load_ftpfn", lambda path: model)
        argv += ["--model", str(run_dir / "reference.safetensors")]
    else:
        def unexpected_load(*args, **kwargs):
            pytest.fail("Baseline evaluation attempted to load a checkpoint")
        monkeypatch.setattr(command, "load_eval_model", unexpected_load)
    monkeypatch.setattr(sys, "argv", argv)
    command.main()

    result = pd.read_parquet(output)
    assert set(result.method) == {method}
    assert set(result.benchmark) == {"lcbench_ifbo"}
    assert set(result.seed) == {0, 1}
    assert result.groupby("seed").size().eq(4).all()
    assert np.isfinite(result.regret).all()
    assert (result.groupby("seed").regret.diff().dropna() <= 1e-6).all()
    assert not output.with_suffix(".partial.parquet").exists()
    if method == "icicle":
        assert result.step.eq(7).all()
        assert result.checkpoint.eq(str(run_dir.resolve())).all()
    elif method == "ftpfn":
        assert result.step.isna().all()
        assert result.checkpoint.eq(str((run_dir / "reference.safetensors").resolve())).all()
    else:
        assert result.checkpoint.isna().all() and result.step.isna().all()


def test_regret_preserves_completed_batches(tiny_run, tmp_path, monkeypatch):
    from experiments import hpo_regret as command

    _, data_dir = tiny_run
    output = tmp_path / "interrupted.parquet"
    run = command.run_random_search

    def interrupt_second_seed(task, n_iters, seed):
        if seed == 1:
            raise RuntimeError("interrupted")
        return run(task, n_iters, seed)

    monkeypatch.setattr(command, "run_random_search", interrupt_second_seed)
    monkeypatch.setattr(sys, "argv", [
        "hpo_regret", "--method", "random_search", "--benchmark", "lcbench_ifbo",
        "--data-dir", str(data_dir), "--out", str(output),
        "--iters", "4", "--seeds", "0", "1", "--chain-batch", "1",
    ])
    with pytest.raises(RuntimeError, match="interrupted"):
        command.main()
    result = pd.read_parquet(output.with_suffix(".partial.parquet"))
    assert result.seed.tolist() == [0] * 4
    assert not output.exists()


def test_eval_compares_checkpoints(tiny_run, tmp_path, monkeypatch):
    from experiments import eval as command

    run_dir, data_dir = tiny_run
    second = tmp_path / 'second-model'
    output = tmp_path / 'predictions.parquet'
    monkeypatch.setattr(sys, 'argv', [
        'eval', str(run_dir), str(second), '--benchmark', 'lcbench_ifbo',
        '--data-dir', str(data_dir), '--out', str(output),
        '--scenarios', '2', '--n-total', '16', '--feature-keys', '2',
    ])
    command.main()

    result = pd.read_parquet(output)
    assert len(result) == 8
    assert set(result.checkpoint) == {str(run_dir.resolve()), str(second.resolve())}
    assert result.step.eq(7).all()
    assert 'protocol_version' not in result
    assert np.isfinite(result[['ll', 'mse']]).all().all()
    columns = ['task', 'scenario_idx', 'feature_key_idx', 'context_size', 'll', 'mse']
    pd.testing.assert_frame_equal(
        result.iloc[:4][columns].reset_index(drop=True),
        result.iloc[4:][columns].reset_index(drop=True),
    )
    assert not output.with_suffix('.partial.parquet').exists()
    assert not (run_dir / 'eval').exists()


def test_eval_keeps_completed_checkpoints(tiny_run, tmp_path, monkeypatch):
    from experiments import eval as command

    run_dir, data_dir = tiny_run
    second = tmp_path / 'second-model'
    output = tmp_path / 'predictions.parquet'

    load_model = command.load_eval_model

    def load(path, **kwargs):
        if path == second:
            raise RuntimeError('interrupted')
        return load_model(path, **kwargs)

    monkeypatch.setattr(command, 'load_eval_model', load)
    monkeypatch.setattr(sys, 'argv', [
        'eval', str(run_dir), str(second), '--benchmark', 'lcbench_ifbo',
        '--data-dir', str(data_dir), '--out', str(output),
        '--scenarios', '1', '--n-total', '16', '--feature-keys', '1',
    ])
    with pytest.raises(RuntimeError, match='interrupted'):
        command.main()
    result = pd.read_parquet(output.with_suffix('.partial.parquet'))
    assert result.checkpoint.tolist() == [str(run_dir.resolve())]
    assert not output.exists()


def test_published_ftpfn_predictive_entrypoint(tiny_run, tmp_path, monkeypatch):
    from icicle_jax.ftpfn.model import FTPFN

    _, data_dir = tiny_run
    model = FTPFN(config_dim=10, emsize=16, num_heads=2, hidden_dim=32,
                  num_layers=1, num_bars=1000, max_curves=16, rngs=nnx.Rngs(0))
    monkeypatch.setattr(eval, 'load_ftpfn', lambda path: model)
    output = tmp_path / 'ftpfn.parquet'
    monkeypatch.setattr(sys, 'argv', [
        'eval', str(tmp_path / 'reference.safetensors'), '--benchmark', 'lcbench_ifbo',
        '--data-dir', str(data_dir), '--out', str(output), '--scenarios', '2',
        '--n-total', '16', '--feature-keys', '2',
    ])
    eval.main()
    result = pd.read_parquet(output)
    assert len(result) == 2 and result.architecture.eq('ftpfn').all()
    assert result.feature_keys.eq(1).all()
    assert result.feature_key_idx.eq(0).all()
    assert np.isfinite(result[['ll', 'mse']]).all().all()
    assert result.groupby('scenario_idx').ll.nunique().eq(1).all()
