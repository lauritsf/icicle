import dataclasses
import sys

import numpy as np
import pandas as pd
import pytest
from flax import nnx
import jax.numpy as jnp

from experiments import hpo_regret, qtmeta_hpo
from icicle_jax.loading import ModelConfig, build_model
from icicle_jax.bar_distribution import BarDistribution
from reproduction.bench.qtmeta import load_qtmeta
from tests.reproduction.bench.test_adapters import write_qtmeta, _qt_base_args


@pytest.mark.parametrize('method', ['icicle', 'ftpfn', 'asha', 'random_search'])
@pytest.mark.parametrize('chain_batch', [1, 32])
def test_time_budget_entrypoint(tmp_path, monkeypatch, method, chain_batch):
    write_qtmeta(tmp_path, 'mtlbm/micro/set0/TOY',
                 {'a': [10, 40, 80], 'b': [20, 50, 90]},
                 {'a': [1800, 1800, 1800], 'b': [1800, 1800, 1800]},
                 {'a': [0, 0, 0], 'b': [0, 0, 0]},
                 {'a': _qt_base_args(), 'b': _qt_base_args()})
    task = load_qtmeta('micro', 'mtlbm/micro/set0/TOY', data_dir=tmp_path)
    model_cfg = ModelConfig(architecture='ftpfn' if method == 'ftpfn' else 'icicle',
                           config_dim=33, emsize=40, num_heads=2, hidden_dim=80,
                           n_feat_layers=0 if method == "ftpfn" else 1,
                           ragged=method != "ftpfn", qk_norm=method != "ftpfn",
                           n_icl_layers=1, num_bars=8, dtype='float32', icl_backend='xla', remat=False)
    model = build_model(model_cfg, nnx.Rngs(0))
    criterion = BarDistribution(borders=jnp.linspace(0, 1, 9))
    monkeypatch.setattr(hpo_regret, 'load_eval_model', lambda *a, **kw: (
        model, criterion, {'model': dataclasses.asdict(model_cfg)}, 7))
    output = tmp_path / 'result.parquet'
    argv = ['qtmeta_hpo', '--method', method, '--out', str(output), '--data-dir', str(tmp_path),
            '--seeds', '0', '1', '--split', 'micro', '--chain-batch', str(chain_batch)]
    if method in ('icicle', 'ftpfn'):
        argv += ['--model', str(tmp_path / 'run')]
    monkeypatch.setattr(sys, 'argv', argv)
    qtmeta_hpo.main()
    frame = pd.read_parquet(output)
    assert set(frame.seed) == {0, 1}
    assert set(frame.task) == {task.name}
    for seed, run in frame.groupby('seed'):
        assert run.accepted.tolist() == [True, True, False]
        assert run.cost.tolist() == [1800., 3600., 3600.]
        assert np.isnan(run.observed.iloc[-1])
        assert run.f_best.iloc[-1] == run.f_best.iloc[-2]
        first = qtmeta_hpo.initial_index(qtmeta_hpo.PROTOCOL, task.name, seed, 2)
        assert run.chosen.iloc[0] == first
    assert frame.runner.eq('qtmeta-time-v8').all()
    assert frame.chain_batch.eq(chain_batch).all()
    assert not output.with_suffix('.partial.parquet').exists()
