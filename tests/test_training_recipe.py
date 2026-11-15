"""Config loading and CLI overrides for the training command."""

import dataclasses
import json
import sys
from pathlib import Path

import pytest

from experiments import train as entry
from icicle_jax.loading import ModelConfig
from icicle_jax.train import OptimConfig, PriorConfig, DataConfig, RunConfig

ROOT = Path(__file__).resolve().parents[1]


def test_release_config_and_overrides(monkeypatch, tmp_path):
    config = ROOT / 'configs/icicle.json'
    recipe = entry.load_recipe(config)
    calls = []
    monkeypatch.setattr(entry, 'train', lambda *args: calls.append(args))
    monkeypatch.setattr(sys, 'argv', [
        'train', 'icicle', '--run.seed', '7', '--run.checkpoint-dir', str(tmp_path / 'ckpt'),
        '--prior.layers', '3', '6', '--model.architecture', 'ftpfn',
    ])
    entry.main()
    model, optim, prior, data, run = calls.pop()
    assert model == dataclasses.replace(recipe.model, architecture='ftpfn')
    assert (optim, data) == (recipe.optim, recipe.data)
    assert prior == dataclasses.replace(recipe.prior, layers=(3, 6))
    assert run == dataclasses.replace(recipe.run, seed=7, checkpoint_dir=str(tmp_path / 'ckpt'))


def test_smoke_config():
    recipe = entry.load_recipe(ROOT / 'configs/smoke.json')
    assert recipe.model == ModelConfig(config_dim=6, emsize=32, num_heads=4, hidden_dim=64,
                                      n_icl_layers=2, num_bars=64, icl_backend='xla')
    assert recipe.optim == OptimConfig(warmup_steps=2, decay_start=6, decay_steps=2)
    assert recipe.prior == PriorConfig(hidden=(8, 16), table_networks=64, table_configs=16)
    assert recipe.data == DataConfig(seq_len=64, batch_size=4, n_steps_max=16)
    assert recipe.run == RunConfig(num_steps=8, print_every=1, val_every=4, val_batches=2)


def test_unknown_config_field_rejected(tmp_path):
    values = json.loads((ROOT / 'configs/icicle.json').read_text())
    values['model']['num_head'] = values['model'].pop('num_heads')
    config = tmp_path / 'bad.json'
    config.write_text(json.dumps(values))
    with pytest.raises(TypeError, match='num_head'):
        entry.load_recipe(config)


def test_ftpfn_config_and_cli(monkeypatch):
    recipe = entry.load_recipe(ROOT / 'configs/ftpfn_icicle_prior.json')
    assert recipe.model.architecture == 'ftpfn'
    assert (recipe.model.emsize, recipe.model.config_dim, recipe.model.ftpfn_max_curves) == (512, 10, 1000)
    assert recipe.optim.peak_lr == 3e-4
    assert (recipe.optim.warmup_steps, recipe.optim.decay_start, recipe.optim.decay_steps) == (4000, 12000, 4000)
    assert recipe.run.num_steps == 16000
    assert recipe.run.checkpoint_dir is None
    assert recipe.prior == entry.load_recipe(ROOT / 'configs/icicle.json').prior
    calls = []
    monkeypatch.setattr(entry, 'train', lambda *args: calls.append(args))
    monkeypatch.setattr(sys, 'argv', ['train', 'ftpfn-icicle-prior', '--run.seed', '2'])
    entry.main()
    assert calls == [(recipe.model, recipe.optim, recipe.prior, recipe.data,
                      dataclasses.replace(recipe.run, seed=2))]


def test_ftpfn_paper_recipe():
    recipe = entry.load_recipe(ROOT / 'configs/ftpfn_paper.json')
    assert recipe.model.architecture == 'ftpfn'
    assert recipe.model.ftpfn_initialization == 'pytorch'
    assert recipe.model.dtype == 'float32'
    assert (recipe.model.emsize, recipe.model.num_heads, recipe.model.hidden_dim,
            recipe.model.n_icl_layers, recipe.model.config_dim) == (512, 4, 1024, 6, 10)
    assert recipe.optim.peak_lr == 1e-4
    assert recipe.optim.warmup_steps == recipe.run.num_steps // 4
    assert recipe.data.batch_size * recipe.run.num_steps == 2_000_000
    assert recipe.prior.layers == (8, 15)
    assert recipe.prior.extensions == 'none'
    assert recipe.prior.recall_probability == 0
    assert not recipe.prior.hp_types_poisson
    assert not recipe.prior.arity_trunc_poisson
