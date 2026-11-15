"""Scaling sweeps and subprocess result persistence."""

import json
import sys
from types import SimpleNamespace

import pandas as pd

from experiments import bench_scaling as scaling


def test_sweeps_change_only_the_named_axis():
    cases = list(scaling.benchmark_cases(scaling.Args()))
    for axis in scaling.SWEEPS:
        group = [case for case in cases if case['axis'] == axis]
        assert [case[axis] for case in group] == scaling.SWEEPS[axis]
        for other in {'P', 'C', 'D'} - {axis}:
            assert len({case[other] for case in group}) == 1
        assert all(case['P'] >= case['C'] for case in group)


def test_saves_failed_and_trailing_skipped_cases(tmp_path, monkeypatch):
    output = tmp_path / 'nested' / 'scaling.parquet'
    cases = [
        dict(model='icicle', backend='xla', axis='D', P=2000, C=512, D=33),
        dict(model='ftpfn', backend='xla', axis='D', P=2000, C=512, D=33),
    ]
    monkeypatch.setattr(scaling, 'benchmark_cases', lambda args: iter(cases))
    monkeypatch.setattr(scaling.subprocess, 'run', lambda *a, **kw: SimpleNamespace(
        returncode=1, stdout='', stderr='RuntimeError: device failure\n'
    ))
    monkeypatch.setattr(sys, 'argv', ['scaling', '--out', str(output), '--ftpfn', 'reference.safetensors'])
    scaling.main()
    result = pd.read_parquet(output)
    assert result.status.tolist() == ['failed', 'cannot_run']
    assert result.reason.iloc[0] == 'RuntimeError: device failure'
    assert result.peak_bytes.isna().all()


def test_scaling_subprocess_with_tiny_model(tmp_path, monkeypatch):
    config = json.loads(scaling.Args().config.read_text())
    config['model'].update(
        emsize=16, num_heads=2, hidden_dim=32, n_icl_layers=1,
        num_bars=8, dtype='float32', feat_chunk=0,
    )
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    output = tmp_path / 'scaling.parquet'
    case = dict(model='icicle', backend='xla', axis='P', P=8, C=4, D=2)
    monkeypatch.setattr(scaling, 'benchmark_cases', lambda args: iter([case]))
    monkeypatch.setattr(sys, 'argv', [
        'scaling', '--config', str(config_path), '--out', str(output), '--repeats', '2',
    ])
    scaling.main()
    result = pd.read_parquet(output)
    assert result.status.tolist() == ['ok'], result.reason.tolist()
    assert result.t_median_s.gt(0).all()
    resolved = json.loads(result.model_config.iloc[0])
    assert resolved['emsize'] == 16 and resolved['config_dim'] == 2
    assert result.dtype.tolist() == ['float32']
