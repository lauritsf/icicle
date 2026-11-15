"""Exercise the serving benchmark with small ONNX exports."""

import json
import sys

import numpy as np
import pytest

from experiments import serving_bench as bench
from icicle.model_format import expected_tensors, make_manifest
from icicle.onnx_export import build_model_graph


@pytest.fixture
def export_dir(tmp_path):
    config = dict(
        config_dim=3, emsize=8, num_heads=2, hidden_dim=16,
        n_feat_layers=1, n_icl_layers=1, num_bars=8, qk_norm=True,
    )
    manifest = make_manifest(config, step=7)
    rng = np.random.default_rng(0)
    tensors = {
        name: (rng.standard_normal(shape) * 0.1).astype(np.float32)
        for name, shape in expected_tensors(config).items()
    }
    (tmp_path / 'icicle.json').write_text(json.dumps(manifest))
    for fused in (False, True):
        name = 'icicle_fused.onnx' if fused else 'icicle.onnx'
        (tmp_path / name).write_bytes(build_model_graph(manifest, tensors, fused=fused).SerializeToString())
    return tmp_path


def configure(monkeypatch, export_dir, output, *extra):
    monkeypatch.setattr(bench, 'BASE', dict(p=8, c=2, d=3))
    monkeypatch.setattr(bench, 'AXES', dict(p=[8, 16], c=[2, 4], d=[3, 12]))
    monkeypatch.setattr(sys, 'argv', [
        'serving_bench', '--model-path', str(export_dir / 'icicle.onnx'), '--out', str(output),
        '--provider', 'cpu', '--repeats', '1', '--max-points', '8', '--threads', '1', *extra,
    ])


@pytest.mark.parametrize('graph', ['portable', 'fused'])
def test_serving_command(export_dir, tmp_path, monkeypatch, graph):
    output = tmp_path / 'results' / 'serving.json'
    configure(monkeypatch, export_dir, output, '--graph', graph)
    bench.main()
    report = json.loads(output.read_text())
    assert report['reference_max_error'] < 1e-3
    assert report['reference'] == 'portable-onnx-cpu'
    for cell in report['cells']:
        if cell['p'] > 8:
            assert 'skipped' in cell
        if 'skipped' not in cell:
            assert cell['median_ms'] > 0
    wide = next(cell for cell in report['cells'] if cell['axis'] == 'd' and cell['d'] == 12)
    assert 'skipped' in wide


def test_partial_results_include_active_provider(export_dir, tmp_path, monkeypatch):
    output = tmp_path / 'serving.json'
    configure(monkeypatch, export_dir, output)
    run = bench.time_cell
    calls = 0

    def interrupt(session, feeds, repeats):
        nonlocal calls
        calls += 1
        if calls == 2:
            saved = json.loads(output.read_text())
            assert saved['cells'][0]['median_ms'] > 0
            raise KeyboardInterrupt
        return run(session, feeds, repeats)

    monkeypatch.setattr(bench, 'time_cell', interrupt)
    with pytest.raises(KeyboardInterrupt):
        bench.main()
    saved = json.loads(output.read_text())
    assert saved['cells'][0]['median_ms'] > 0


def test_reference_failure_stops_timing(export_dir, tmp_path, monkeypatch):
    output = tmp_path / 'serving.json'
    make_session = bench.make_session

    class NonfiniteSession:
        def run(self, outputs, feeds):
            return [np.full((len(feeds['y']), 8), np.nan, np.float32)]

    calls = 0

    def corrupt_reference(*args, **kwargs):
        nonlocal calls
        calls += 1
        return NonfiniteSession() if calls == 2 else make_session(*args, **kwargs)

    monkeypatch.setattr(bench, 'make_session', corrupt_reference)
    configure(monkeypatch, export_dir, output)
    with pytest.raises(ValueError, match='nonfinite'):
        bench.main()
    assert not output.exists()
