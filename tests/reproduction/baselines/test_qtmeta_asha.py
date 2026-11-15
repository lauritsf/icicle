import numpy as np

from reproduction.baselines.qtmeta_asha import run_asha


def test_exact_cap_and_exhaustion():
    objective = np.array([[.1, .5, .8], [.2, .6, .9]])
    costs = np.full_like(objective, 2.)
    kw = dict(paired_index=0, task_id='toy', seed=0, permutation_seed='test')
    rows = run_asha(objective, costs, cap=4., **kw)
    assert [r['accepted'] for r in rows] == [True, True, False]
    assert [r['cost'] for r in rows] == [2., 4., 4.]
    assert rows[-1]['observed'] is None
    assert rows[-1]['f_best'] == rows[-2]['f_best']
    rows = run_asha(objective, costs, cap=100., **kw)
    assert len(rows) == 6
    assert len({(r['chosen'], r['budget']) for r in rows}) == 6
    assert rows[-1]['f_best'] == .9


def test_ragged_prefix_and_float64_ranking():
    objective = np.array([[.5, .7, np.nan], [.5 + 1e-10, .8, .9]])
    costs = np.where(np.isfinite(objective), 1., np.nan)
    rows = run_asha(objective, costs, cap=100., paired_index=0,
                    task_id='toy', seed=0, permutation_seed='test')
    assert len(rows) == 5
    assert [(r['chosen'], r['budget']) for r in rows[:3]] == [(0, 1), (1, 1), (1, 2)]
    assert rows[1]['observed'] == .5 + 1e-10


def test_raw_scores_preserve_precision_and_order(monkeypatch, tmp_path):
    from reproduction.bench import qtmeta
    from reproduction.bench.task import BenchmarkTask

    values = {'b': [50.00000001, 80.], 'a': [50.]}
    tables = {'eval_top1': {'mtlbm/micro/set0/TOY': values}}
    monkeypatch.setattr(qtmeta, '_split_tables', lambda *args: tables)
    task = BenchmarkTask('qtmeta/micro/set0-TOY', np.zeros((2, 1), np.float32),
                         np.array([[.5, np.nan], [.5, .8]], np.float32),
                         np.array([.5, 1.], np.float32))
    scores = qtmeta.raw_scores(task, data_dir=tmp_path)
    assert scores.dtype == np.float64
    assert scores[1, 0] > scores[0, 0]
    assert np.isnan(scores[0, 1])
