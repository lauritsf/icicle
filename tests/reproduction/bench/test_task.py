"""Unit tests for the shared BenchmarkTask helpers."""

import numpy as np
import pytest

from reproduction.bench.task import (
    BenchmarkTask,
    normalize_hp,
    resolve_data_dir,
    valid_lengths,
)


def test_valid_lengths_and_cat_card_default():
    curves = np.array([[0.1, 0.2, 0.3], [0.4, np.nan, np.nan]], dtype=np.float32)
    np.testing.assert_array_equal(valid_lengths(curves), [3, 1])
    task = BenchmarkTask("t", np.zeros((2, 1), np.float32), curves, np.ones(3, np.float32))
    assert task.cat_card is None


def test_normalize_hp_linear():
    out = normalize_hp(np.array([0.1, 0.545, 0.99]), 0.1, 0.99, log=False)
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-12)


def test_normalize_hp_log():
    out = normalize_hp(np.array([1e-4, 1e-3, 1e-1]), 1e-4, 1e-1, log=True)
    np.testing.assert_allclose(out, [0.0, 1 / 3, 1.0], atol=1e-12)


def test_normalize_hp_clips_out_of_bounds():
    out = normalize_hp(np.array([0.0, 2.0]), 0.5, 1.0, log=False)
    np.testing.assert_allclose(out, [0.0, 1.0])


def test_resolve_data_dir_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("ICICLE_BENCH_DATA", "/from/env")
    assert resolve_data_dir(tmp_path) == tmp_path  # explicit arg wins
    assert str(resolve_data_dir(None)) == "/from/env"
    monkeypatch.delenv("ICICLE_BENCH_DATA")
    with pytest.raises(RuntimeError, match="ICICLE_BENCH_DATA"):
        resolve_data_dir(None)


@pytest.mark.parametrize("curve", [[.1, np.nan, .3], [.1, np.inf], [np.nan, np.nan]])
def test_invalid_curve_prefix(curve):
    with pytest.raises(ValueError, match="finite prefix"):
        valid_lengths(np.asarray([curve]))
