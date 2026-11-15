"""Mapping observed metrics into the model input range."""

import numpy as np
import pytest

from icicle.hpo.metric_map import DEFAULT_BAND, MetricMap


BAND_LO, BAND_HI = DEFAULT_BAND


def test_known_range_lands_on_the_band_not_on_zero_and_one():
    y = MetricMap(worst=0.0, best=1.0)(np.array([0.0, 0.5, 1.0]))
    np.testing.assert_allclose(y, [BAND_LO, 0.5, BAND_HI], atol=1e-6)
    # both reserves are non-empty -- the acquisition needs the upper one to have room
    assert y[0] > 0.0 and y[-1] < 1.0


def test_reserve_absorbs_values_outside_the_declared_range_before_clipping():
    m = MetricMap(worst=0.0, best=1.0)
    # one full anchor-range beyond each end is exactly the edge of [0, 1] for this band
    np.testing.assert_allclose(m(np.array([-1.0, 2.0])), [0.0, 1.0], atol=1e-6)
    inside = m(np.array([-0.5, 1.5]))
    assert 0.0 < inside[0] < BAND_LO and BAND_HI < inside[1] < 1.0


def test_non_finite_is_a_death_at_zero():
    m = MetricMap(worst=0.0, best=1.0)
    np.testing.assert_allclose(m(np.array([np.nan, np.inf, -np.inf, 0.5])), [0, 0, 0, 0.5])


def test_bounded_map_maximized_and_reversed():
    up = MetricMap(worst=0.0, best=200.0)
    span = BAND_HI - BAND_LO
    np.testing.assert_allclose(
        up(np.array([0.0, 50.0, 200.0])), [BAND_LO, BAND_LO + 0.25 * span, BAND_HI], atol=1e-6
    )
    # reversed bounds invert a minimized metric: worst=100 -> band_lo, best=0 -> band_hi
    down = MetricMap(worst=100.0, best=0.0, maximize=False)
    np.testing.assert_allclose(
        down(np.array([100.0, 25.0, 0.0])), [BAND_LO, BAND_LO + 0.75 * span, BAND_HI], atol=1e-6
    )
    assert down.maximize is False


def test_round_trip_returns_the_original_scale():
    # exact anywhere the map has not clipped -- the band plus its reserve, which for the
    # default band is a full anchor-range beyond each end. Only the clip is lossy.
    for m, raw in (
        (MetricMap(worst=0.0, best=1.0), np.array([-0.9, 0.0, 0.5, 1.0, 1.9])),
        (MetricMap(worst=100.0, best=0.0, maximize=False), np.array([-90.0, 0.0, 40.0, 100.0, 190.0])),
        (MetricMap(worst=-5.0, best=5.0), np.array([-14.0, -5.0, 0.0, 5.0, 14.0])),
    ):
        np.testing.assert_allclose(m.to_raw(m(raw)), raw, atol=1e-3)


@pytest.mark.parametrize("maximize", [True, False])
def test_cold_start_sits_at_the_band_midpoint(maximize):
    m = MetricMap(maximize=maximize)
    assert not m.ready
    np.testing.assert_allclose(m(np.array([3.0, 100.0])), 0.5 * (BAND_LO + BAND_HI), atol=1e-6)
    np.testing.assert_array_equal(np.isnan(m.to_raw(np.array([0.5]))), [True])
    m.fit([1.0])  # a single point still carries no scale
    assert not m.ready


def test_adaptive_learns_both_edges_and_the_incumbent_tops_the_band():
    m = MetricMap()
    m.fit([0.1, 0.4, 0.9])
    assert m.ready
    y = m(np.array([0.9]))
    assert y[0] == pytest.approx(BAND_HI, abs=1e-6)  # best observed -> top of the band
    assert m(np.array([1.5]))[0] > BAND_HI  # future improvement lands in the reserve


def test_adaptive_worst_edge_is_robust_to_one_catastrophic_run():
    good = list(np.linspace(0.4, 0.9, 40))
    clean, poisoned = MetricMap(), MetricMap()
    clean.fit(good)
    poisoned.fit([*good, -1e6])
    # a 0.05 quantile floor keeps the scale; a min/max floor would collapse it to nothing
    assert abs(poisoned(np.array([0.7]))[0] - clean(np.array([0.7]))[0]) < 0.05


def test_adaptive_one_sided_pins_the_known_edge():
    m = MetricMap(best=0.0, maximize=False)  # a loss: floor known, worst case unknown
    m.fit([2.0, 4.0, 8.0])
    assert m(np.array([0.0]))[0] == pytest.approx(BAND_HI, abs=1e-6)  # the known edge pins
    # The 0.05 quantile places the worst observation in the reserve below the band.
    assert m(np.array([7.6]))[0] == pytest.approx(BAND_LO, abs=1e-4)
    assert 0.0 < m(np.array([8.0]))[0] < BAND_LO


def test_adaptive_with_both_edges_fixed_ignores_observations():
    m = MetricMap(worst=0.0, best=10.0)
    before = m(np.array([5.0]))[0]
    m.fit([1.0, 2.0, 3.0])
    assert m(np.array([5.0]))[0] == pytest.approx(before)


def test_fit_replaces_history_and_ignores_nonfinite_observations():
    mapping = MetricMap().fit([-1000., 1000.])
    current = [2., 3., 4.]
    mapping.fit([*current, np.nan, np.inf, -np.inf])
    np.testing.assert_array_equal(mapping(current), MetricMap().fit(current)(current))
    mapping.fit([])
    assert not mapping.ready
    np.testing.assert_array_equal(mapping([3., np.nan]), [.5, 0.])


def test_loss_uses_all_observations_with_zero_as_best():
    mapping = MetricMap(best=0., maximize=False).fit([4., 6., 8.])
    np.testing.assert_allclose(mapping([7.8, 0.]), [BAND_LO, BAND_HI], atol=1e-7)
    mapping.fit([4., 6., 8., 20.])
    np.testing.assert_allclose(mapping([18.2, 0.]), [BAND_LO, BAND_HI], atol=1e-7)
    np.testing.assert_allclose(mapping.to_raw(mapping([4., 6., 8., 20.])), [4., 6., 8., 20.], atol=1e-5)


def test_minimized_metric_learns_both_bounds_in_original_units():
    values = np.array([-4., 0., 8.])
    mapping = MetricMap(maximize=False).fit(values)
    np.testing.assert_allclose(mapping.to_raw(DEFAULT_BAND), [7.2, -4.], atol=1e-6)
    assert np.all(np.diff(mapping(values)) < 0)
    np.testing.assert_allclose(mapping.to_raw(mapping(values)), values, atol=1e-6)


@pytest.mark.parametrize("worst,best,maximize", [(1., 0., True), (0., 1., False), (1., 1., True),
                                               (np.nan, 1., True), (None, np.inf, False)])
def test_invalid_fixed_bounds(worst, best, maximize):
    with pytest.raises(ValueError):
        MetricMap(worst=worst, best=best, maximize=maximize)
