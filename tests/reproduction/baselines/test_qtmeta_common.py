import numpy as np

from reproduction.baselines.qtmeta_common import (
    initial_index,
    normalized_regret,
    random_search_events,
    validate_table_task,
)

PROTOCOL = {
    "initialization": {"domain_label": "test/start"},
    "random_search": {"domain_label": "test/order"},
}


def test_paired_initialization_is_stable_and_task_specific():
    a = initial_index(PROTOCOL, "qtmeta/micro/set0-A", 200, 101)
    b = initial_index(PROTOCOL, "qtmeta/micro/set0-A", 200, 101)
    assert a == b
    assert 0 <= a < 101
    assert (
        len(
            {
                initial_index(PROTOCOL, f"qtmeta/micro/set0-{task}", seed, 101)
                for task in ("A", "B")
                for seed in (200, 201, 202)
            }
        )
        > 1
    )


def test_table_validation_pins_ragged_prefix_and_finite_costs():
    curves = np.asarray([[0.2, 0.4, np.nan], [0.3, np.nan, np.nan]])
    costs = np.asarray([[2.0, 3.0, np.nan], [1.0, np.nan, np.nan]])
    np.testing.assert_array_equal(validate_table_task(curves, costs), [2, 1])


def test_random_search_uses_paired_start_and_excludes_overshoot():
    curves = np.asarray([[0.2, 0.4, 0.6], [0.3, 0.5, 0.7]], dtype=float)
    costs = np.asarray([[2.0, 2.0, 3.0], [1.0, 3.0, 3.0]], dtype=float)
    ids = ["a", "b"]
    events = random_search_events(
        protocol=PROTOCOL,
        task="qtmeta/micro/set0-A",
        bo_seed=200,
        candidate_ids_ordered=ids,
        curves=curves,
        costs=costs,
        cap_seconds=5.0,
        max_iterations=20,
    )
    expected_first = initial_index(PROTOCOL, "qtmeta/micro/set0-A", 200, 2)
    assert events[0]["proposed_config_index"] == expected_first
    assert events[-1]["accepted"] is False
    assert events[-1]["cumulative_table_cost_seconds"] <= 5.0
    assert events[-1]["cumulative_table_cost_seconds"] + events[-1]["table_step_cost_seconds"] > 5.0
    assert all(np.isfinite(event["table_step_cost_seconds"]) for event in events)


def test_random_search_allows_only_explicit_smoke_truncation():
    curves = np.asarray([[0.2, 0.3, 0.4, 0.5, 0.6]], dtype=float)
    costs = np.ones_like(curves)
    events = random_search_events(
        protocol=PROTOCOL,
        task="qtmeta/micro/set0-A",
        bo_seed=200,
        candidate_ids_ordered=["a"],
        curves=curves,
        costs=costs,
        cap_seconds=100.0,
        max_iterations=4,
        allow_truncation=True,
    )
    assert len(events) == 4
    assert all(event["accepted"] for event in events)


