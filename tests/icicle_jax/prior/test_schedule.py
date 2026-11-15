"""Check configuration capacities, context prefixes and query sampling."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from icicle_jax.prior.observations import sample_config_indices, sample_fidelities


@pytest.mark.parametrize("n_steps", [1, 2, 7])
def test_ordering_shape_and_cap(n_steps):
    seq_len = 24
    sample = jax.jit(sample_config_indices, static_argnames=("seq_len",))
    for seed in range(4):
        ordering = np.asarray(sample(jax.random.key(seed), seq_len, jnp.int32(n_steps)))
        assert ordering.shape == (seq_len,)
        assert ordering.min() >= 0 and ordering.max() < seq_len
        counts = np.bincount(ordering, minlength=seq_len)
        assert counts.sum() == seq_len
        assert counts.max() <= n_steps


@pytest.mark.parametrize("n_steps", [1, 6, 7])
@pytest.mark.parametrize("context_size", [0, 14, 24])
@pytest.mark.parametrize("recall_probability", [0.0, 0.75, 1.0])
def test_context_prefix_and_query_steps(n_steps, context_size, recall_probability):
    ordering = sample_config_indices(jax.random.key(0), 24, n_steps)
    sample = jax.jit(sample_fidelities, static_argnames=("n_steps_max", "recall_probability"))
    fidelities = np.asarray(sample(
        jax.random.key(1), ordering, jnp.int32(context_size), jnp.int32(n_steps),
        n_steps_max=7, recall_probability=recall_probability,
    ))
    assert np.all((fidelities > 0.0) & (fidelities <= 1.0))
    ordering = np.asarray(ordering)
    steps = np.rint(fidelities * n_steps).astype(int)
    context = np.arange(24) < context_size
    for config_idx in np.unique(ordering):
        context_steps = np.sort(steps[(ordering == config_idx) & context])
        query_steps = steps[(ordering == config_idx) & ~context]
        np.testing.assert_array_equal(context_steps, np.arange(1, len(context_steps) + 1))
        assert len(np.unique(query_steps)) == len(query_steps)
        assert np.all((query_steps >= 1) & (query_steps <= n_steps))
        if recall_probability == 0.0:
            assert np.all(query_steps > len(context_steps))


def test_recall_can_revisit_context_steps():
    ordering = jnp.zeros(6, jnp.int32)
    recalled = 0
    for seed in range(4):
        fidelities = np.asarray(sample_fidelities(
            jax.random.key(seed), ordering, 4, 6, n_steps_max=7, recall_probability=1.0,
        ))
        recalled += np.count_nonzero(np.isin(fidelities[4:], fidelities[:4]))
    assert recalled > 0
