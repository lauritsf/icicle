"""Check FT-PFN token packing, gradients and training resume."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from icicle_jax.ftpfn.train import TrainableFTPFN, pack_tokens, presentation_randomness
from icicle_jax.loading import ModelConfig
from icicle_jax.train import DataConfig, OptimConfig, PriorConfig, RunConfig, train


def _model(*, remat=True, dtype=None, config_dim=10, max_curves=8, random_slots=False,
           initialization="flax"):
    return TrainableFTPFN(
        emsize=16,
        num_heads=2,
        hidden_dim=32,
        num_layers=2,
        num_bars=8,
        config_dim=config_dim,
        max_curves=max_curves,
        dtype=dtype,
        remat=remat,
        random_slots=random_slots,
        initialization=initialization,
        rngs=nnx.Rngs(17),
    )


def _inputs():
    hps = jnp.asarray([[[0.1, 0.2, 9.0], [0.4, 0.5, 9.0], [0.7, 0.8, 9.0]]], jnp.float32)
    idx = jnp.asarray([[2, 0, 2, 1, 1, 0]])
    fidelity = jnp.asarray([[0.1, 0.1, 0.2, 0.1, 0.2, 0.7]], jnp.float32)
    y = jnp.asarray([[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]], jnp.float32)
    return hps, idx, fidelity, y


def test_packing_scales_active_hps_and_ignores_query_occurrences():
    hps, idx, fidelity, _ = _inputs()
    pack = jax.jit(lambda n: pack_tokens(hps, n, idx, fidelity, 3, config_dim=10, max_curves=8))
    x = np.asarray(pack(jnp.array(2)))
    np.testing.assert_array_equal(x[0, :, 0], [3, 1, 3, 0, 0, 1])
    np.testing.assert_allclose(x[..., 2:4], np.asarray(hps)[0, np.asarray(idx), :2] * 5)
    np.testing.assert_array_equal(x[..., 4:], 0)
    x_one = np.asarray(pack(jnp.array(1)))
    np.testing.assert_array_equal(x_one[..., 3:], 0)
    np.testing.assert_allclose(x_one[..., 2], np.asarray(hps)[0, np.asarray(idx), 0] * 10)


def test_pack_tokens_scatter_active_values_to_requested_slots():
    hps = jnp.array([[[0.2, 0.7, 0.9], [0.4, 0.6, 0.8]]])
    idx = jnp.array([[0, 1]])
    fidelity = jnp.array([[0.1, 0.2]])
    permutation = jnp.array([[2, 0, 1]])

    packed = pack_tokens(
        hps,
        jnp.int32(2),
        idx,
        fidelity,
        jnp.int32(1),
        config_dim=3,
        max_curves=8,
        slot_permutations=permutation,
    )

    # Source slots 0 and 1 move to destinations 2 and 0; inactive source slot 2 stays zero.
    np.testing.assert_allclose(np.asarray(packed[0, :, 2:]), [[1.05, 0.0, 0.3], [0.9, 0.0, 0.6]])


def test_random_slot_augmentation_preserves_paired_curve_keys():
    key = jax.random.key(19)
    fixed_keys, fixed_permutations = presentation_randomness(key, 3, 7, False)
    random_keys, random_permutations = presentation_randomness(key, 3, 7, True)

    np.testing.assert_array_equal(jax.random.key_data(fixed_keys), jax.random.key_data(random_keys))
    assert fixed_permutations is None
    assert random_permutations.shape == (3, 7)
    for permutation in np.asarray(random_permutations):
        np.testing.assert_array_equal(np.sort(permutation), np.arange(7))


def test_forward_accepts_an_explicit_outcome_independent_slot_presentation():
    hps, idx, fidelity, y = _inputs()
    model = _model(config_dim=3)
    key = jax.random.key(23)
    permutation = jnp.array([[2, 0, 1]])
    actual = model(
        hps,
        2,
        idx,
        fidelity,
        y,
        3,
        feature_key=key,
        slot_permutations=permutation,
    )
    curve_keys, _ = presentation_randomness(key, 1, 3, False)
    offsets = jax.vmap(lambda k: jax.random.randint(k, (), 0, model.max_curves))(curve_keys)
    packed = pack_tokens(
        hps,
        2,
        idx,
        fidelity,
        3,
        config_dim=3,
        max_curves=model.max_curves,
        curve_offsets=offsets,
        slot_permutations=permutation,
    )
    expected = model.forward_tokens(packed, y, 3)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_explicit_slots_cannot_override_training_augmentation():
    hps, idx, fidelity, y = _inputs()
    model = _model(config_dim=3, random_slots=True)
    with pytest.raises(ValueError, match="random_slots=False"):
        model(
            hps,
            2,
            idx,
            fidelity,
            y,
            3,
            feature_key=jax.random.key(23),
            slot_permutations=jnp.array([[2, 0, 1]]),
        )


@pytest.mark.parametrize("remat", [False, True])
def test_forward_and_gradients_match_reference_active_width(remat):
    hps, idx, fidelity, y = _inputs()
    model = _model(remat=remat)
    key = jax.random.key(0)
    curve_keys = jax.vmap(lambda i: jax.random.fold_in(key, i))(jnp.arange(hps.shape[0]))
    offsets = jax.vmap(lambda k: jax.random.randint(k, (), 0, model.max_curves))(curve_keys)
    packed = pack_tokens(
        hps,
        2,
        idx,
        fidelity,
        3,
        config_dim=model.config_dim,
        max_curves=model.max_curves,
        curve_offsets=offsets,
    )
    reference_x = packed[..., :4].at[..., 2:].divide(5.0)
    actual = model(hps, 2, idx, fidelity, y, 3, feature_key=key)
    expected = model.model(reference_x, y, 3)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)
    actual_grads = nnx.grad(lambda m: jnp.mean(m.forward_tokens(packed, y, 3)[:, 3:] ** 2))(model)
    expected_grads = nnx.grad(lambda m: jnp.mean(m(reference_x, y, 3)[:, 3:] ** 2))(model.model)
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual_grads["model"]), jax.tree.leaves(expected_grads), strict=True
    ):
        np.testing.assert_allclose(actual_leaf, expected_leaf, atol=3e-6, rtol=3e-5)


def test_33_hyperparameter_model_uses_widened_projection_and_shifted_curve_rows():
    model = _model(config_dim=33, max_curves=64)
    hps = jnp.linspace(0.0, 1.0, 4 * 33, dtype=jnp.float32).reshape(1, 4, 33)
    idx = jnp.asarray([[0, 1, 2, 3, 0, 2]])
    fidelity = jnp.linspace(0.1, 0.6, 6, dtype=jnp.float32)[None]
    y = jnp.linspace(0.2, 0.7, 6, dtype=jnp.float32)[None]
    logits = model(hps, 33, idx, fidelity, y, 4, feature_key=jax.random.key(9))
    assert logits.shape == (1, 6, 8)
    assert model.model.config_enc.kernel.shape == (33, 16)
    assert model.model.idcurve_enc.embedding.shape == (65, 16)
    assert np.isfinite(logits).all()


def test_queries_cannot_change_other_queries_or_supply_y():
    hps, idx, fidelity, y = _inputs()
    model = _model()
    forward = nnx.jit(lambda m, i, f, yy: m(hps, 2, i, f, yy, 3, feature_key=jax.random.key(0)))
    before = forward(model, idx, fidelity, y)
    after = forward(model, idx.at[:, 3].set(0), fidelity.at[:, 3].set(0.99), y.at[:, 3:].set(99))
    np.testing.assert_allclose(before[:, 4:], after[:, 4:], atol=2e-6, rtol=2e-6)
    perm = jnp.asarray([2, 0, 1, 3, 4, 5])
    permuted = forward(model, idx[:, perm], fidelity[:, perm], y[:, perm])
    np.testing.assert_allclose(before[:, 3:], permuted[:, 3:], atol=2e-6, rtol=2e-6)


def test_bfloat16_compute_keeps_finite_float32_parameter_gradients():
    hps, idx, fidelity, y = _inputs()
    model = _model(dtype=jnp.bfloat16)
    loss, grads = nnx.jit(
        nnx.value_and_grad(
            lambda m: jnp.mean(
                m(hps, 2, idx, fidelity, y, 3, feature_key=jax.random.key(0)).astype(jnp.float32)
                ** 2
            )
        )
    )(model)
    assert np.isfinite(loss)
    for value in jax.tree.leaves(nnx.state(model, nnx.Param)) + jax.tree.leaves(grads):
        assert value.dtype == jnp.float32
        assert np.isfinite(value).all()


def test_training_resume_matches_continuous_state(tmp_path):
    model_cfg = ModelConfig(
        architecture="ftpfn",
        config_dim=3,
        emsize=16,
        num_heads=2,
        hidden_dim=32,
        n_feat_layers=0,
        n_icl_layers=1,
        num_bars=8,
        icl_backend="xla",
        ragged=False,
        qk_norm=False,
        ftpfn_max_curves=16,
        dtype="float32",
    )
    optim_cfg = OptimConfig(peak_lr=1e-4, warmup_steps=1, decay_start=3, decay_steps=1)
    prior_cfg = PriorConfig(layers=(2, 2), hidden=(8, 8), table_networks=8, table_configs=4)
    data_cfg = DataConfig(seq_len=16, batch_size=2, n_steps_max=8)
    run = RunConfig(
        num_steps=4, seed=11, print_every=1, val_every=2, val_batches=1, checkpoint_every=2
    )
    args = (model_cfg, optim_cfg, prior_cfg, data_cfg)
    continuous = train(*args, run, log=lambda _: None)
    resume_run = dataclasses.replace(run, num_steps=2, checkpoint_dir=str(tmp_path / "run/ckpt"))
    train(*args, resume_run, log=lambda _: None)
    resumed = train(*args, dataclasses.replace(resume_run, num_steps=4), log=lambda _: None)
    for a, b in zip(
        jax.tree.leaves(nnx.state(continuous)), jax.tree.leaves(nnx.state(resumed)), strict=True
    ):
        np.testing.assert_array_equal(a, b)


def test_pytorch_initialization():
    model = _model(initialization="pytorch").model
    for layer in model.layers:
        for output in (layer.linear2, layer.attn.out):
            np.testing.assert_array_equal(output.kernel[...], 0)
            np.testing.assert_array_equal(output.bias[...], 0)
        embed = layer.linear1.kernel.shape[0]
        bound = np.sqrt(6 / (4 * embed))
        for projection in (layer.attn.query, layer.attn.key, layer.attn.value):
            assert np.max(np.abs(projection.kernel[...])) <= bound
            np.testing.assert_array_equal(projection.bias[...], 0)
    for layer in (model.epoch_enc, model.config_enc, model.y_enc,
                  model.decoder_linear1, model.decoder_linear2):
        bound = 1 / np.sqrt(layer.kernel.shape[0])
        assert np.max(np.abs(layer.kernel[...])) <= bound
        if layer.bias is not None:
            assert np.max(np.abs(layer.bias[...])) <= bound
            assert np.any(np.asarray(layer.bias[...]) != 0)
    assert abs(float(jnp.std(model.idcurve_enc.embedding[...]))) > 0.7


def test_paper_cosine_schedule():
    from icicle_jax.ftpfn.train import paper_schedule

    schedule = paper_schedule(OptimConfig(peak_lr=1e-4,
                                         warmup_steps=20_000), 80_000)
    np.testing.assert_allclose([schedule(s) for s in (0, 20_000, 50_000, 80_000)],
                               [0, 1e-4, 5e-5, 0], atol=1e-10)


def test_task_horizons_and_empty_context():
    from icicle_jax.loading import build_model
    from icicle_jax.train import build_table, resolve_prior
    from icicle_jax.ftpfn.train import paper_samplers
    from flax import nnx

    model_cfg = ModelConfig(architecture="ftpfn", emsize=8, num_heads=1, hidden_dim=8,
                           n_feat_layers=0, n_icl_layers=1, config_dim=2, num_bars=8,
                           ragged=False, qk_norm=False, icl_backend="xla",
                           ftpfn_initialization="pytorch", dtype="float32")
    prior_cfg = PriorConfig(layers=(2, 2), hidden=(8, 8), table_networks=4, table_configs=4)
    data_cfg = DataConfig(seq_len=8, batch_size=8, n_steps_max=8)
    table = build_table(model_cfg, prior_cfg, log=lambda _: None)
    observation_config = resolve_prior(prior_cfg, log=lambda _: None)
    sample, _ = paper_samplers(model_cfg, prior_cfg, data_cfg, table=table, observation_config=observation_config)
    batch = sample(jax.random.key(0), jnp.int32(2), jnp.int32(8), jnp.int32(1))
    assert len(np.unique(np.asarray(batch.fidelities[:, 0]))) > 1
    empty = sample(jax.random.key(1), jnp.int32(2), jnp.int32(8), jnp.int32(0))
    model = build_model(model_cfg, nnx.Rngs(0))
    logits = model(empty.config_hps, jnp.int32(2), empty.point_config_idx,
                   empty.fidelities, empty.y, 0, feature_key=jax.random.key(0))
    assert np.isfinite(np.asarray(logits)).all()
