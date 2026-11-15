"""Compare the trainable wrapper with the pinned upstream PyTorch implementation.

Set ICICLE_IFBO_SOURCE to the prepared source directory produced by
reproduction.prepare.prepare_ftpfn_reference. No upstream code is copied here.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from icicle_jax.ftpfn import precise_attention_fn
from icicle_jax.ftpfn.weights import build_ftpfn
from icicle_jax.ftpfn.train import TrainableFTPFN, pack_tokens


@pytest.mark.parametrize("config_dim,remat,double", [
    (10, False, True), (10, True, True), (33, True, True), (10, True, False),
])
def test_training_logits_and_gradients_match_pytorch(monkeypatch, config_dim, remat, double):
    source = os.environ.get("ICICLE_IFBO_SOURCE")
    if not source:
        pytest.skip("set ICICLE_IFBO_SOURCE to the prepared upstream source")
    monkeypatch.syspath_prepend(source)
    import torch
    from ifbo.encoders import Normalize, Linear
    from ifbo.priors.ftpfn_prior import MultiCurvesEncoder
    from ifbo.transformer import TransformerModel

    jax.config.update("jax_enable_x64", True)
    dtype = jnp.float64 if double else jnp.float32
    torch_dtype = torch.float64 if double else torch.float32
    atol, rtol = (1e-8, 1e-7) if double else (2e-6, 1e-4)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    reference = TransformerModel(
        MultiCurvesEncoder(config_dim + 2, 16), ninp=16, nhead=2, nhid=32,
        nlayers=2, dropout=0.0,
        y_encoder=torch.nn.Sequential(Normalize(0.5, np.sqrt(1 / 12)), Linear(1, 16)),
        decoder_dict={"standard": (None, 8)},
    ).to(torch_dtype).train()
    # Exercise attention and FFN gradients beyond the source's zero residual initialization.
    with torch.no_grad():
        for layer in reference.transformer_encoder.layers:
            layer.linear2.weight.normal_(std=0.05)
            layer.self_attn.out_proj.weight.normal_(std=0.05)
    state = {k: v.detach().numpy().copy() for k, v in reference.state_dict().items()}
    model = TrainableFTPFN(
        emsize=16, num_heads=2, hidden_dim=32, num_layers=2, num_bars=8,
        config_dim=config_dim, max_curves=1000, dtype=None, remat=remat,
        random_slots=False, rngs=nnx.Rngs(0),
    )
    model.model = build_ftpfn(state, num_heads=2, param_dtype=dtype,
                              attention_fn=precise_attention_fn if double else None)
    hps = jnp.asarray(np.random.default_rng(0).uniform(size=(2, 3, config_dim)), dtype=dtype)
    indices = jnp.array([[0, 1, 0, 2, 1, 2], [1, 0, 1, 2, 0, 2]])
    fidelity = jnp.broadcast_to(jnp.array([.2, .3, .5, .7, .8, 1.], dtype=dtype), (2, 6))
    packed = pack_tokens(hps, config_dim, indices, fidelity, 3,
                         config_dim=config_dim, max_curves=1000,
                         curve_offsets=jnp.array([0, 997]))
    y = jnp.asarray(np.random.default_rng(1).uniform(size=(2, 6)), dtype=dtype)
    x_torch = torch.tensor(np.asarray(packed).transpose(1, 0, 2), dtype=torch_dtype)
    y_torch = torch.tensor(np.asarray(y[:, :3]).T, dtype=torch_dtype)
    expected = reference((x_torch, y_torch), single_eval_pos=3)
    torch_loss = expected.square().mean()
    torch_loss.backward()

    actual = model.forward_tokens(packed, y, 3)[:, 3:]
    np.testing.assert_allclose(actual, expected.detach().numpy().transpose(1, 0, 2),
                               rtol=rtol, atol=atol)
    loss, grads = nnx.value_and_grad(
        lambda m: jnp.mean(m.forward_tokens(packed, y, 3)[:, 3:] ** 2)
    )(model)
    np.testing.assert_allclose(loss, torch_loss.detach().numpy(), rtol=rtol, atol=atol)
    torch_grads = {name: p.grad.numpy() for name, p in reference.named_parameters()}
    mapped_grads = build_ftpfn(torch_grads, num_heads=2, param_dtype=dtype)
    for actual, expected in zip(jax.tree.leaves(grads["model"]),
                                jax.tree.leaves(nnx.state(mapped_grads, nnx.Param)), strict=True):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
