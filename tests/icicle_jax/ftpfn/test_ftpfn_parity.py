"""Compare FT-PFN logits with the published PyTorch checkpoint on a fixed input.

Run on CPU in float64 with the precise attention implementation. Skip if the
converted checkpoint is unavailable.
"""

import os
from pathlib import Path

import numpy as np
import pytest

ARTIFACT = Path(os.environ.get("ICICLE_FTPFN_WEIGHTS", "checkpoints/ftpfn_v0.0.1.safetensors"))
FIXTURE = Path(__file__).parent / "golden" / "reference_io.npz"


@pytest.mark.skipif(not ARTIFACT.exists(), reason="converted FT-PFN checkpoint not on disk")
def test_ftpfn_matches_published_logits():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from icicle_jax.ftpfn import load_ftpfn, precise_attention_fn

    ref = np.load(FIXTURE)
    model = load_ftpfn(ARTIFACT, param_dtype=jnp.float64, attention_fn=precise_attention_fn)
    # The reference layout is torch's (seq, batch, feat); the model is batch-first.
    x = jnp.asarray(ref["x"], jnp.float64).transpose(1, 0, 2)
    sep = int(ref["single_eval_pos"])
    # Reference y covers only the context prefix; pad the query positions with
    # NaN (the model must ignore them for parity to hold).
    y = np.full((x.shape[0], x.shape[1]), np.nan)
    y[:, :sep] = ref["y"].T
    logits = model(x, jnp.asarray(y), sep)[:, sep:]
    np.testing.assert_allclose(
        np.asarray(logits), ref["logits"].transpose(1, 0, 2), atol=1e-8, rtol=1e-8
    )
