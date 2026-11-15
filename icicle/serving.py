"""Load an ONNX model and predict from normalized curve observations.

The feature frame is fixed for each OnnxSurrogate instance. Each query returns
one probability per value bucket; self.borders gives the bucket edges.
"""

import json
from pathlib import Path

import numpy as np


def orthonormal_frame(rng, rows: int, dim: int) -> np.ndarray:
    """Draw a Haar-uniform orthonormal frame scaled to the sincos norm.

    QR of a Gaussian with the Mezzadri sign fix, rows scaled to ``sqrt(dim / 2)``. Needs
    ``rows <= dim``. Slicing leading rows of a larger frame stays orthonormal, so one frame
    drawn at the maximum HP count serves every call width.
    """
    if rows > dim:
        raise ValueError(f"frame needs rows {rows} <= dim {dim}")
    g = rng.standard_normal((rows, dim)).astype(np.float32)
    q, r = np.linalg.qr(g.T)
    return ((q * np.sign(np.diag(r))).T * np.sqrt(dim / 2.0)).astype(np.float32)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def make_session(model_path, providers=None, *, intra_op_num_threads=None):
    """Create an ONNX Runtime session with explicit provider and precision settings.

    Disable CUDA TF32 to avoid reducing matrix-multiplication precision.
    providers accepts names or (name, options) pairs. Explicit options override
    our defaults. None or an empty list uses ONNX Runtime defaults. Raise if
    the first requested provider cannot be selected.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    if intra_op_num_threads is not None:
        if intra_op_num_threads < 1:
            raise ValueError("intra_op_num_threads must be positive or None")
        options.intra_op_num_threads = intra_op_num_threads
    defaults = {
        "CUDAExecutionProvider": {"use_tf32": "0"},
        # Enable ROCm's tunable attention operator for fused graphs.
        "ROCMExecutionProvider": {"tunable_op_enable": "1", "tunable_op_tuning_enable": "1"},
    }
    requested = []
    configured = None
    if providers:
        configured = []
        for provider in providers:
            name, overrides = (provider, {}) if isinstance(provider, str) else provider
            requested.append(name)
            configured.append((name, defaults.get(name, {}) | overrides))
    session = ort.InferenceSession(str(model_path), sess_options=options, providers=configured)
    if requested and session.get_providers()[0] != requested[0]:
        raise RuntimeError(
            f"{requested[0]} unavailable; onnxruntime fell back to {session.get_providers()}"
        )
    return session



class OnnxSurrogate:
    """Predict learning curves with ONNX Runtime.

    With no model_path, load the model bundled with ICICLE.
    Metadata is read from the JSON file with the same stem as model_path.
    config_dim sets the maximum HP count. Supply frame to reproduce a particular
    feature encoding, or seed to draw one. providers selects the runtime backend;
    intra_op_num_threads limits its CPU threads.
    """

    def __init__(
        self,
        model_path=None,
        *,
        config_dim=None,
        seed=0,
        providers=None,
        frame=None,
        intra_op_num_threads=None,
    ):
        model_path = (
            Path(model_path) if model_path is not None
            else Path(__file__).parent / "models" / "icicle.onnx"
        )
        model_path = model_path.resolve()
        m = json.loads(model_path.with_suffix(".json").read_text())["model"]
        if m.get("curve_identity", "none") != "none":
            raise ValueError("ICICLE exports with curve identities are not supported")
        self.emsize = m["emsize"]
        self.num_bars = m["num_bars"]
        self.borders = np.linspace(0.0, 1.0, self.num_bars + 1)
        max_hyperparameters = m["config_dim"] if config_dim is None else config_dim
        if max_hyperparameters < 1:
            raise ValueError("config_dim must be positive")
        # Load immediately; recreate the unpicklable session after Ray restores this object.
        self._model_path = str(model_path)
        self._providers = providers
        self._intra_op_num_threads = intra_op_num_threads
        self._session = make_session(
            self._model_path, providers, intra_op_num_threads=intra_op_num_threads
        )
        rng = np.random.default_rng(seed)
        if m.get("feature_identity_kind", "haar") != "haar":
            raise ValueError("unsupported feature identity distribution")
        if frame is None:
            self._frame = orthonormal_frame(rng, max_hyperparameters, self.emsize)
        else:
            self._frame = np.asarray(frame, np.float32)
            if (
                self._frame.ndim != 2
                or self._frame.shape[0] < max_hyperparameters
                or self._frame.shape[1] != self.emsize
            ):
                raise ValueError(
                    f"frame shape {self._frame.shape} must have at least {max_hyperparameters} rows "
                    f"and exactly {self.emsize} columns"
                )

    @property
    def session(self):
        if self._session is None:
            self._session = make_session(
                self._model_path,
                self._providers,
                intra_op_num_threads=self._intra_op_num_threads,
            )
        return self._session

    def __getstate__(self):
        return self.__dict__ | {"_session": None}

    def __call__(self, ctx_hps, ctx_fid, ctx_y, qry_hps, qry_fid):
        ctx_hps = np.atleast_2d(np.asarray(ctx_hps, np.float32))
        qry_hps = np.atleast_2d(np.asarray(qry_hps, np.float32))
        n_ctx, h = ctx_hps.shape
        if n_ctx < 1:
            raise ValueError("the surrogate needs at least one context observation")
        if h > len(self._frame):
            raise ValueError(f"HP count {h} exceeds the frame width {len(self._frame)}")
        # distinct-config table over context + queries; points reference rows by index.
        # np.unique sorts rows, so the packing is deterministic for a given set of configs.
        all_hps = np.concatenate([ctx_hps, qry_hps], axis=0)
        config_hps, inverse = np.unique(all_hps, axis=0, return_inverse=True)
        q = len(qry_hps)
        feeds = {
            "config_hps": config_hps,
            "feature_identity": self._frame[:h],
            "num_params": np.array(h, np.int64),
            "point_config_idx": inverse.astype(np.int64),
            "fidelities": np.concatenate(
                [np.asarray(ctx_fid, np.float32), np.asarray(qry_fid, np.float32)]
            ),
            "y": np.concatenate(
                [np.asarray(ctx_y, np.float32), np.zeros(q, np.float32)]  # query y is masked
            ),
            "single_eval_pos": np.array(n_ctx, np.int64),
        }
        (logits,) = self.session.run(["logits"], feeds)
        return _softmax(logits[n_ctx : n_ctx + q])
