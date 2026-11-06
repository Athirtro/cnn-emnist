"""CNN from scratch (NumPy only) + simulated distributed training.

Technique replicated from the reference notebook:
- Activation registry (dataclass with fn/deriv as functions of output A)
  and initializers (small/xavier/he) with default_init per activation.
- Params as list per layer; forward returns activations/caches;
  back_prop returns grads; softmax+cross-entropy with dZ = A - Y.
- Plain SGD (no momentum/state), gradients normalised by batch size
  (``normalize=True``); workers can request unnormalised SUM grads.
- Explicit seed and separate NumPy generators for init and shuffling.

No autograd / torch / cupy is used in the model. torchvision is used
only in :func:`load_emnist` to download EMNIST once, then converted
to NumPy.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

__all__ = [
    "Activation", "ACTIVATIONS", "INITIALIZERS",
    "Conv2D", "MaxPool2D", "Flatten", "Dense", "CNN",
    "WorkerMessage", "Worker", "Aggregator", "Trainer",
    "load_emnist", "run_experiment", "plot_history",
    "flatten_params", "unflatten_params", "deep_copy_params",
    "im2col", "col2im",
]

# ---------------------------------------------------------------- registries
# Initializers: (n_in, n_out, rng) -> array shape (n_out, n_in), float32
INITIALIZERS: dict[str, Callable[..., np.ndarray]] = {
    "small":  lambda n_in, n_out, rng: (rng.standard_normal((n_out, n_in)) * 0.01).astype(np.float32),
    "xavier": lambda n_in, n_out, rng: (rng.standard_normal((n_out, n_in)) * (1 / n_in) ** 0.5).astype(np.float32),
    "he":     lambda n_in, n_out, rng: (rng.standard_normal((n_out, n_in)) * (2 / n_in) ** 0.5).astype(np.float32),
}


@dataclass(frozen=True)
class Activation:
    fn: Callable[[np.ndarray], np.ndarray]      # Z -> A
    deriv: Callable[[np.ndarray], np.ndarray]   # A -> dA/dZ (as function of A)
    default_init: str = "xavier"


ACTIVATIONS: dict[str, Activation] = {
    "sigmoid": Activation(
        fn=lambda Z: 1 / (1 + np.exp(-Z)),
        deriv=lambda A: A * (1 - A),
        default_init="xavier",
    ),
    "tanh": Activation(
        fn=np.tanh,
        deriv=lambda A: 1 - A * A,
        default_init="xavier",
    ),
    "relu": Activation(
        fn=lambda Z: np.maximum(Z, 0),
        deriv=lambda A: (A > 0).astype(A.dtype),
        default_init="he",
    ),
    "identity": Activation(fn=lambda Z: Z, deriv=lambda A: np.ones_like(A), default_init="xavier"),
}

# ---------------------------------------------------------------- im2col
def _out_size(size: int, k: int, stride: int, padding: int) -> int:
    return (size + 2 * padding - k) // stride + 1


def im2col(X: np.ndarray, KH: int, KW: int, stride: int, padding: int) -> np.ndarray:
    """(N,C,H,W) -> (N*Hout*Wout, C*KH*KW). Vectorised; loop only over kernel."""
    N, C, H, W = X.shape
    Hout, Wout = _out_size(H, KH, stride, padding), _out_size(W, KW, stride, padding)
    Xp = np.pad(X, ((0, 0), (0, 0), (padding, padding), (padding, padding))) if padding else X
    cols = np.empty((N, C, KH, KW, Hout, Wout), dtype=X.dtype)
    for i in range(KH):
        for j in range(KW):
            cols[:, :, i, j, :, :] = Xp[:, :, i:i + stride * Hout:stride, j:j + stride * Wout:stride]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * Hout * Wout, C * KH * KW)


def col2im(dcols: np.ndarray, X_shape: tuple[int, ...], KH: int, KW: int,
           stride: int, padding: int) -> np.ndarray:
    N, C, H, W = X_shape
    Hout, Wout = _out_size(H, KH, stride, padding), _out_size(W, KW, stride, padding)
    dXp = np.zeros((N, C, H + 2 * padding, W + 2 * padding), dtype=dcols.dtype)
    dcols6 = dcols.reshape(N, Hout, Wout, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)
    for i in range(KH):
        for j in range(KW):
            dXp[:, :, i:i + stride * Hout:stride, j:j + stride * Wout:stride] += dcols6[:, :, i, j, :, :]
    return dXp[:, :, padding:padding + H, padding:padding + W] if padding else dXp


# ---------------------------------------------------------------- layers
class Conv2D:
    """Cross-correlation conv, W:(F,C,KH,KW), b:(F,)."""

    def __init__(self, out_channels: int, in_channels: int, kernel_size: int,
                 stride: int = 1, padding: int = 0, initializer: Callable | str = "xavier",
                 rng: np.random.Generator | None = None, dtype=np.float32):
        self.F, self.C, self.K = out_channels, in_channels, kernel_size
        self.stride, self.padding, self.dtype = stride, padding, dtype
        rng = rng or np.random.default_rng(0)
        init = INITIALIZERS[initializer] if isinstance(initializer, str) else initializer
        n_in = in_channels * kernel_size * kernel_size
        W = init(n_in, out_channels, rng).reshape(out_channels, in_channels, kernel_size, kernel_size)
        self.params = (W.astype(dtype), np.zeros((out_channels,), dtype=dtype))
        self.out_shape: tuple[int, int, int] | None = None

    def forward(self, X: np.ndarray):
        W, b = self.params
        N, _, H, Wd = X.shape
        Hout, Wout = _out_size(H, self.K, self.stride, self.padding), _out_size(Wd, self.K, self.stride, self.padding)
        cols = im2col(X, self.K, self.K, self.stride, self.padding)          # (NHW, CKK)
        out = cols @ W.reshape(self.F, -1).T + b                              # (NHW, F)
        A = out.reshape(N, Hout, Wout, self.F).transpose(0, 3, 1, 2).astype(self.dtype)
        self.out_shape = (self.F, Hout, Wout)
        return A, (X, cols)

    def backward(self, dA: np.ndarray, cache):
        X, cols = cache
        W, _ = self.params
        N = X.shape[0]
        Hout, Wout = dA.shape[2], dA.shape[3]
        dA_cols = dA.transpose(0, 2, 3, 1).reshape(N * Hout * Wout, self.F)   # (NHW, F)
        dW = (dA_cols.T @ cols).reshape(self.F, self.C, self.K, self.K)
        db = dA_cols.sum(axis=0)
        dcols = dA_cols @ W.reshape(self.F, -1)                               # (NHW, CKK)
        dX = col2im(dcols, X.shape, self.K, self.K, self.stride, self.padding)
        return dX.astype(self.dtype), (dW.astype(self.dtype), db.astype(self.dtype))


class MaxPool2D:
    params = None

    def __init__(self, pool_size: int = 2, stride: int | None = None):
        self.P = pool_size
        self.S = stride if stride is not None else pool_size

    def forward(self, X: np.ndarray):
        N, C, H, W = X.shape
        Hout, Wout = _out_size(H, self.P, self.S, 0), _out_size(W, self.P, self.S, 0)
        out = np.full((N, C, Hout, Wout), -np.inf, dtype=X.dtype)
        argmax = np.empty((N, C, Hout, Wout), dtype=np.int64)
        for i in range(self.P):
            for j in range(self.P):
                patch = X[:, :, i:i + self.S * Hout:self.S, j:j + self.S * Wout:self.S]
                mask = patch > out
                # offset in flattened pool window for argmax routing
                offset = i * self.P + j
                argmax = np.where(mask, offset, argmax)
                out = np.where(mask, patch, out)
        return out, (X.shape, argmax)

    def backward(self, dA: np.ndarray, cache):
        X_shape, argmax = cache
        N, C, Hout, Wout = dA.shape
        dX = np.zeros(X_shape, dtype=dA.dtype)
        for i in range(self.P):
            for j in range(self.P):
                offset = i * self.P + j
                sel = (argmax == offset)
                dX[:, :, i:i + self.S * Hout:self.S, j:j + self.S * Wout:self.S] += np.where(sel, dA, 0)
        return dX, None


class Flatten:
    params = None

    def forward(self, X: np.ndarray):
        return X.reshape(X.shape[0], -1), X.shape

    def backward(self, dA: np.ndarray, cache):
        return dA.reshape(cache), None


class Dense:
    """W:(out,in), b:(out,). Input may be (N,in)."""

    def __init__(self, in_features: int, out_features: int,
                 initializer: Callable | str = "xavier",
                 rng: np.random.Generator | None = None, dtype=np.float32):
        rng = rng or np.random.default_rng(0)
        init = INITIALIZERS[initializer] if isinstance(initializer, str) else initializer
        self.params = (init(in_features, out_features, rng).astype(dtype),
                       np.zeros((out_features,), dtype=dtype))
        self.out = out_features

    def forward(self, X: np.ndarray):
        W, b = self.params
        return X @ W.T + b, X

    def backward(self, dA: np.ndarray, cache):
        X = cache
        W, _ = self.params
        dW = dA.T @ X
        db = dA.sum(axis=0)
        return dA @ W, (dW, db)


# ---------------------------------------------------------------- param utils
def deep_copy_params(params):
    return copy.deepcopy(params)


def flatten_params(params) -> np.ndarray:
    return np.concatenate([p.ravel() for layer in params if layer is not None
                           for p in layer]).astype(np.float32) if any(p is not None for p in params) else np.zeros(0, np.float32)


def unflatten_params(vec: np.ndarray, template) -> list:
    out, off = [], 0
    for layer in template:
        if layer is None:
            out.append(None)
            continue
        new_layer = []
        for arr in layer:
            n = arr.size
            new_layer.append(vec[off:off + n].reshape(arr.shape).copy())
            off += n
        out.append(tuple(new_layer))
    return out


# ---------------------------------------------------------------- CNN
class CNN:
    """Conv(k,filters,stride,pad)->act->MaxPool repeated; Flatten->Dense*->Softmax.

    conv_blocks: sequence of (n_filters, kernel, stride, padding, pool_size|None).
    Base case: ((8, 3, 1, 1, 2),) with input (1,28,28) -> Dense 47.
    """

    def __init__(self, input_shape=(1, 28, 28), conv_blocks=((8, 3, 1, 1, 2),),
                 hidden_sizes=(), output_size: int = 47, activation="relu",
                 initializer: str | None = None, rng: np.random.Generator | None = None,
                 dtype=np.float32):
        self.dtype, self.output_size = dtype, output_size
        self.activation = ACTIVATIONS[activation] if isinstance(activation, str) else activation
        if initializer is None:
            initializer = self.activation.default_init
        self.initializer = initializer
        init_rng = rng if rng is not None else np.random.default_rng(0)

        self.layers: list = []
        self._apply_activation: list[bool] = []  # per layer: apply activation after it
        shape: tuple | None = tuple(input_shape)  # (C,H,W) then (D,)
        for (filters, k, stride, pad, pool) in conv_blocks:
            conv = Conv2D(filters, shape[0], k, stride, pad, initializer, init_rng, dtype)
            self.layers.append(conv); self._apply_activation.append(True)
            _, _, h, w = conv.forward(np.zeros((1, *shape), dtype=dtype))[0].shape
            shape = (filters, h, w)
            if pool:
                self.layers.append(MaxPool2D(pool)); self._apply_activation.append(False)
                h, w = _out_size(h, pool, pool, 0), _out_size(w, pool, pool, 0)
                shape = (filters, h, w)
        self.layers.append(Flatten()); self._apply_activation.append(False)
        in_f = int(np.prod(shape))
        for hidden in hidden_sizes:
            self.layers.append(Dense(in_f, hidden, initializer, init_rng, dtype))
            self._apply_activation.append(True)
            in_f = hidden
        self.layers.append(Dense(in_f, output_size, initializer, init_rng, dtype))
        self._apply_activation.append(False)  # softmax handled separately
        self.params: list = [getattr(l, "params", None) for l in self.layers]

    # -- sync params <-> layers
    def _set_params(self, params):
        self.params = deep_copy_params(params)
        for layer, p in zip(self.layers, self.params):
            if p is not None:
                layer.params = p

    def get_params_vector(self) -> np.ndarray:
        return flatten_params(self.params)

    def set_params_vector(self, vec: np.ndarray):
        self._set_params(unflatten_params(np.asarray(vec, dtype=self.dtype), self.params))

    @staticmethod
    def _softmax(Z: np.ndarray) -> np.ndarray:
        e = np.exp(Z - Z.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def _one_hot(Y: np.ndarray, n: int) -> np.ndarray:
        oh = np.zeros((Y.size, n), dtype=np.float32)
        oh[np.arange(Y.size), Y] = 1
        return oh

    def forward(self, X):
        X = np.asarray(X, dtype=self.dtype)
        caches, A = [], X
        for layer, apply_act in zip(self.layers, self._apply_activation):
            Z, cache = layer.forward(A)
            # softmax is only after the final Dense; activation flag is False there
            A_out = self.activation.fn(Z) if apply_act else Z
            caches.append((cache, apply_act, A_out))
            A = A_out
        return self._softmax(A), caches

    _forward_prop = forward

    def back_prop(self, caches, Y, normalize: bool = True):
        """Returns grads list per layer (None where no params). dZ=A-Y at softmax."""
        Y = np.asarray(Y)
        m = Y.size
        A = caches[-1][2]
        probs = self._softmax(A)
        dA = probs - self._one_hot(Y, self.output_size)  # dZ wrt logits
        grads: list = [None] * len(self.layers)
        for idx in reversed(range(len(self.layers))):
            layer = self.layers[idx]
            cache, apply_act, A_out = caches[idx]
            # For layers with activation, dA is wrt A_out; convert to wrt Z.
            # For final layer apply_act=False so dA is already wrt Z (logits).
            dZ = dA * self.activation.deriv(A_out) if apply_act else dA
            dX, grad = layer.backward(dZ, cache)
            grads[idx] = grad
            dA = dX
        if normalize and m > 0:
            grads = [None if g is None else tuple((a / m).astype(self.dtype) for a in g) for g in grads]
        return grads

    _back_prop = back_prop

    def _update_params(self, grads, alpha: float):
        new = [None if p is None else tuple((a - alpha * b).astype(self.dtype) for a, b in zip(p, g))
               for p, g in zip(self.params, grads)]
        self._set_params(new)

    update_params = _update_params

    def loss(self, X, Y) -> float:
        probs, _ = self.forward(X)
        p = np.clip(probs[np.arange(len(Y)), np.asarray(Y)], 1e-12, 1)
        return float(-np.mean(np.log(p)))

    def predict(self, X, batch_size: int = 512):
        X = np.asarray(X)
        preds = [np.argmax(self.forward(X[i:i + batch_size])[0], axis=1)
                 for i in range(0, len(X), batch_size)]
        return np.concatenate(preds) if preds else np.zeros(0, dtype=int)

    def evaluate(self, X, Y) -> float:
        return float(np.mean(self.predict(X) == np.asarray(Y)))

    def fit(self, X, Y, epochs: int = 1, alpha: float = 0.1, batch_size: int = 128,
            shuffle_rng: np.random.Generator | None = None, test_data=None) -> dict:
        """Single-process SGD. history per epoch: train/test accuracy."""
        X, Y = np.asarray(X, dtype=self.dtype), np.asarray(Y)
        rng = shuffle_rng or np.random.default_rng(0)
        hist = {"train_acc": [], "test_acc": [], "steps": 0}
        for _ in range(epochs):
            idx = rng.permutation(len(Y))
            for s in range(0, len(Y), batch_size):
                b = idx[s:s + batch_size]
                _, caches = self.forward(X[b])
                self._update_params(self.back_prop(caches, Y[b], normalize=True), alpha)
                hist["steps"] += 1
            hist["train_acc"].append(self.evaluate(X, Y))
            if test_data is not None:
                hist["test_acc"].append(self.evaluate(test_data[0], test_data[1]))
        return hist


# ---------------------------------------------------------------- distributed
@dataclass
class WorkerMessage:
    worker_id: int
    step: int
    grads: list          # accumulated SUM (unnormalised), same structure as params
    n_samples: int


class Worker:
    """Independent deep copy of params + own shard; accumulates SUM grads."""

    def __init__(self, worker_id: int, model: CNN):
        self.worker_id = worker_id
        self.model = copy.deepcopy(model)  # independent memory
        self.grads_acc: list | None = None
        self.n_samples = 0
        self.step = 0

    def reset(self):
        self.grads_acc = None
        self.n_samples = 0

    def compute(self, micro_batches, step: int | None = None) -> WorkerMessage:
        """micro_batches: iterable of (Xb, Yb). Accumulates SUM grads."""
        self.reset()
        step = self.step if step is None else step
        for Xb, Yb in micro_batches:
            Yb = np.asarray(Yb)
            if Yb.size == 0:
                continue
            _, caches = self.model.forward(Xb)
            g = self.model.back_prop(caches, Yb, normalize=False)  # SUM
            if self.grads_acc is None:
                self.grads_acc = [None if gg is None else tuple(a.copy() for a in gg) for gg in g]
            else:
                for acc, gg in zip(self.grads_acc, g):
                    if acc is not None:
                        for i in range(len(acc)):
                            acc[i] += gg[i]
            self.n_samples += int(Yb.size)
        msg = WorkerMessage(self.worker_id, step, self.grads_acc, self.n_samples)
        self.step += 1
        return msg

    def apply(self, global_grads, alpha: float):
        self.model._update_params(global_grads, alpha)


class Aggregator:
    """Weighted average: sum(grads) / total_samples (not simple mean)."""

    @staticmethod
    def aggregate(messages: list[WorkerMessage]) -> list:
        total = sum(m.n_samples for m in messages)
        if total == 0:
            raise ValueError("no samples in messages")
        if messages[0].grads is None:
            raise ValueError("empty grads")
        out = []
        for li, layer_grads in enumerate(messages[0].grads):
            if layer_grads is None:
                out.append(None)
                continue
            summed = tuple(sum(m.grads[li][k] for m in messages if m.n_samples > 0) for k in range(len(layer_grads)))
            out.append(tuple((s / total).astype(np.float32) for s in summed))
        return out

    aggregate_grads = aggregate


@dataclass
class Trainer:
    model_template: CNN
    train_x: np.ndarray
    train_y: np.ndarray
    n_workers: int = 1
    seed: int = 0
    global_batch_size: int | None = 128
    local_batch_size: int | None = None
    accumulate_steps: int = 1
    shard_proportions: list[float] | None = None
    history: dict = field(default_factory=dict)

    def __post_init__(self):
        self.train_x = np.asarray(self.train_x, dtype=np.float32)
        self.train_y = np.asarray(self.train_y)
        self.shuffle_rng = np.random.default_rng(self.seed + 1)
        self.workers = [Worker(i, self.model_template) for i in range(self.n_workers)]
        self.syncs = 0

    @property
    def local_bs(self) -> int:
        if self.local_batch_size is not None:
            return self.local_batch_size
        g = self.global_batch_size or 128
        return max(1, g // self.n_workers)

    def _shards(self, idx: np.ndarray) -> list[np.ndarray]:
        """Reparto equitativo por defecto: tamaños difieren como máximo en 1.

        ``np.array_split`` reparte las muestras barajadas (sin solapamiento)
        en ``n_workers`` porciones de igual tamaño (±1 si N no es divisible).
        Solo si se pasa ``shard_proportions`` se usa un reparto desigual.
        """
        if self.shard_proportions is None:
            return list(np.array_split(idx, self.n_workers))
        assert len(self.shard_proportions) == self.n_workers
        sizes = [int(round(p * len(idx))) for p in self.shard_proportions[:-1]]
        sizes.append(len(idx) - sum(sizes))
        shards, off = [], 0
        for s in sizes:
            shards.append(idx[off:off + s]); off += s
        return shards

    def train(self, epochs: int = 1, alpha: float = 0.1, test_data=None) -> dict:
        hist = {"train_acc": [], "test_acc": [], "samples_per_worker": [],
                "samples_per_step": [], "bytes_per_worker": [], "syncs": 0,
                "compute_time_per_worker": [], "aggregation_time": [], "steps": 0}
        nbytes = int(self.workers[0].model.get_params_vector().nbytes)
        for _ in range(epochs):
            idx = self.shuffle_rng.permutation(len(self.train_y))
            shards = self._shards(idx)
            offsets = [0] * self.n_workers
            epoch_samples = [0] * self.n_workers
            while any(offsets[i] < len(shards[i]) for i in range(self.n_workers)):
                messages, agg_t0 = [], time.perf_counter()
                compute_times = []
                for wi, worker in enumerate(self.workers):
                    take = self.local_bs * self.accumulate_steps
                    sel = shards[wi][offsets[wi]:offsets[wi] + take]
                    offsets[wi] += len(sel)
                    # split into micro-batches of size local_bs
                    micros = [(self.train_x[sel[i:i + self.local_bs]],
                               self.train_y[sel[i:i + self.local_bs]])
                              for i in range(0, len(sel), self.local_bs)]
                    t0 = time.perf_counter()
                    msg = worker.compute(micros)
                    compute_times.append(time.perf_counter() - t0)
                    messages.append(msg)
                    epoch_samples[wi] += msg.n_samples
                agg_t0 = time.perf_counter()
                global_grads = Aggregator.aggregate(messages)
                hist["aggregation_time"].append(time.perf_counter() - agg_t0)
                hist["compute_time_per_worker"].append(compute_times)
                hist["samples_per_step"].append([m.n_samples for m in messages])
                hist["bytes_per_worker"].append(nbytes)
                for worker in self.workers:
                    worker.apply(global_grads, alpha)
                self.syncs += 1
                hist["steps"] += 1
            hist["samples_per_worker"].append(epoch_samples)
            hist["train_acc"].append(self.workers[0].model.evaluate(self.train_x, self.train_y))
            if test_data is not None:
                hist["test_acc"].append(self.workers[0].model.evaluate(test_data[0], test_data[1]))
        hist["syncs"] = self.syncs
        self.history = hist
        return hist

    @property
    def model(self) -> CNN:
        return self.workers[0].model


# ---------------------------------------------------------------- data / experiments
def load_emnist(root: str = "./data", fraction: float = 1.0, download: bool = True):
    """torchvision only here: download EMNIST, convert once to NumPy (N,1,28,28) in [0,1]."""
    import torchvision  # local import: not used by the model

    train_ds = torchvision.datasets.EMNIST(root=root, split="balanced", train=True, download=download)
    test_ds = torchvision.datasets.EMNIST(root=root, split="balanced", train=False, download=download)
    def conv(ds):
        x = ds.data.numpy().transpose(0, 2, 1).reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
        return x, ds.targets.numpy().astype(np.int64)
    Xtr, Ytr = conv(train_ds)
    Xte, Yte = conv(test_ds)
    if fraction < 1.0:
        n = int(len(Ytr) * fraction)
        Xtr, Ytr = Xtr[:n], Ytr[:n]
    return (Xtr, Ytr), (Xte, Yte)


def plot_history(history_data, title: str = "Historial de Entrenamiento"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    if isinstance(history_data, dict):
        for label, h in history_data.items():
            if h:
                ax.plot(range(1, len(h) + 1), h, label=str(label))
    elif isinstance(history_data, (list, tuple)):
        ax.plot(range(1, len(history_data) + 1), history_data, label="Precisión vs Iteraciones")
    ax.set_xlabel("Épocas"); ax.set_ylabel("Precisión (Accuracy)")
    ax.set_ylim(0, 1); ax.set_title(title); ax.grid(True, linestyle="--", alpha=0.6); ax.legend()
    return fig


def run_experiment(title, train_data, test_data, *, epochs: int = 2, alpha: float = 0.1,
                   seed: int = 0, n_workers: int = 1, global_batch_size: int = 128,
                   local_batch_size: int | None = None, accumulate_steps: int = 1,
                   shard_proportions=None, **model_kwargs):
    """Train, evaluate, return (model, texto, figura, historial).

    Por defecto (``shard_proportions=None``) el dataset se reparte de forma
    equitativa entre los workers (ver :meth:`Trainer._shards`).
    """
    (Xtr, Ytr), (Xte, Yte) = train_data, test_data
    init_rng = np.random.default_rng(seed)          # init generator
    model = CNN(rng=init_rng, **model_kwargs)
    if n_workers == 1 and shard_proportions is None:
        trainer = Trainer(model, Xtr, Ytr, n_workers=1, seed=seed,
                          global_batch_size=global_batch_size,
                          local_batch_size=local_batch_size, accumulate_steps=accumulate_steps)
    else:
        trainer = Trainer(model, Xtr, Ytr, n_workers=n_workers, seed=seed,
                          global_batch_size=global_batch_size,
                          local_batch_size=local_batch_size, accumulate_steps=accumulate_steps,
                          shard_proportions=shard_proportions)
    hist = trainer.train(epochs=epochs, alpha=alpha, test_data=(Xte, Yte))
    acc = trainer.model.evaluate(Xte, Yte)
    n = len(Yte)
    text = (f"**{title}**\n\n- Precisión de prueba: {acc:.4f} ({round(acc*n)}/{n})\n"
            f"- Precisión de entrenamiento (última época): {hist['train_acc'][-1]:.4f}\n"
            f"- Workers: {n_workers} | Sincronizaciones: {hist['syncs']} | "
            f"Bytes de gradiente por worker: {hist['bytes_per_worker'][0] if hist['bytes_per_worker'] else 0}\n"
            f"- Muestras por worker (última época): {hist['samples_per_worker'][-1] if hist['samples_per_worker'] else []}")
    fig = plot_history({"Entrenamiento": hist["train_acc"], "Prueba": hist["test_acc"]}, title=title)
    return trainer.model, text, fig, hist
