"""Verificaciones obligatorias: gradient check, equivalencia, sincronía, N=1."""
import copy
import numpy as np
import pytest

from cnn import CNN, Worker, Aggregator, Trainer, flatten_params

EPS = 1e-3
ATOL = 2e-3


def _small_model(seed=0, **kw):
    kw.setdefault("input_shape", (1, 8, 8))
    kw.setdefault("conv_blocks", ((2, 3, 1, 1, 2),))
    kw.setdefault("output_size", 3)
    kw.setdefault("activation", "relu")
    return CNN(rng=np.random.default_rng(seed), **kw)


def _loss(model, X, Y):
    return model.loss(X, Y)


def _gradcheck(model, X, Y, param_indices):
    """Numeric grads for selected flat indices vs analytic (normalised)."""
    _, caches = model.forward(X)
    grads = model.back_prop(caches, Y, normalize=True)
    analytic = flatten_params(grads)
    vec0 = model.get_params_vector()
    numeric = np.zeros(len(param_indices))
    for k, idx in enumerate(param_indices):
        v = vec0.copy(); v[idx] += EPS; model.set_params_vector(v)
        lp = _loss(model, X, Y)
        v = vec0.copy(); v[idx] -= EPS; model.set_params_vector(v)
        lm = _loss(model, X, Y)
        numeric[k] = (lp - lm) / (2 * EPS)
    model.set_params_vector(vec0)
    return analytic[param_indices], numeric


def test_gradient_check_conv_pool_dense():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((3, 1, 8, 8)).astype(np.float32)
    Y = np.array([0, 1, 2])
    model = _small_model(0)
    # params: conv W (2*9=18), b 2, dense (2*4*4=32?) -> pick few across layers
    idx = [0, 5, 18, 20, 21, 30, 40]  # spread conv, bias, dense
    n = model.get_params_vector().size
    idx = [i for i in idx if i < n]
    a, num = _gradcheck(model, X, Y, idx)
    assert np.allclose(a, num, atol=ATOL, rtol=1e-2), f"analytic {a} vs numeric {num}"


def _split(X, Y, sizes):
    out, off = [], 0
    for s in sizes:
        out.append((X[off:off + s], Y[off:off + s])); off += s
    return out


def test_equivalence_aggregated_vs_single_unequal_partial():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((11, 1, 8, 8)).astype(np.float32)  # 11: partial last batch
    Y = rng.integers(0, 3, size=11)
    for sizes in ([4, 4, 3], [5, 3, 3], [11], [1, 2, 8]):
        model = _small_model(3)
        _, caches = model.forward(X)
        single = model.back_prop(caches, Y, normalize=True)
        workers = [Worker(i, model) for i in range(len(sizes))]
        msgs = [w.compute([mb], step=0) for w, mb in zip(workers, _split(X, Y, sizes))]
        agg = Aggregator.aggregate(msgs)
        for gs, ga in zip(single, agg):
            if gs is None:
                assert ga is None
            else:
                for a, b in zip(gs, ga):
                    assert np.allclose(a, b, atol=1e-6), (sizes, a, b)


def test_sync_invariant_after_each_step():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((16, 1, 8, 8)).astype(np.float32)
    Y = rng.integers(0, 3, size=16)
    model = _small_model(5)
    trainer = Trainer(model, X, Y, n_workers=4, seed=0, global_batch_size=8)
    for epoch in range(2):
        # run one epoch manually via train(epochs=1) then check copies equal
        trainer.train(epochs=1, alpha=0.05)
        vecs = [w.model.get_params_vector() for w in trainer.workers]
        for v in vecs[1:]:
            assert np.max(np.abs(v - vecs[0])) == 0.0


def test_n1_identical_to_non_distributed():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((16, 1, 8, 8)).astype(np.float32)
    Y = rng.integers(0, 3, size=16)
    # distributed N=1
    m1 = _small_model(9)
    t = Trainer(m1, X, Y, n_workers=1, seed=0, global_batch_size=4)
    t.train(epochs=2, alpha=0.1)
    # single process with same init and same shuffle seed
    m2 = _small_model(9)
    shuffle = np.random.default_rng(1)  # Trainer(seed=0) -> shuffle_rng seed+1
    m2.fit(X, Y, epochs=2, alpha=0.1, batch_size=4, shuffle_rng=shuffle)
    assert np.allclose(t.model.get_params_vector(), m2.get_params_vector(), atol=1e-6)


def test_independent_copies_identical_start_no_shared_memory():
    model = _small_model(0)
    w1, w2 = Worker(0, model), Worker(1, model)
    assert np.array_equal(w1.model.get_params_vector(), w2.model.get_params_vector())
    w1.model.params[0][0][0, 0, 0, 0] += 1.0
    assert not np.array_equal(w1.model.get_params_vector(), w2.model.get_params_vector())


def test_aggregator_weighted_not_simple_mean():
    # Unequal shards: simple mean would give wrong result; weighted must match single.
    rng = np.random.default_rng(3)
    X = rng.standard_normal((10, 1, 8, 8)).astype(np.float32)
    Y = rng.integers(0, 3, size=10)
    model = _small_model(4)
    _, caches = model.forward(X)
    single = model.back_prop(caches, Y, normalize=True)
    workers = [Worker(i, model) for i in range(2)]
    msgs = [w.compute([mb]) for w, mb in zip(workers, _split(X, Y, [9, 1]))]
    agg = Aggregator.aggregate(msgs)
    for gs, ga in zip(single, agg):
        if gs is not None:
            for a, b in zip(gs, ga):
                assert np.allclose(a, b, atol=1e-6)


def test_flatten_roundtrip():
    model = _small_model(0)
    vec = model.get_params_vector()
    model.set_params_vector(vec + 0.01)
    assert np.allclose(model.get_params_vector(), vec + 0.01)
