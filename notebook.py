# /// script
# dependencies = [
#     "marimo",
#     "matplotlib>=3.8",
#     "numpy>=2.0",
#     "torch>=2.0",
#     "torchvision>=0.20",
# ]
# requires-python = ">=3.14"
# ///
import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # EMNIST — CNN NumPy + Simulación Distribuida

    CNN implementada desde cero solo con **NumPy** (`cnn.py`), sin autograd.
    Simulación distribuida por acumulación de gradiente: `Worker` → `Aggregator` → `Trainer`,
    comunicados solo por mensajes de datos.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from cnn import CNN, Worker, Aggregator, Trainer, load_emnist, run_experiment, plot_history
    return mo, np, plt, CNN, Worker, Aggregator, Trainer, load_emnist, run_experiment, plot_history


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dataset

    EMNIST Balanced: 112.800 train / 18.800 test, 47 clases, 28×28.
    `torchvision` solo descarga el dataset; se convierte una vez a NumPy
    `(N,1,28,28)` en `[0,1]` con corrección de orientación (transpuesta).
    """)
    return


@app.cell
def _(load_emnist, mo):
    FRACTION = 0.05  # fracción del train para desarrollo; pon 1.0 para el experimento completo
    SEED = 0
    train_data, test_data = load_emnist(root="./data", fraction=FRACTION, download=True)
    mo.md(f"Train: {train_data[0].shape} | Test: {test_data[0].shape}")
    return FRACTION, SEED, test_data, train_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Caso base

    1 worker, Conv 3×3 (8 filtros) → ReLU → MaxPool 2×2 → Flatten → Dense 47,
    init He, lote global 128, alpha fijo, épocas fijas, semilla común.
    """)
    return


@app.cell
def _(SEED, mo, run_experiment, test_data, train_data):
    BASE = dict(epochs=3, alpha=0.1, seed=SEED, global_batch_size=128)
    model0, text0, fig0 = run_experiment("Base (1 worker)", train_data, test_data, **BASE)
    mo.vstack([mo.md(text0), fig0])
    return BASE, model0


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Experimentos ablativos (una sola variable por experimento, misma semilla/escala)

    - **A.** N workers (2,4,8) con lote **global** fijo → el lote local se reduce.
    - **B.** N workers (2,4,8) con lote **local** fijo → el lote global crece.
    - **C.** Shards desiguales 50/30/20 % (3 workers) → valida la ponderación.
    - **D.** Pasos de acumulación por micro-lote (1, 2, 4) antes de sincronizar.
    """)
    return


@app.cell
def _(BASE, mo, plt, run_experiment, test_data, train_data):
    # A: lote global fijo
    results_a = {}
    for n in (1, 2, 4, 8):
        _, text, fig = run_experiment(f"A — {n} workers (lote global fijo)",
                                      train_data, test_data, n_workers=n, **BASE)
        results_a[n] = text
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title("A — N workers, lote global fijo (128)"); ax.set_ylim(0, 1)
    ax.set_xlabel("Épocas"); ax.set_ylabel("Precisión de prueba"); ax.grid(True, linestyle="--", alpha=0.6)
    mo.vstack([mo.md("\n\n".join(results_a.values())), fig])
    return results_a


@app.cell
def _(BASE, mo, plt, run_experiment, test_data, train_data):
    # B: lote local fijo (128 por worker)
    results_b = {}
    for n in (1, 2, 4, 8):
        _, text, fig = run_experiment(f"B — {n} workers (lote local fijo)",
                                      train_data, test_data, n_workers=n,
                                      local_batch_size=128, **{k: v for k, v in BASE.items() if k != "global_batch_size"})
        results_b[n] = text
    mo.md("\n\n".join(results_b.values()))
    return results_b


@app.cell
def _(BASE, mo, run_experiment, test_data, train_data):
    # C: shards desiguales 50/30/20
    _, text_c, fig_c = run_experiment("C — shards 50/30/20", train_data, test_data,
                                      n_workers=3, shard_proportions=[0.5, 0.3, 0.2], **BASE)
    mo.vstack([mo.md(text_c), fig_c])
    return


@app.cell
def _(BASE, mo, run_experiment, test_data, train_data):
    # D: acumulación de micro-lotes
    results_d = {}
    for acc in (1, 2, 4):
        _, text, fig = run_experiment(f"D — acumulación {acc}", train_data, test_data,
                                      n_workers=2, accumulate_steps=acc, **BASE)
        results_d[acc] = text
    mo.md("\n\n".join(results_d.values()))
    return results_d


if __name__ == "__main__":
    app.run()
