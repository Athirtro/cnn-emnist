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

    return load_emnist, mo, plt, run_experiment


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
    return SEED, test_data, train_data


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Evaluación por número de workers

    Única variable: **número de workers** (1, 2, 4, 8). Todo lo demás es idéntico
    en todos los casos: Conv 3×3 (8 filtros) → ReLU → MaxPool 2×2 → Flatten →
    Dense 47, init He, lote global 128, alpha y épocas fijos, misma semilla.

    El dataset se reparte de forma **equitativa** entre los workers en cada época
    (`Trainer._shards` con `np.array_split`): tras el barajado global con semilla
    común, cada worker recibe una porción disjunta del mismo tamaño (±1 si el
    total no es divisible entre el número de workers), y en cada paso cada uno
    procesa su parte del lote global. El lote global es por tanto fijo (128) y
    el lote local es `128 / n_workers`.
    """)
    return


@app.cell
def _(SEED, mo, plt, run_experiment, test_data, train_data):
    CONFIG = dict(epochs=3, alpha=0.1, seed=SEED, global_batch_size=128)
    WORKER_CASES = (1, 2, 4, 8)

    texts, histories = [], {}
    for n in WORKER_CASES:
        _, text, _, hist = run_experiment(
            f"{n} worker(s) — reparto equitativo", train_data, test_data,
            n_workers=n, **CONFIG)
        texts.append(text)
        histories[n] = hist

    # Curvas comparables: misma escala (0–1) y mismos ejes para todos los casos.
    fig, ax = plt.subplots(figsize=(8, 5))
    for n, hist in histories.items():
        ax.plot(range(1, len(hist["test_acc"]) + 1), hist["test_acc"],
                marker="o", label=f"{n} worker(s)")
    ax.set_xlabel("Épocas"); ax.set_ylabel("Precisión de prueba (Accuracy)")
    ax.set_ylim(0, 1); ax.set_title("Precisión de prueba por número de workers")
    ax.grid(True, linestyle="--", alpha=0.6); ax.legend()
    plt.close(fig)

    mo.vstack([mo.md("\n\n".join(texts)), fig])
    return CONFIG, WORKER_CASES, histories


if __name__ == "__main__":
    app.run()
