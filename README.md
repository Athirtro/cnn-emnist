# CNN EMNIST — NumPy + simulación distribuida

Red neuronal convolucional implementada **desde cero con NumPy/CuPy** (alias `xp = cp` en GPU o `xp = np` como respaldo, sin autograd ni torch en el modelo) sobre EMNIST Balanced,
con simulación de entrenamiento distribuido mediante acumulación de gradiente.

## Estructura

| Archivo | Descripción |
|---|---|
| `cnn.py` | Modelo (`CNN`, `Conv2D`, `MaxPool2D`, `Flatten`, `Dense`), registros `ACTIVATIONS`/`INITIALIZERS`, `Worker`, `Aggregator`, `Trainer`, `load_emnist`, `run_experiment`, `plot_history` |
| `test_cnn.py` | Verificaciones obligatorias (pytest) |
| `notebook.py` | Cuaderno [marimo](https://marimo.io) que evalúa solo los casos de workers (1, 2, 4, 8) con reparto equitativo |
| `pyproject.toml` | Proyecto gestionado con [uv](https://docs.astral.sh/uv/) |

## Backend NumPy/CuPy (`xp`)

Al principio de `cnn.py` y del cuaderno hay un bloque `try/except`:
intenta `import cupy as xp` (con GPU CUDA visible) y, si falla, usa
`import numpy as xp`. `BACKEND`/`cnn.BACKEND` indica con qué librería se
trabaja (`"CuPy"` o `"NumPy"`) y el cuaderno lo muestra en su primera celda.
El modelo y las pruebas solo cambian el alias `np` → `xp`; el barajado de
índices y `load_emnist` permanecen en NumPy (host). Se puede forzar el
respaldo con `CNN_BACKEND=numpy uv run pytest -q`. Instalación de CuPy con uv:
`uv add "cupy-cuda13x>=14.0"` (requiere controlador CUDA y GPU; aquí
RTX 5050, CUDA 13.5).

## Técnica (del cuaderno de referencia)

- Registros de activaciones: dataclass `Activation(fn, deriv, default_init)` con
  `deriv` en función de la salida `A`; inicializadores `small`/`xavier`/`he`
  con `default_init` por activación.
- Parámetros como lista por capa; `forward` devuelve activaciones/caches;
  `back_prop` devuelve gradientes; softmax + entropía cruzada con `dZ = A − Y`.
- SGD simple sin momentum ni estado; gradientes normalizados por el tamaño del lote
  (`back_prop(..., normalize=True)`; los workers piden suma sin normalizar).
- Convolución vectorizada con `im2col`/`col2im` (sin bucles por píxel), `float32`
  por defecto (`dtype` configurable). Semilla explícita y generadores NumPy
  separados para inicialización y barajado.
- Arquitectura configurable: `Conv(k, filtros, stride, padding) → activación → MaxPool`
  repetido, luego `Flatten → Dense → Softmax`. Caso base: Conv 3×3 (8 filtros) → ReLU →
  MaxPool 2×2 → Flatten → Dense 47.

## Simulación distribuida (solo mensajes de datos)

1. **Worker**: copia profunda independiente de los parámetros + shard propio.
   En cada paso procesa micro-lotes y acumula el gradiente como **suma sin normalizar**,
   contando muestras. Devuelve `WorkerMessage(worker_id, step, grads, n_samples)`.
2. **Aggregator**: gradiente global = suma de gradientes / total de muestras
   (media ponderada por `n_samples`, no media simple; correcto con shards desiguales
   y lotes parciales).
3. **Trainer**: reparte shards (barajado global con semilla común, sin solapamiento),
   orquesta pasos, devuelve el gradiente global y cada worker aplica SGD sobre su copia.

Las copias inician con valores idénticos pero son objetos independientes.
Los parámetros son aplanables a un buffer único (`get/set_params_vector`) para
medir/transportar el gradiente. El protocolo de mensajes permite sustituir los
workers por procesos reales sin cambiar el agregador.

**Instrumentación** (por paso/época, en `Trainer.history`): muestras por worker,
bytes de gradiente por worker (nº parámetros × bytes/elemento), nº de
sincronizaciones, tiempo de cómputo por worker, tiempo de agregación,
precisión de entrenamiento y de prueba.

## Dataset

EMNIST Balanced (112.800 train / 18.800 test, 47 clases). `load_emnist()` usa
`torchvision` **solo** para descargar, convierte una vez a NumPy `(N,1,28,28)`
en `[0,1]` con corrección de orientación (transpuesta). Parámetro `fraction`
para usar una fracción del train en desarrollo.

## Instalación y ejecución (todo con uv)

```bash
uv sync                 # crea .venv e instala dependencias
uv run pytest -q        # tests
uv run marimo edit notebook.py   # abre el cuaderno (casos de workers)
# o ejecutar el cuaderno como script:
uv run python notebook.py
```

> El cuaderno usa `FRACTION = 0.05` por defecto para desarrollo rápido
> (semilla `SEED = 0`); pon `1.0` para el dataset completo.
> Dependencias del cuaderno (cabecera `/// script`): marimo, matplotlib,
> numpy, torch, torchvision — torch/torchvision solo para descargar EMNIST.

## Tests

```bash
uv run pytest -q
```

1. **Gradient check** por diferencias finitas en conv, pooling y dense.
2. **Equivalencia**: gradiente agregado (N workers) == gradiente de un único
   proceso sobre el lote global, con shards desiguales y último lote parcial.
3. **Invariante de sincronía**: tras cada paso la diferencia máxima entre
   copias de workers es 0.
4. **N=1 idéntico** al entrenamiento no distribuido.

## Evaluación por número de workers (cuaderno)

El cuaderno evalúa el modelo **únicamente** con los diferentes casos de workers
(1, 2, 4, 8). La única variable es el número de workers: arquitectura base
(Conv 3×3/8 → ReLU → MaxPool 2×2 → Flatten → Dense 47), ReLU, init He, lote
global 128 (el lote local es `128 / n_workers`), alpha y épocas fijos y misma
semilla en todos los casos.

El dataset se reparte de forma **equitativa** entre los workers en cada época:
tras el barajado global con semilla común, `Trainer._shards` usa
`np.array_split` para dar a cada worker una porción disjunta del mismo tamaño
(±1 si el total no es divisible). Las curvas de precisión de prueba se dibujan
en una única figura con la misma escala (0–1) y deben coincidir salvo ruido
numérico.

## Criterios de aceptación

- Todos los tests pasan (`uv run pytest -q`).
- Curvas con distinto N coinciden salvo ruido numérico.
- Instrucciones reproducibles con semilla explícita (`seed=0`).
