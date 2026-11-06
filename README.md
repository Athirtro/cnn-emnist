# CNN EMNIST — NumPy + simulación distribuida

Red neuronal convolucional implementada **desde cero solo con NumPy**
(sin autograd, sin torch/cupy en el modelo) sobre EMNIST Balanced,
con simulación de entrenamiento distribuido mediante acumulación de gradiente.

## Estructura

| Archivo | Descripción |
|---|---|
| `cnn.py` | Modelo (`CNN`, `Conv2D`, `MaxPool2D`, `Flatten`, `Dense`), registros `ACTIVATIONS`/`INITIALIZERS`, `Worker`, `Aggregator`, `Trainer`, `load_emnist`, `run_experiment`, `plot_history` |
| `test_cnn.py` | Verificaciones obligatorias (pytest) |
| `notebook.py` | Cuaderno [marimo](https://marimo.io) con caso base y experimentos A–D |
| `pyproject.toml` | Proyecto gestionado con [uv](https://docs.astral.sh/uv/) |

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
uv run marimo edit notebook.py   # abre el cuaderno (experimentos A–D)
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

## Experimentos ablativos (cuaderno)

Caso base: 1 worker, arquitectura base, ReLU, init He, lote 128, alpha fijo,
épocas fijas. Cada experimento cambia **una sola** variable:

- **A.** N workers (2, 4, 8) con lote global fijo (el lote local se reduce) →
  las curvas coinciden con la base salvo ruido numérico.
- **B.** N workers (2, 4, 8) con lote local fijo (el lote global crece).
- **C.** Shards desiguales 50/30/20 % → valida la ponderación del agregador.
- **D.** Pasos de acumulación por micro-lote (1, 2, 4) antes de sincronizar.

## Criterios de aceptación

- Todos los tests pasan (`uv run pytest -q`).
- Curvas de A con distinto N coinciden con la base salvo ruido numérico.
- Instrucciones reproducibles con semilla explícita (`seed=0`).
