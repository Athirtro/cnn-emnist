# Cómo funciona una CNN, partiendo de la MLP del cuaderno

Este documento explica la red neuronal convolucional (`CNN` en `cnn.py`)
tomando como punto de partida la red neuronal densa (MLP) del cuaderno de
referencia, de la que hereda casi toda la maquinaria de entrenamiento.
La única idea nueva es **cómo se calculan las primeras capas**; el final
de la red y el aprendizaje son los mismos de la MLP.

## 1. La MLP que ya conocemos

En la MLP del cuaderno la entrada es un vector aplanado: una imagen de
28×28 se convierte en `X` de forma `(784, m)` (una columna por muestra).
Cada capa densa hace:

```text
Z = W·A_prev + b          # activación lineal (pre-activación)
A = f(Z)                  # f: sigmoid, tanh o ReLU
```

y la última capa termina en softmax, con pérdida de entropía cruzada.
El truco que hace funcionar todo el entrenamiento es que, con softmax +
entropía cruzada, el gradiente de la salida es simplemente:

```text
dZ = A - Y                # A: probabilidades, Y: one-hot de la clase
```

a partir del cual se propagan hacia atrás los gradientes de cada capa:

```text
dW = (1/m) · dZ·A_prevᵀ        db = (1/m) · Σ dZ
dZ_prev = (Wᵀ·dZ) ⊙ f′(A_prev) # f′ evaluada como función de la salida A
```

y se actualiza con SGD simple, sin momentum ni estado:

```text
W ← W − α·dW                    b ← b − α·db
```

En el cuaderno esto se organiza con dos registros que este repo conserva
tal cual en `cnn.py`:

- `ACTIVATIONS`: `Activation(fn, deriv, default_init)`, donde `deriv`
  recibe la **salida** `A` (p. ej. para ReLU, `A > 0`; para tanh,
  `1 − A²`; para sigmoid, `A·(1−A)`).
- `INITIALIZERS`: `small`, `xavier`, `he`, elegidos por defecto según la
  activación (`he` para ReLU).

### El problema de la MLP con imágenes

La MLP trata cada píxel como una variable independiente: en la primera
capa densa, el peso que conecta el píxel (5, 5) con una neurona oculta es
un parámetro distinto del que conecta el píxel (6, 5), aunque ambos
informen del mismo trazo desplazado una posición. Consecuencias:

1. **Explosión de parámetros.** Pasar de 784 entradas a, digamos, 128
   neuronas ya cuesta 784·128 ≈ 100.000 parámetros solo en la primera
   capa, y el modelo debe aprender por separado cada posición.
2. **Sin estructura espacial.** La MLP no sabe que dos píxeles vecinos
   están cerca: si el mismo dígito aparece desplazado dos píxeles, es un
   problema completamente nuevo para ella.

La CNN corrige exactamente estos dos puntos, y por eso conviene
entenderla como *una MLP con dos restricciones adicionales en sus
primeras capas*: **compartir pesos** y **respetar la vecindad espacial**.

## 2. La idea central: convolución = capa densa con pesos compartidos

Una capa convolucional aplica el mismo patrón lineal en cada posición
de la imagen. Para un filtro de 3×3, en cada parche de 3×3 píxeles se
calcula:

```text
Z[f, i, j] = Σ_{u,v} W[f, u, v] · X[i+u, j+v] + b[f]
```

Es decir: es la misma multiplicación `W·x + b` de la capa densa, pero
`x` ya no es la imagen entera sino un parche local, y el **mismo `W`
se reutiliza en todas las posiciones** `(i, j)`. Cada filtro `f` produce
un mapa de activación completo (uno por posición), y con `F` filtros
obtenemos `F` mapas: son las *características* que la red aprende —
bordes, esquinas, trazos, etc.

De esta única decisión se derivan las dos propiedades que la MLP no tenía:

| Propiedad | Qué significa | Por qué aparece |
|---|---|---|
| **Compartir pesos** | El filtro de 3×3 tiene 9 pesos por canal de entrada, no uno por píxel de la imagen | El mismo `W` se aplica en cada posición |
| **Equivariancia a traslaciones** | Si la entrada se desplaza, el mapa de salida se desplaza igual | El detector no depende de la posición absoluta |

En el caso base de este repo (`Conv 3×3, 8 filtros, stride 1, padding 1`)
la capa conv tiene solo `8·(1·3·3) + 8 = 80` parámetros, frente a los
~100.000 de una densa equivalente. Y esos 80 parámetros se entrenan con
*las 28×28 posiciones de cada imagen a la vez*: cada muestra aporta
muchísima más evidencia sobre el filtro que sobre un peso denso.

### Cómo se calcula sin bucles por píxel: `im2col`

En `cnn.py`, `Conv2D.forward` no recorre la imagen píxel a píxel. Usa
`im2col`: cada parche 3×3 se copia como una fila de una matriz
`(N·H_out·W_out, C·3·3)`, y la convolución entera se convierte en una
única multiplicación de matrices:

```text
cols (N·H_out·W_out, C·K·K)  @  W.reshape(F, C·K·K)ᵀ  →  (N·H_out·W_out, F)
```

El bucle de Python solo recorre las 9 posiciones del kernel (`K·K`),
nunca los píxeles de la imagen; todo lo demás es álgebra vectorizada
en NumPy/CuPy (`xp`). El backward usa la operación inversa, `col2im`,
para repartir el gradiente de cada parche de vuelta a la imagen.

## 3. Activación y pooling: lo que viene después de la conv

Igual que en la MLP, cada `Z` pasa por la activación, con la peculiaridad
de que aquí la salida ya no es un vector sino un volumen `(N, F, H, W)`:

```text
Conv2D  →  ReLU  →  MaxPool2D
```

- **ReLU** (`ACTIVATIONS["relu"]`): `max(Z, 0)`, derivada `A > 0`. Es la
  misma función de la MLP; en `CNN.forward` se aplica tras cada conv y
  tras cada densa oculta, nunca tras la salida (que usa softmax).
- **MaxPool 2×2**: en cada ventana de 2×2 se queda solo el máximo. No
  tiene parámetros: su *forward* guarda, para cada ventana, la posición
  del máximo (`argmax`); su *backward* envía el gradiente solo a esa
  posición y cero al resto. Sirve para dos cosas: reducir el tamaño
  espacial (28×28 → 14×14) y quedarse con la característica más fuerte
  de cada zona, lo que da cierta tolerancia a pequeños desplazamientos.

Repetir el bloque `Conv → activación → MaxPool` es opcional: en
`CNN.__init__`, `conv_blocks` es una lista de tuplas
`(filtros, kernel, stride, padding, pool)` y el caso base usa una sola.

## 4. Flatten + Dense: la CNN termina siendo la MLP

Tras las capas espaciales hay que volver a un vector para clasificar.
`Flatten` simplemente reordena `(N, C, H, W) → (N, C·H·W)`, y a partir
de ahí la red es **exactamente la MLP del cuaderno**:

```text
Flatten → Dense(47) → Softmax
```

`Dense.forward` es `X @ Wᵀ + b` con `W` de forma `(47, C·H·W)`, y su
backward es el de siempre: `dW = dAᵀ·X`, `db = Σ dA`, `dX = dA·W`.
Este último `dX` es el gradiente que se inyecta de vuelta en el
`Flatten`, que lo reordena a `(N, C, H, W)` para seguir propagando por
el pooling y la conv.

### Recorrido completo del caso base (EMNIST, 28×28, 47 clases)

```text
Entrada          (N, 1, 28, 28)     # 1 canal (grises), layout NCHW
Conv 3×3, 8 filtros, pad 1, s 1   → (N, 8, 28, 28)    # 80 parámetros
ReLU                             → (N, 8, 28, 28)
MaxPool 2×2                      → (N, 8, 14, 14)
Flatten                          → (N, 1568)           # 8·14·14
Dense 47 + Softmax               → (N, 47)             # 73.743 parámetros
```

Total: 73.823 parámetros, la inmensa mayoría en la densa final (como en
la MLP). La diferencia es que ahora esos parámetros reciben entradas
que ya son características aprendidas invariantes a la posición, no
píxeles crudos.

## 5. El entrenamiento es el mismo, capa por capa

`CNN.forward` devuelve las activaciones y los caches de cada capa;
`CNN.back_prop` recorre las capas en orden inverso y, en cada una,
aplica la misma receta de las capas densas:

1. **Última capa**: `dZ = A − Y` (softmax + entropía cruzada).
2. **Dense**: `dW`, `db`, y `dA_prev = dZ·W`.
3. **Flatten**: reordena el gradiente a volumen.
4. **MaxPool**: lo pasa solo por la posición del máximo.
5. **Conv**: `dW` es, de nuevo, una multiplicación de matrices con
   `im2col` (`dW = dcolsᵀ·cols`), y `dX` se reconstruye con `col2im`,
   sumando la contribución de cada parche a cada píxel de la entrada.

Los gradientes se normalizan dividiendo por `m` (tamaño del lote) en
`back_prop(..., normalize=True)`, y la actualización es SGD plano:
`W ← W − α·dW`. Esto se comprueba en `test_cnn.py` con un gradient
check por diferencias finitas que cubre conv, pooling y dense sin
usar autograd en ningún punto.

## 6. Dónde encaja la simulación distribuida

Nada de lo anterior cambia cuando se reparte el entrenamiento entre
workers (`Worker` / `Aggregator` / `Trainer` en `cnn.py`): cada worker
tiene una copia independiente de los mismos parámetros, procesa su
shard con el forward/backward de arriba y devuelve la **suma** de
gradientes sin normalizar junto con su número de muestras. El
agregador divide la suma total entre el total de muestras — la media
ponderada correcta aunque los shards sean desiguales — y cada worker
aplica el mismo paso SGD sobre su copia, de modo que todas las copias
permanecen idénticas tras cada sincronización.

## 7. Resumen: qué se hereda de la MLP y qué es nuevo

| Aspecto | MLP del cuaderno | CNN de este repo |
|---|---|---|
| Entrada | Vector `(784, m)` | Volumen `(N, 1, 28, 28)` |
| Primera operación | `Z = W·X + b` sobre toda la imagen | `Z[f,i,j] = Σ W[f]·parche[i,j] + b[f]` (mismo `W` en cada posición) |
| Parámetros de la 1ª capa | Uno por píxel y neurona | Uno por posición del kernel y filtro (80 en el caso base) |
| Activación | `f(Z)` sobre un vector | `f(Z)` sobre un volumen; mismos registros `ACTIVATIONS` |
| Reducción espacial | No existe | `MaxPool` (sin parámetros, gradiente al argmax) |
| Final de la red | Dense + softmax, `dZ = A − Y` | Idéntico: `Flatten → Dense(47) → Softmax` |
| Backward | `dW = dZ·A_prevᵀ`, cadena con `f′(A)` | Idéntico en Dense; `im2col/col2im` en Conv |
| Optimizador | SGD simple, gradiente / `m` | Idéntico |
| Cómputo | `xp` (NumPy o CuPy) | Idéntico, alias `xp` (`cp` en GPU, `np` de respaldo) |

En una frase: **la CNN es una MLP cuya primera capa densa se ha
especializado para imágenes** — en lugar de conectar cada píxel con
cada neurona, conecta cada parche local con un conjunto compartido de
filtros, apila esos mapas de activación, los compacta con pooling y
solo entonces los aplana y los entrega a la MLP de siempre.
