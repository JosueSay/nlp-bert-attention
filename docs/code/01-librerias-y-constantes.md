# Librerías y constantes

[Guía completa del laboratorio](../00-guide.md)

Este documento explica las celdas de importaciones y de constantes del notebook `kafka.ipynb`. Es la primera parte de la documentación del código; continúa en [Funciones auxiliares](./02-funciones-auxiliares.md) y [Partes A, B, C y D](./03-partes-a-b-c-d.md).

## Versiones instaladas

| Paquete | Versión | Origen |
| --- | --- | --- |
| `torch` | 2.13.0 | `requirements.txt` |
| `transformers` | 5.16.1 | `requirements.txt` |
| `tokenizers` | 0.23.1 | dependencia de `transformers` |
| `tabulate` | 0.10.0 | `requirements.txt` |
| `re`, `contextlib`, `pathlib` | — | biblioteca estándar de Python |

## Librerías

### `import re`

Módulo de expresiones regulares de la biblioteca estándar. Se usa en dos lugares, ambos dentro de `getCorpus`:

- `re.compile(...)` construye `SENTENCE_PATTERN`, el patrón que segmenta el texto en oraciones. Compilar una sola vez y reutilizar el objeto evita recompilar el patrón en cada llamada.
- `re.sub(r"\s+", " ", texto)` colapsa cualquier secuencia de espacios, tabulaciones y saltos de línea en un único espacio. Es lo que une los renglones del libro en un texto continuo antes de segmentar.

### `from contextlib import redirect_stdout`

`redirect_stdout` es un administrador de contexto (context manager) de la biblioteca estándar. Mientras se está dentro de su bloque `with`, reemplaza temporalmente `sys.stdout` por el objeto que se le pase. Al salir del bloque, restaura el `stdout` original.

Traducido: todo lo que un `print()` escribiría en pantalla, se escribe en el archivo.

```python
with open("salida.txt", "w") as file:
    with redirect_stdout(file):
        print("esto no aparece en pantalla")  # va al archivo
print("esto sí aparece en pantalla")
```

Por qué está en este proyecto: la función `saveResult` lo usa para capturar la salida de las funciones de impresión (`printTokenizedCorpus`, `printAttentionShapes`, `printAttentionAnalysis`) y volcarla en los `.txt` de `results/`. Así el mismo código sirve para mostrar en el notebook y para generar la evidencia entregable, sin duplicar lógica ni tener que reescribir las funciones para que devuelvan cadenas.

Detalle importante que explica el diseño de `saveResult`: por eso recibe una función (un `lambda`) y no un resultado ya calculado. La función debe **ejecutarse dentro** del bloque `with` para que sus `print` queden capturados. Si se le pasara `printTokenizedCorpus(corpus)` directamente, Python evaluaría esa llamada antes de entrar al `with`, imprimiría en pantalla y le pasaría `None` a `saveResult`.

Nota técnica: `redirect_stdout` es reentrante pero no es seguro entre hilos, porque `sys.stdout` es una variable global del proceso.

### `from pathlib import Path`

`Path` representa una ruta del sistema de archivos como un objeto, en lugar de como una cadena de texto. La clase concreta que se instancia depende del sistema operativo: `PosixPath` en Linux y WSL, `WindowsPath` en Windows. Esto evita tener que preocuparse por si el separador es `/` o `\`.

Lo que se usa en el notebook:

- Operador `/` para unir rutas. `model_dir / model_name` produce `models/bert-base-multilingual-cased`. Es más legible que `os.path.join(...)` o que concatenar cadenas a mano.
- `.exists()` devuelve `True` si la ruta ya existe en disco. En `getModel` decide si hay que descargar el modelo o si ya está en caché local.
- `.mkdir(parents=True, exist_ok=True)` crea el directorio. `parents=True` crea también los directorios intermedios que falten; `exist_ok=True` evita que lance `FileExistsError` si ya existe.
- `.parent` devuelve el directorio que contiene a la ruta. En `saveResult` se usa como `log_path.parent` para asegurar que `results/` exista antes de escribir el archivo dentro.

`Path` acepta `/` como separador incluso en Windows y lo normaliza internamente, por eso `Path("data/corpus.txt")` es válido en cualquier plataforma.

### `from tabulate import tabulate`

`tabulate` convierte una lista de listas en una tabla de texto formateada y la **devuelve como cadena**. No imprime nada por sí sola: en el notebook siempre aparece envuelta en `print(tabulate(...))`.

Argumentos que usa el proyecto:

- `headers=["Ranking", "Token", "Índice", "Atención"]` define los nombres de las columnas.
- `tablefmt="grid"` dibuja el marco completo con `+---+` y `|`. Es el estilo que se ve en los archivos de `results/`.
- `floatfmt=".6f"` fuerza seis decimales fijos en las columnas numéricas. Sin esto, valores pequeños como `0.000886` se mostrarían en notación científica o con distinta cantidad de decimales por fila, y las tablas quedarían difíciles de comparar visualmente.

Es una dependencia puramente de presentación: no interviene en ningún cálculo.

### `import torch`

PyTorch es el motor de tensores sobre el que corre el modelo. Los pesos de atención que devuelve BERT son objetos `torch.Tensor`, no listas de Python.

Lo que se usa en el notebook:

- `torch.no_grad()` es un administrador de contexto que desactiva el registro del grafo de autograd. Durante la inferencia no se van a calcular gradientes, así que construir ese grafo solo gasta memoria y tiempo. Envuelve todas las llamadas `model(**inputs)`.
- `torch.topk(weights, k)` devuelve los `k` valores más grandes de un tensor junto con sus posiciones, en dos tensores: `(values, indices)`, ordenados de mayor a menor. Es lo que produce el ranking de los 5 tokens más atendidos.
- `.tolist()` convierte un tensor a tipos nativos de Python, necesario para que `tabulate` pueda formatear los números.
- Indexado de tensores: `attentions[layer][0, head, token_index]` selecciona capa, elemento del lote, cabeza y fila en una sola expresión.

Distinción que conviene tener clara, porque son dos cosas separadas que suelen confundirse:

- `model.eval()` cambia el **modo** de los módulos: apaga el dropout y fija el comportamiento de las capas de normalización. En este modelo importa de verdad, porque su `config.json` declara `hidden_dropout_prob: 0.1` y `attention_probs_dropout_prob: 0.1`. Sin `.eval()`, el 10 % de los pesos de atención se anularía al azar en cada pasada y los resultados no serían reproducibles.
- `torch.no_grad()` no cambia el modo de nada: solo evita que se guarde información para el cálculo de gradientes.

Se necesitan las dos. El notebook llama a `.eval()` dentro de `loadModel` y usa `no_grad()` en cada inferencia.

### `from transformers import AutoTokenizer, AutoModel`

Las clases `Auto*` de Hugging Face son fábricas: leen el `config.json` del modelo, identifican la arquitectura y devuelven una instancia de la clase concreta correspondiente. No hay que saber de antemano qué clase usar.

En este caso el `config.json` local declara `"model_type": "bert"` y `"architectures": ["BertModel"]`, de modo que:

- `AutoTokenizer.from_pretrained(model_path)` devuelve un `BertTokenizerFast`, que implementa el algoritmo de subpalabras WordPiece (es el responsable de los tokens con prefijo `##`).
- `AutoModel.from_pretrained(model_path)` devuelve un `BertModel`.

`AutoModel` es la clase **base**, sin cabeza de tarea. Devuelve representaciones (`last_hidden_state`, `pooler_output`) y, si se pide, las atenciones. No produce logits ni predicciones. Las variantes con cabeza serían `AutoModelForMaskedLM`, `AutoModelForSequenceClassification`, `AutoModelForTokenClassification`, etc.

Que aquí se use `AutoModel` es una decisión con consecuencias para el análisis: como no hay tarea ni predicción, la atención observada no puede interpretarse como la explicación de una salida concreta. Solo describe cómo se construyen las representaciones internas.

En `loadModel` se pasa `output_attentions=True` a `from_pretrained`. Ese argumento se guarda en la configuración del modelo, así que aplica a todas las llamadas posteriores sin repetirlo. Hace que la salida del `forward` incluya el campo `attentions`.

Nota: las implementaciones rápidas de atención (SDPA, FlashAttention) no materializan la matriz completa de pesos, así que no pueden devolverla. Cuando se pide `output_attentions=True`, `transformers` vuelve a la implementación "eager" para poder exponerla. Puede aparecer un aviso en consola indicándolo; no es un error.

## Constantes

```python
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results")
CORPUS_PATH = Path("data/corpus.txt")
MODEL_NAME = "bert-base-multilingual-cased"
SENTENCES_LIMIT = 1
TOP_K = 5

SENTENCE_PATTERN = re.compile(
    r'(?:(?<=[.!?…])|(?<=[.!?…][»")\]]))'
    r'\s+'
    r'(?=[«¡¿—"(\[]*[A-ZÁÉÍÓÚÜÑ])'
)
```

### Las tres rutas

`MODEL_DIR`, `RESULTS_DIR` y `CORPUS_PATH` son objetos `Path` en lugar de cadenas para poder usar el operador `/` y los métodos `.exists()` y `.mkdir()` sin importar `os`.

Son **rutas relativas**, así que se resuelven contra el directorio de trabajo del kernel de Jupyter, que es la carpeta desde la que se abrió el notebook. Como el notebook está en la raíz del repositorio, funcionan. Si se moviera el archivo a una subcarpeta, todas romperían.

Cada una cumple un papel distinto:

- `MODEL_DIR` es la caché local del modelo. `getModel` comprueba si `models/bert-base-multilingual-cased` existe; si no, descarga desde el Hub y guarda con `save_pretrained`. En las ejecuciones siguientes ya no hay descarga ni conexión a internet.
- `RESULTS_DIR` es donde `saveResult` deja los `.txt` que son la evidencia entregable del laboratorio.
- `CORPUS_PATH` apunta al fragmento de *La metamorfosis* que se analiza.

Detalle que conviene notar: `MODEL_DIR / MODEL_NAME` funciona como nombre de carpeta plano porque `bert-base-multilingual-cased` no lleva `/`. Si se cambiara a un modelo con organización en el identificador, por ejemplo `dccuchile/bert-base-spanish-wwm-cased`, el operador `/` crearía una carpeta anidada `models/dccuchile/bert-base-spanish-wwm-cased`. Sigue funcionando, pero la estructura de directorios cambia.

### `MODEL_NAME = "bert-base-multilingual-cased"`

Es el identificador del modelo en el Hub de Hugging Face. Se descompone así:

- `bert`: arquitectura Transformer encoder-only, que es exactamente lo que pide la guía.
- `base`: el tamaño intermedio. Según el `config.json` descargado: 12 capas (`num_hidden_layers`), 12 cabezas de atención (`num_attention_heads`), dimensión oculta 768 (`hidden_size`), capa intermedia de 3072 y un máximo de 512 posiciones.
- `multilingual`: entrenado sobre las Wikipedias de 104 idiomas con un vocabulario compartido de 119 547 piezas (`vocab_size`). Es lo que permite procesar español sin entrenar nada.
- `cased`: conserva mayúsculas y minúsculas. `Gregorio` y `gregorio` son tokens distintos. Para este laboratorio es deseable, porque los nombres propios del texto se mantienen identificables.

Dos consecuencias que aparecen directamente en los resultados:

- 768 dimensiones repartidas entre 12 cabezas dan 64 dimensiones por cabeza. Cada capa produce 12 distribuciones de atención distintas, y hay 12 capas: 144 matrices por oración. Inspeccionar una capa y una cabeza es mirar una porción muy pequeña.
- Al ser un vocabulario compartido entre 104 idiomas, le tocan menos piezas al español que a un modelo monolingüe. Por eso hay tanta fragmentación en subpalabras: `querido` se parte en `quer` + `##ido` y `enseguida` en `ens` + `##egu` + `##ida`.

### `SENTENCES_LIMIT = 1`

Solo controla cuántas oraciones se muestran en pantalla, en las celdas que hacen `corpus[:SENTENCES_LIMIT]` y `tokenized_corpus[:SENTENCES_LIMIT]`.

No afecta a lo que se guarda en disco: las llamadas a `saveResult` reciben el corpus completo, por lo que los `.txt` contienen las 79 líneas. Es una constante de presentación, para que el notebook ejecutado no quede con cientos de líneas de salida.

### `TOP_K = 5`

Cuántos tokens entran en el ranking de atención. El valor no es arbitrario: la Parte C de la guía pide explícitamente "los 5 tokens con mayor peso de atención".

Se pasa como `top_n` a `getTopAttention`, que lo usa en `torch.topk(weights, k=min(top_n, len(tokens)))`. El `min` protege el caso de oraciones más cortas que 5 tokens, que existen en este corpus: hay líneas de 2 y 3 palabras.

### `SENTENCE_PATTERN`

Expresión regular compilada que marca dónde termina una oración. `getCorpus` la usa con `.split()` para segmentar el corpus.

Se lee en tres piezas:

- `(?:(?<=[.!?…])|(?<=[.!?…][»")\]]))` es una mirada hacia atrás: exige que justo antes del punto de corte haya un signo terminal, opcionalmente seguido de un signo de cierre. La alternativa con dos ramas existe porque Python solo admite miradas hacia atrás de longitud fija, así que hay que enumerar los casos en lugar de usar un cuantificador.
- `\s+` consume el espacio que separa las dos oraciones. Es la única parte que se consume, por eso desaparece del resultado.
- `(?=[«¡¿—"(\[]*[A-ZÁÉÍÓÚÜÑ])` es una mirada hacia adelante: exige que lo que sigue empiece una oración, es decir, cero o más signos de apertura y después una mayúscula. Incluye `¡`, `¿`, `«` y la raya de diálogo `—` porque el texto los usa, y las vocales acentuadas porque el corpus está en español.

El diseño con miradas hacia atrás y hacia adelante es deliberado: al no consumir los signos, estos quedan dentro de la oración a la que pertenecen y el texto se conserva íntegro.

Qué evita en concreto sobre este corpus:

- No parte en `verle?», Gregorio pensaba`, porque después del `»` viene una coma y no un espacio seguido de mayúscula. Ese interrogante cierra una cita, no la oración.
- Sí parte en `entrada a su habitación. —Pasa, no se le ve`, porque la raya de diálogo está contemplada como inicio válido.
- Sí parte entre `¡Dejadme entrar a ver a Gregorio!` y `¡Pobre hijo mío!`, que son dos exclamaciones independientes.

Sobre `data/corpus.txt` produce 33 oraciones.

Limitación conocida: la última oración del corpus queda incompleta (`... pero donde`), porque el extracto son dos páginas del libro y la frase continúa en la página siguiente. No es un fallo del patrón.

## Resumen

| Elemento | Tipo | Para qué sirve aquí |
| --- | --- | --- |
| `re` | estándar | Normalizar espacios y segmentar el corpus en oraciones |
| `redirect_stdout` | estándar | Capturar los `print` de las funciones de reporte en los `.txt` de `results/` |
| `Path` | estándar | Rutas portables, con `/`, `.exists()`, `.mkdir()` y `.parent` |
| `tabulate` | presentación | Formatear los rankings de atención como tablas legibles |
| `torch` | cómputo | Tensores, `no_grad()` para inferencia y `topk()` para el ranking |
| `AutoTokenizer` | modelo | Tokenizador WordPiece que produce `[CLS]`, `[SEP]` y piezas `##` |
| `AutoModel` | modelo | BERT base sin cabeza de tarea, con `output_attentions=True` |
| `MODEL_DIR` | ruta | Caché local del modelo, evita descargarlo cada vez |
| `RESULTS_DIR` | ruta | Destino de la evidencia entregable |
| `CORPUS_PATH` | ruta | Fragmento de *La metamorfosis* a analizar |
| `MODEL_NAME` | configuración | 12 capas, 12 cabezas, 768 dimensiones, vocabulario multilingüe con distinción de mayúsculas |
| `SENTENCES_LIMIT` | presentación | Recorta solo lo que se imprime en el notebook |
| `TOP_K` | análisis | Tamaño del ranking, fijado por la Parte C de la guía |
| `SENTENCE_PATTERN` | análisis | Regla de corte que separa el corpus en oraciones completas |
