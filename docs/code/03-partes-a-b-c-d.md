# Partes A, B, C y D

[Guía completa del laboratorio](../00-guide.md)

Este documento explica las funciones que implementan las cuatro partes del procedimiento. Las librerías y constantes están en [Librerías y constantes](./01-librerias-y-constantes.md), y las funciones de carga y de datos en [Funciones auxiliares](./02-funciones-auxiliares.md).

## Panorama

| Parte | Qué pide la guía | Funciones | Evidencia |
| --- | --- | --- | --- |
| A | Tokenizar y mostrar los tokens de cada oración | `printTokenizedCorpus` | `results/tokenization.txt` |
| B | Ejecutar el modelo con atenciones y reportar la forma de las matrices | `getAttentions`, `printAttentionShapes` | `results/attention_shapes.txt` |
| C | Los 5 tokens más atendidos desde 2 tokens, en 2 capas y 2 cabezas | `getTopAttention`, `printTopAttention`, `printAttentionAnalysis` | `results/attention_analysis.txt` |
| D | Comparar dos oraciones y explicar si cambia la atención | `compareAttention` | `results/attention_comparison_gregorio.txt`, `results/attention_comparison_madre.txt` |

## El patrón de tres celdas

Las cuatro partes siguen la misma estructura, que conviene reconocer antes de entrar en el detalle:

1. **Definición.** Una o varias funciones que solo imprimen. No saben nada de archivos.
2. **Persistencia.** Una llamada a `saveResult` con un `lambda`, que ejecuta la función sobre el corpus completo y vuelca la salida al `.txt` correspondiente.
3. **Vista previa.** Una llamada directa recortada con `SENTENCES_LIMIT`, para que el notebook ejecutado muestre un ejemplo legible en lugar de cientos de líneas.

Los pasos 2 y 3 usan exactamente la misma función. Esa separación entre generar y persistir es la que se explica en la sección de `saveResult` del documento anterior, y es la razón por la que no hay código duplicado entre lo que se ve en pantalla y lo que se entrega.

## Parte A: tokenización

```python
def printTokenizedCorpus(tokenized_corpus):
    for item in tokenized_corpus:
        print("Oración:", item["sentence"])
        print("Tokens:", item["tokens"])
        print()
```

Es la función más simple del notebook. Recorre la lista de diccionarios que produjo `tokenizeCorpus` e imprime, para cada oración, el texto original y su lista de tokens. El trabajo real ya está hecho: aquí solo se presenta.

Que imprima las dos cosas juntas es lo que permite cumplir lo que pide la Parte A, que no es solo "mostrar los tokens" sino identificar las diferencias entre palabras lingüísticas y tokens del modelo. Sin el texto al lado, la comparación no se puede hacer.

### Qué se ve en la evidencia

Tres fenómenos, todos visibles en `results/tokenization.txt`.

**Tokens especiales.** Toda lista empieza con `[CLS]` y termina con `[SEP]`. No corresponden a ninguna palabra: son posiciones que BERT añade y que participan en la atención como cualquier otro token. `[CLS]` está pensado como agregador de la secuencia completa y `[SEP]` como marca de final.

**Subpalabras.** El prefijo `##` marca una pieza que continúa la anterior. En la primera oración del corpus:

```text
'quit', '##ado'          -> quitado
's', '##ában', '##a'     -> sábana
'ai', '##slar', '##se'   -> aislarse
'ag', '##rada', '##ble'  -> agradable
```

La razón está en el vocabulario del modelo: 119 547 piezas repartidas entre 104 idiomas, de modo que al español le tocan relativamente pocas y las palabras poco frecuentes se arman por partes.

**Palabra lingüística contra token del modelo.** No hay correspondencia uno a uno. Sobre el corpus completo, 922 palabras producen 1401 tokens, una razón de 1,52 tokens por palabra. En `corpus[5]`, 31 palabras se convierten en 44 tokens, de los cuales 9 llevan el prefijo `##`.

Esto tiene una consecuencia directa para las Partes C y D: la atención se reparte entre piezas, no entre palabras. Una palabra dividida en tres tokens ocupa tres filas y tres columnas de la matriz, y el peso que recibe se distribuye entre ellas.

## Parte B: ejecución del modelo

### `getAttentions`

```python
def getAttentions(sentence, tokenizer, model):
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        add_special_tokens=True
    )

    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.attentions
```

Ejecuta el modelo sobre una oración y devuelve solo las atenciones.

- `return_tensors="pt"` hace que el tokenizador devuelva tensores de PyTorch en vez de listas de Python. Es necesario porque el modelo espera tensores. También añade la dimensión de lote: `input_ids` pasa a tener forma `(1, n)`.
- `model(**inputs)` desempaqueta el diccionario, de modo que equivale a `model(input_ids=..., token_type_ids=..., attention_mask=...)`.
- `torch.no_grad()` evita construir el grafo de gradientes, que en inferencia solo gastaría memoria.
- `outputs.attentions` existe porque `loadModel` pasó `output_attentions=True` al cargar el modelo. Sin eso valdría `None`.

Lo que devuelve es una **tupla de 12 tensores**, uno por capa.

### `printAttentionShapes`

```python
def printAttentionShapes(corpus, tokenizer, model):
    for sentence in corpus:
        attentions = getAttentions(sentence, tokenizer, model)

        print("Oración:", sentence)
        print("Número de capas:", len(attentions))

        for i, attention in enumerate(attentions):
            print(
                f"Capa {i}:",
                tuple(attention.shape)
            )

        print()
```

Recorre el corpus y, para cada oración, imprime cuántas capas hay y la forma de la matriz de cada una. `len(attentions)` da 12 porque la tupla tiene un elemento por capa. `tuple(attention.shape)` convierte el objeto `torch.Size` en una tupla para que se imprima de forma compacta.

### Cómo se lee la forma

La guía anticipa `(batch_size, número_de_cabezas, número_de_tokens, número_de_tokens)`. Para `corpus[5]` sale:

```text
Número de capas: 12
Capa 0: (1, 12, 44, 44)
...
Capa 11: (1, 12, 44, 44)
```

Cada componente significa algo distinto:

- **12 capas**, que es la longitud de la tupla, no parte de la forma. Corresponde a `num_hidden_layers` del `config.json`.
- **1**, el tamaño del lote. Se procesa una oración a la vez, así que siempre vale 1. Es la dimensión que se indexa con el `0` en `attentions[layer][0, head, token_index]`.
- **12**, las cabezas, de `num_attention_heads`. Cada capa produce doce distribuciones de atención independientes.
- **44 × 44**, la matriz propiamente dicha. Las filas son "desde qué token miro" y las columnas "hacia qué token miro". `n` incluye `[CLS]` y `[SEP]`.

Dos propiedades que hay que tener presentes al interpretar las tablas de las Partes C y D:

**Cada fila suma 1.** La atención se calcula con un softmax sobre la fila, así que es una distribución de probabilidad. Esto da un punto de referencia inmediato: si el reparto fuera uniforme, cada token recibiría `1/n`. Para `n = 44` eso son 0,023. Un peso solo es "alto" comparado con ese valor, y el valor cambia con la longitud de la oración.

**`n` varía mucho.** En este corpus va de 9 a 101 tokens, con una mediana de 35. La matriz crece de forma cuadrática: la oración más corta produce matrices de 81 celdas y la más larga de 10 201. Por eso comparar pesos crudos entre oraciones de distinta longitud no es directo.

Y una cifra que conviene tener en la cabeza: 12 capas por 12 cabezas son **144 matrices de atención por oración**. Cuando en la Parte C se inspecciona una capa y una cabeza, se está mirando una de esas 144.

## Parte C: análisis de atención

### `getTopAttention`

Es la función central del laboratorio. Hace cinco cosas en orden.

**Uno, tokenizar y recuperar los tokens como cadenas.**

```python
inputs = tokenizer(sentence, return_tensors="pt", add_special_tokens=True)
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
```

El `[0]` extrae el primer y único elemento del lote. La lista `tokens` es la que traduce índices a palabras legibles al armar las tablas, y por eso tiene que incluir los especiales: sus posiciones deben coincidir exactamente con los ejes de la matriz.

**Dos, ejecutar el modelo.** Igual que `getAttentions`, con `torch.no_grad()`.

**Tres, localizar la fila que se va a analizar.**

```python
occurrences = [
    index
    for index, token in enumerate(tokens)
    if token == target_token
]
```

Se construye la lista completa de posiciones donde aparece el token, y el parámetro `occurrence` elige cuál. Una palabra repetida no es un solo token para el modelo: cada aparición ocupa una fila distinta de la matriz y tiene su propia representación contextual. En `corpus[5]`, `Gregorio` está en los índices 8 y 28; el primero es objeto de `visitar a` y el segundo es sujeto de `escuchó`. Son dos cosas diferentes y hay que declarar cuál se examina.

Hay dos validaciones. Si la lista queda vacía, el token no existe en la oración y se lanza `ValueError` mostrando la lista completa, que es lo que permite ver si el problema es que la palabra se partió en subpalabras. Si se pide una ocurrencia que no existe, el error indica en qué índices sí aparece.

**Cuatro, extraer la fila.**

```python
weights = attentions[layer][0, head, token_index]
```

Esta línea concentra casi todo el contenido conceptual de la parte. Se lee de izquierda a derecha:

- `attentions[layer]` elige el tensor de una capa, con forma `(1, 12, n, n)`.
- `[0, ...]` elige el único elemento del lote.
- `[..., head, ...]` elige una de las doce cabezas. Queda una matriz `n × n`.
- `[..., token_index]` elige **una fila**. Queda un vector de longitud `n`.

Ese vector es la respuesta a "desde el token en la posición `token_index`, ¿cuánta atención va a cada token de la oración?". Suma 1.

Conviene notar que es la fila y no la columna. La columna respondería a la pregunta inversa, "quién me atiende a mí", y daría resultados distintos: la matriz no es simétrica.

**Cinco, ordenar y formatear.**

```python
top_values, top_indices = torch.topk(weights, k=min(top_n, len(tokens)))
```

`torch.topk` devuelve los `k` valores más altos y sus posiciones, ya ordenados de mayor a menor. El `min` protege el caso de una oración con menos de 5 tokens; en este corpus la más corta tiene 9, pero la protección no cuesta nada.

Después se arma una lista de filas `[ranking, token, índice, peso]`, traduciendo cada índice a su cadena mediante `tokens[index]`. El `enumerate(..., start=1)` es lo que produce un ranking que empieza en 1 en vez de en 0, porque va dirigido a un lector humano.

La función devuelve un diccionario:

```python
return {
    "token_index": token_index,
    "occurrences": occurrences,
    "rows": rows
}
```

Devolver las tres cosas y no solo las filas es lo que permite que la evidencia registre qué se analizó, y no solo el resultado.

### `printTopAttention`

Toma el diccionario y lo presenta. La cabecera incluye la oración, el token, sus ocurrencias, el índice analizado, la capa y la cabeza; luego la tabla con `tabulate`.

Las dos líneas de ocurrencias e índice son las que hacen la evidencia autocontenida. Sin ellas, una cabecera que dijera solo `Token analizado: Gregorio` sería ambigua en cualquier oración donde la palabra se repita, y el lector no tendría forma de saber qué fila se examinó, porque la columna `Índice` de la tabla se refiere a los tokens **atendidos**, no al analizado.

### `printAttentionAnalysis`

```python
for target_token in target_tokens:
    for layer in layers:
        for head in heads:
            printTopAttention(...)
```

Imprime una cabecera con la oración y su lista de tokens, y luego recorre el producto cartesiano de las tres listas. Con 2 tokens, 2 capas y 2 cabezas salen **8 tablas**, que son las que contiene `results/attention_analysis.txt`.

Ese diseño es lo que permite responder por separado dos de las preguntas de análisis: fijando el token y la cabeza y variando la capa se aísla el efecto de la profundidad; fijando el token y la capa y variando la cabeza se aísla el efecto de la especialización de cada cabeza.

### La celda de parámetros

```python
sentence = corpus[5]

target_tokens = ["madre", "Gregorio"]
layers = [2, 8]
heads = [0, 5]
```

Toda la configuración del experimento está aislada en una sola celda, separada de las funciones. Cumple el mínimo que fija la guía —al menos 2 capas, 2 cabezas y 2 tokens— y las elecciones no son arbitrarias:

- `madre` y `Gregorio` son tokens completos del vocabulario, no subpalabras, así que `getTopAttention` puede localizarlos. Además son núcleos semánticos de la oración: un sustantivo común de parentesco y un nombre propio.
- Las capas 2 y 8 están una en el primer tercio y otra en el último, lo bastante separadas como para que una diferencia de comportamiento sea visible.
- Las cabezas 0 y 5 son dos de las doce disponibles en cada capa.

## Parte D: comparación

### `compareAttention`

```python
def compareAttention(sentence_1, sentence_2, target_token, layer, head, tokenizer, model,
                     top_n, occurrence_1=0, occurrence_2=0):
```

Llama dos veces a `getTopAttention` manteniendo constantes el token, la capa y la cabeza, y variando únicamente la oración. Después imprime las dos tablas una debajo de otra, cada una con sus ocurrencias y su índice analizado.

Los parámetros `occurrence_1` y `occurrence_2` van separados porque son dos oraciones distintas: la misma palabra puede aparecer una vez en una y tres en la otra.

El diseño de la función es el que hace válida la comparación. Al fijar capa, cabeza y token, la única variable que cambia es el contexto, que es exactamente lo que la guía pide contrastar.

### Comparación 1: `Gregorio`, contraste de función sintáctica

```python
gregorio_sentence_1 = corpus[5]
gregorio_sentence_2 = corpus[21]
```

El contraste es de función sintáctica sobre la misma forma superficial:

| | Oración | Papel de `Gregorio` | Índice |
| --- | --- | --- | --- |
| 1 | `... visitar a Gregorio ...` | objeto, marcado por la preposición `a` | 8 |
| 2 | `Gregorio había bajado ...` | sujeto, en posición inicial | 1 |

La guía propone como ejemplo un par mínimo de ambigüedad léxica (`banco` como entidad financiera o como asiento). Ese tipo de par no aparece en el fragmento de Kafka, así que el contraste se construye sobre la otra variable disponible: la misma palabra, con el mismo significado, ocupando dos funciones gramaticales distintas.

Sale en `results/attention_comparison_gregorio.txt`.

### Comparación 2: `madre`, contraste de contexto léxico

```python
madre_sentence_1 = corpus[5]
madre_sentence_2 = corpus[19]
```

La segunda comparación mantiene **la misma capa y la misma cabeza** que la primera, la 8 y la 5. Lo único que cambia es el token desde el que se mira.

El motivo está en los resultados de la Parte C. En `corpus[5]`, con esa misma capa y cabeza, los dos tokens analizados se comportan de forma opuesta:

| Token que pregunta | Reparto principal |
| --- | --- |
| `Gregorio` | `[SEP]` 0,419 y `[CLS]` 0,335: más del 75 % en tokens especiales |
| `madre` | `padre` 0,374 y `hermana` 0,374: casi el 75 % en sustantivos de parentesco |

Es decir que el volcado sobre tokens especiales no es una propiedad de la cabeza, sino de la pareja cabeza más token: la cabeza busca algo que `madre` tiene en esa oración y `Gregorio` no.

La comparación 2 pone a prueba si ese comportamiento léxico sobrevive a un cambio de contexto:

| | Oración | Tokens | Otros parentescos |
| --- | --- | --- | --- |
| 1 | `La madre había querido visitar a Gregorio ... pero el padre y la hermana ...` | 44 | `padre`, `hermana` |
| 2 | `La madre acudió eufórica, pero se quedó muda al llegar a la puerta.` | 20 | ninguno |

El control es bueno: en las dos oraciones `madre` ocupa el índice 2, va precedida del determinante `La` y cumple la función de sujeto. Se mantienen constantes la posición, la función sintáctica, la capa, la cabeza y el token. La única variable libre es si el resto de la oración contiene o no otros sustantivos de parentesco.

Queda un factor sin controlar que conviene declarar al interpretar: las oraciones tienen distinta longitud, 44 frente a 20 tokens, de modo que el reparto uniforme de referencia pasa de 0,023 a 0,050. Un mismo valor absoluto pesa más en la oración corta.

Sale en `results/attention_comparison_madre.txt`.

## Sobre el costo de cómputo

Cada función es autocontenida: `getAttentions` y `getTopAttention` tokenizan y ejecutan el modelo por su cuenta, sin depender de resultados previos. Eso significa que las 8 tablas de la Parte C implican 8 pasadas hacia adelante sobre la misma oración, y que las dos comparaciones de la Parte D repiten la de `corpus[5]` dos veces más.

Es una redundancia deliberada, no un descuido: es lo que permite llamar a cualquiera de estas funciones de forma aislada, sin preparar estado previo. Con oraciones de hasta 101 tokens y un modelo de 12 capas en CPU, el costo es imperceptible. En un corpus grande convendría calcular las atenciones una vez y pasarlas como argumento.
