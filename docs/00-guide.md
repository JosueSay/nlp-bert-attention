# Laboratorio 2

## 1. Descripción

Analizarás oraciones complejas utilizando un modelo Transformer encoder-only, preferiblemente BERT. El objetivo es inspeccionar matrices de atención y estudiar cómo diferentes capas y cabezas distribuyen atención entre tokens.

La actividad no busca afirmar que los pesos de atención sean explicaciones causales completas. Busca desarrollar una lectura crítica de representaciones internas de un Transformer.

## 2. Objetivos

* Tokenizar texto con el tokenizer de BERT.
* Ejecutar un modelo Transformer con `output_attentions=True`.
* Identificar dimensiones de capas, cabezas y tokens.
* Extraer los tokens con mayor atención desde tokens seleccionados.
* Comparar patrones entre oraciones, capas y cabezas.
* Analizar límites de interpretar atención como explicación.

## 3. Corpus

La Metamorfosis Kafka

* 2 paginas consecutivas al azar.

## 4. Modelo Recomendado

```text
bert-base-multilingual-cased
```

Instalación:

```python
!pip install -q transformers torch pandas
```

## 5. Procedimiento

### Parte A: Tokenización

Carga el tokenizer y muestra los tokens de cada oración.

Debes identificar:

* Tokens especiales como `[CLS]` y `[SEP]`.
* Palabras divididas en subpalabras.
* Diferencias entre palabras lingüísticas y tokens del modelo.

### Parte B: Ejecución Del Modelo

Ejecuta el modelo solicitando atenciones:

```python
outputs = modelo(**entradas)
attentions = outputs.attentions
```

Reporta la forma de las matrices. La estructura esperada es aproximadamente:

```text
(batch_size, número_de_cabezas, número_de_tokens, número_de_tokens)
```

### Parte C: Análisis De Atención

Selecciona al menos:

* 2 capas distintas.
* 2 cabezas distintas.
* 2 tokens lingüísticamente relevantes por oración.

Para cada selección, muestra los 5 tokens con mayor peso de atención.

### Parte D: Comparación

Compara al menos dos oraciones. Explica si cambia la atención cuando cambia el contexto.

Ejemplo de contraste:

```text
El banco aprobó el préstamo.
Me senté en el banco del parque.
```

## 6. Preguntas De Análisis

Responde:

1. ¿Qué tokens reciben mayor atención desde cada token seleccionado?
2. ¿Cambian los patrones entre capas?
3. ¿Cambian los patrones entre cabezas?
4. ¿Las palabras con mayor atención son lingüísticamente relevantes?
5. ¿Qué diferencias aparecen entre oraciones simples y complejas?
6. ¿Qué ocurre cuando una palabra se divide en subpalabras?
7. ¿Qué no puedes concluir observando únicamente los pesos de atención?

## 7. Entregables

* Notebook `.ipynb` ejecutado.
* Corpus (Paginas seleccionadas)
* Evidencia de tokens y matrices inspeccionadas.
* Tablas con resultados de atención.
* Análisis escrito
