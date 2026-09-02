# Reflexión Laboratorio 3

- [Enlace a repositorio](https://github.com/JosueSay/nlp-bert-attention)

El corpus son dos páginas consecutivas de *La metamorfosis*, de Franz Kafka, en español: las páginas 22 y 23 del PDF proporcionado en el curso. El texto se conserva sin modificar en `data/corpus.txt` y se segmenta en 33 oraciones, de las cuales la última queda incompleta porque la frase continúa en la página siguiente. El modelo es `bert-base-multilingual-cased`, de 12 capas y 12 cabezas, cargado sin cabeza de tarea.

Los tokens analizados son *madre* y *Gregorio* sobre `corpus[5]`, de 44 tokens, en las capas 2 y 8 y las cabezas 0 y 5. Son ocho tablas en total y la evidencia está en `results/attention_analysis.txt`. Un reparto uniforme sobre 44 tokens daría 0,023 por token, que sirve como base de comparación.

## ¿Qué tokens reciben mayor atención desde cada token seleccionado?

Desde *madre*, en el índice 2.

| Capa | Cabeza | Tokens más atendidos |
| --- | --- | --- |
| 2 | 0 | *[CLS]* 0,120 - *[SEP]* 0,063 - *hermana* 0,054 - *La* 0,041 - *el* 0,035 |
| 2 | 5 | *[CLS]* 0,898 - *La* 0,045 - *madre* 0,025 - *Gregorio* 0,004 - *había* 0,004 |
| 8 | 0 | *y* 0,171 - *habían* 0,136 - *el* 0,125 - *la* 0,123 - *la* 0,066 |
| 8 | 5 | *padre* 0,374 - *hermana* 0,374 - *[SEP]* 0,085 - *madre* 0,071 - *[CLS]* 0,038 |

En la capa 2 domina *[CLS]*, de forma moderada en la cabeza 0 y extrema en la cabeza 5, donde un solo token especial concentra casi el 90 % de la fila. En la capa 8 los especiales dejan de encabezar. La cabeza 0 reparte de forma bastante plana entre determinantes y la conjunción *y*, mientras que la cabeza 5 vuelca el 74,8 % en *padre* y *hermana*, los otros dos sustantivos de parentesco de la oración, con pesos casi idénticos entre sí (0,374190 y 0,373846).

Desde *Gregorio*, en el índice 8 y primera de dos ocurrencias.

| Capa | Cabeza | Tokens más atendidos |
| --- | --- | --- |
| 2 | 0 | *[CLS]* 0,172 - *hermana* 0,066 - *Gregorio* 0,052 - *[SEP]* 0,039 - «,» 0,037 |
| 2 | 5 | *[CLS]* 0,364 - *visitar* 0,176 - *quer* 0,091 - *a* 0,080 - *madre* 0,067 |
| 8 | 0 | *[CLS]* 0,415 - *[SEP]* 0,360 - «.» 0,070 - *Gregorio* 0,038 - *Gregorio* 0,033 |
| 8 | 5 | *[SEP]* 0,419 - *[CLS]* 0,335 - *Gregorio* 0,146 - *Gregorio* 0,056 - «.» 0,029 |

Los tokens especiales encabezan las cuatro combinaciones y en la capa 8 lo hacen de forma abrumadora, con 77,6 % en la cabeza 0 y 75,4 % en la cabeza 5 solo entre *[CLS]* y *[SEP]*. La excepción está en la capa 2, cabeza 5. Tras *[CLS]*, los cuatro receptores restantes son *visitar*, el verbo que rige a *Gregorio* como objeto directo, *a*, la preposición que marca ese objeto, *quer*, primera pieza del verbo principal de la perífrasis, y *madre*, el sujeto de la cláusula. Es la estructura argumental completa a la que pertenece el token.

Los dos *Gregorio* que aparecen en la capa 8 son el propio token y su segunda mención, en el índice 28.

## ¿Cambian los patrones entre capas?

Sí, de forma acusada. Para aislar el efecto se mantienen fijos el token y la cabeza, y solo cambia la profundidad.

| Token | Cabeza | Capa 2 | Capa 8 |
| --- | --- | --- | --- |
| *madre* | 0 | *[CLS]* 0,120 - *[SEP]* 0,063 - *hermana* 0,054 | *y* 0,171 - *habían* 0,136 - *el* 0,125 |
| *madre* | 5 | *[CLS]* 0,898 - *La* 0,045 - *madre* 0,025 | *padre* 0,374 - *hermana* 0,374 - *[SEP]* 0,085 |
| *Gregorio* | 0 | *[CLS]* 0,172 - *hermana* 0,066 - *Gregorio* 0,052 | *[CLS]* 0,415 - *[SEP]* 0,360 - «.» 0,070 |
| *Gregorio* | 5 | *[CLS]* 0,364 - *visitar* 0,176 - *quer* 0,091 | *[SEP]* 0,419 - *[CLS]* 0,335 - *Gregorio* 0,146 |

La atención se concentra al aumentar la profundidad. Midiendo la masa de los cinco primeros puestos, en tres de las cuatro comparaciones el salto es grande, de 0,313 a 0,622, de 0,366 a 0,918 y de 0,779 a 0,985. La excepción es *madre* en la cabeza 5, que ya estaba en 0,976 y baja levemente a 0,942. En la capa 2 la atención tiende a ser difusa y en la capa 8 se juega casi entera en cinco posiciones.

El destino de esa concentración va en direcciones opuestas según el token. Con *madre* en la cabeza 5 hay una inversión completa, porque la capa 2 vuelca el 89,8 % en *[CLS]* y la capa 8 pone el 74,8 % en *padre* y *hermana*. Con *Gregorio* ocurre lo contrario, ya que la capa 2 reparte sobre la estructura argumental y la capa 8 descarga el 75,4 % en *[SEP]* y *[CLS]*.

La relación entre las dos menciones de *Gregorio* aparece solo en la capa 8, con 0,033 y 0,056 en sus dos cabezas. En la capa 2 no figura entre los cinco primeros puestos.

Esto respalda solo a medias la expectativa habitual de que las capas iniciales captan relaciones superficiales y las profundas información semántica. Se cumple con *madre*, donde la capa profunda encuentra los sustantivos de parentesco, y no se cumple con *Gregorio*, donde la combinación más informativa está en la capa 2.

## ¿Cambian los patrones entre cabezas?

Sí, e incluso de forma más sistemática que entre capas.

| Token | Capa | Cabeza 0 | Cabeza 5 |
| --- | --- | --- | --- |
| *madre* | 2 | difusa, máximo 0,120 | *[CLS]* 0,898 |
| *madre* | 8 | *y*, *habían*, *el*, *la*, máximo 0,171 | *padre* 0,374 - *hermana* 0,374 |
| *Gregorio* | 2 | *[CLS]* 0,172 - *hermana* 0,066 | *[CLS]* 0,364 - *visitar* 0,176 |
| *Gregorio* | 8 | *[CLS]* 0,415 - *[SEP]* 0,360 | *[SEP]* 0,419 - *[CLS]* 0,335 |

En las cuatro comparaciones la cabeza 5 concentra más masa que la cabeza 0, con 0,313 frente a 0,976, 0,622 frente a 0,942, 0,366 frente a 0,779 y 0,918 frente a 0,985. La cabeza 0 tiende a repartir y la cabeza 5 a apostar por pocas posiciones. El caso más claro es *madre* en la capa 2, donde con la cabeza 0 el máximo es 0,120 y con la cabeza 5 un solo token se lleva 0,898, siendo la misma capa, la misma entrada y el mismo token.

Concentrarse no significa hacer lo mismo. El destino elegido varía por completo, desde *[CLS]* hasta los sustantivos de parentesco o la estructura argumental. Esto encaja con la idea de que cada cabeza opera sobre un subespacio distinto de las 768 dimensiones, 64 por cabeza, y puede especializarse en un tipo de relación diferente. Hay una excepción, porque en *Gregorio* capa 8 las dos cabezas hacen prácticamente lo mismo y no siempre dos cabezas de la misma capa difieren.

El hallazgo más relevante matiza la noción misma de especialización, ya que una misma cabeza se comporta de forma distinta según el token que la consulta. En la capa 8, cabeza 5, sobre la misma oración, *madre* pone el 74,8 % en contenido léxico y *Gregorio* el 75,4 % en tokens especiales. La razón es que la *Query* la aporta el token que pregunta y la cabeza solo define el espacio donde se compara con las *Keys*. Por eso afirmar que esa cabeza detecta parentescos sería impreciso, y lo correcto es decir que los localiza cuando se la consulta desde *madre* y la oración los contiene. El comportamiento es una propiedad del par cabeza y token, no de la cabeza sola.

## ¿Las palabras con mayor atención son lingüísticamente relevantes?

Solo en parte, y depende de la cabeza y del token que consulta. Clasificando las 40 entradas de las ocho tablas por tipo de receptor.

| Categoría | Apariciones | Masa | Porcentaje |
| --- | --- | --- | --- |
| Tokens especiales | 12 de 40 | 3,310 | 56,1 % |
| Palabras de contenido | 8 de 40 | 1,207 | 20,5 % |
| Palabras funcionales | 10 de 40 | 0,827 | 14,0 % |
| El propio token | 5 de 40 | 0,333 | 5,6 % |
| Puntuación | 3 de 40 | 0,136 | 2,3 % |
| Otra mención del mismo token | 2 de 40 | 0,089 | 1,5 % |

En contra pesa que el 56,1 % de la masa va a *[CLS]* y *[SEP]*, que no son palabras y no significan nada, y que encabezan seis de las ocho combinaciones. Sumando puntuación y palabras funcionales, cerca del 72 % de lo que aparece en los cinco primeros puestos carece de contenido léxico. Tampoco se trata de una atención plana, ya que esos cinco puestos capturan en promedio el 73,8 % de cada fila frente al 11,4 % que daría un reparto uniforme. La distribución es unas seis veces más concentrada que el azar, pero sobre tokens sin significado.

A favor están los tres casos ya descritos, el parentesco de *madre* en capa 8 cabeza 5, la estructura argumental de *Gregorio* en capa 2 cabeza 5 y la correferencia entre sus dos menciones en la capa 8. Son 3 de las 8 combinaciones. La relevancia es además condicional, porque esa misma cabeza baja a 0,5 % en contenido y se repliega sobre sí misma con 0,672 cuando la oración no ofrece ningún sustantivo de parentesco.

El 5,6 % que un token dedica a sí mismo no debe contarse como irrelevante, porque la autoatención incluye la propia posición por diseño y conservar parte de la identidad del token es parte del mecanismo. En conjunto, la atención alta coincide con relaciones lingüísticas reales solo en algunas combinaciones de capa, cabeza y token. En la mayoría de las observadas el peso principal recae sobre posiciones sin contenido, lo que sugiere un comportamiento de descarga más que una relación interpretable.

## ¿Qué diferencias aparecen entre oraciones simples y complejas?

No se ejecutó un experimento que aislara la complejidad sintáctica, pero las dos comparaciones de la Parte D permiten razonar sobre ella. Ambas usan la capa 8 y la cabeza 5, y el desglose de la masa por tipo de receptor deja ver qué variable manda.

| Token | Oración | Tokens | Estructura | Otro contenido | Sí mismo | Especiales |
| --- | --- | --- | --- | --- | --- | --- |
| *madre* | `corpus[5]` | 44 | relativo y coordinación | 74,8 % | 7,1 % | 12,3 % |
| *madre* | `corpus[19]` | 20 | coordinada simple | 0,5 % | 67,2 % | 29,5 % |
| *Gregorio* | `corpus[5]` | 44 | relativo y coordinación | 5,6 % correferencia | 14,6 % | 75,4 % |
| *Gregorio* | `corpus[21]` | 38 | dos subordinadas | 0 % | 14,0 % | 77,6 % |

*Gregorio* funciona como control natural. Entre sus dos oraciones cambian la longitud, de 44 a 38 tokens, y la estructura, de una subordinada de relativo con coordinación a dos subordinadas completivas. El reparto apenas se mueve, con 75,4 % frente a 77,6 % en especiales y 14,6 % frente a 14,0 % en autoatención. La complejidad varió y el resultado no.

Lo que sí cambia el reparto es otra cosa. *Gregorio* es un nombre propio sin familia semántica en ninguna de las dos oraciones, y por eso se comporta igual en ambas. *madre* tiene a *padre* y *hermana* en `corpus[5]` y no tiene ningún otro sustantivo de parentesco en `corpus[19]`, y ahí el reparto se invierte por completo. La variable que predice el destino de la masa es la disponibilidad de un interlocutor léxico compatible, no el número de cláusulas.

Cada fila es un softmax que suma 1, de modo que la masa se conserva y solo puede ir a contenido léxico, a la propia posición o a los tokens especiales. Cuando falta el interlocutor, la masa no desaparece sino que se redistribuye entre los otros dos destinos. Con *madre*, la suma de autoatención y especiales pasa de 19,4 % a 96,7 %, que es casi exactamente lo que dejó de ir a *padre* y *hermana*. Con *Gregorio*, la masa que en `corpus[5]` iba a la segunda mención reaparece en `corpus[21]` sobre *[SEP]*, que sube de 0,419 a 0,580.

La complejidad sí tiene un efecto, aunque indirecto y de escala. En `results/attention_shapes.txt` la longitud del corpus va de 9 a 101 tokens con mediana de 35, y el reparto uniforme de referencia pasa de 0,111 a 0,010. Una oración larga o con varias cláusulas tiene más candidatos compitiendo por la misma masa unitaria, porque el modelo no asigna presupuesto extra por cláusula añadida. Eso predice pesos individuales más diluidos, no un cambio de destino, y es lo que muestran los datos. Un peso de 0,10 es del orden del azar en una oración de 9 tokens y diez veces el azar en una de 101, así que los pesos absolutos no son comparables entre oraciones de distinta longitud sin normalizar contra su propia base.

La conclusión razonable es que la complejidad sintáctica afecta la escala de los pesos y no su destino. Queda como inferencia y no como resultado medido, porque son dos pares de oraciones y una sola cabeza de las 144. Confirmarlo exigiría el mismo token objetivo en oraciones equiparadas en contenido léxico, variando solo longitud y subordinación, y repetirlo en varias cabezas.

## ¿Qué ocurre cuando una palabra se divide en subpalabras?

El tokenizador de BERT usa WordPiece, que intenta primero la palabra completa y, si no figura en el vocabulario, se queda con el prefijo más largo que sí figure y repite el procedimiento con el resto. Las piezas que continúan una palabra se marcan con *##*.

Ese prefijo indica posición y no importancia, ya que significa que la pieza no abre palabra y se une a la anterior sin espacio. La prueba está en los cortes, que no siguen fronteras morfológicas. *sábana* se parte en *s*, *##ában* y *##a*, y *creyó* en *c*, *##rey* y *##ó*. Si *##* marcara menor relevancia, la parte importante de *sábana* sería *s*.

Lo que ocurre es que la palabra deja de existir como unidad en la matriz. *enseguida* ocupa tres filas y tres columnas bajo *ens*, *##egu* e *##ida*, y ninguna de ellas es *enseguida*. La atención que recibe se reparte entre sus piezas, de modo que una palabra con peso apreciable en conjunto puede quedar fuera del top 5 si ninguna pieza alcanza el corte por separado, y compararla a nivel de palabra exigiría sumar las columnas de todas ellas. Tampoco puede usarse como token objetivo, porque `getTopAttention` busca coincidencia exacta y *querido* no existe como token. Por eso se analizaron *madre* y *Gregorio*, ambos completos en el vocabulario. El fenómeno no es marginal, ya que sobre este corpus 922 palabras producen 1401 tokens, una razón de 1,52.

Las piezas de continuación reciben además menos peso. En *Gregorio* sobre `corpus[21]`, la pieza *##a* entra al quinto puesto con 0,010 frente a un uniforme de 0,026, es decir por debajo del azar. En cambio *quer*, pieza inicial de *querido* y por tanto sin *##*, alcanza 0,091 en capa 2 cabeza 5, unas cuatro veces el uniforme. La penalización recae sobre las continuaciones y no sobre la palabra partida en su conjunto.

## ¿Qué no puedes concluir observando únicamente los pesos de atención?

No se puede concluir causalidad. Un peso alto indica una asociación dentro del mecanismo, no una influencia sobre la salida del modelo. Tres límites son propios del mecanismo y no se deducen de lo visto en las preguntas anteriores.

- La atención es solo la mitad del cálculo. La salida de una cabeza es la suma de los *Values* ponderada por los pesos, y mientras la *Key* decide cuánta atención recibe un token, el *Value* decide qué aporta. Un peso alto sobre un token cuyo *Value* contribuye poco tiene un efecto pequeño.
- Cada observación es una de 144, y entre capa y capa intervienen conexiones residuales, normalización y una red *feed-forward* que redistribuyen la información. No existe la atención del modelo en singular, y ninguna afirmación sobre lo que el modelo mira es válida sin especificar capa y cabeza.
- No hay ninguna predicción que explicar. Se cargó `AutoModel` sin cabeza de tarea, de modo que el modelo devuelve representaciones y no logits, y lo observado es la construcción de representaciones contextuales.

A esto se suma lo ya mostrado en las preguntas anteriores. Un peso alto puede ser descarga en *[CLS]* y *[SEP]* en lugar de relación, los hallazgos positivos arrastran variables confundidas y la muestra es de 2 capas de 12, 2 cabezas de 12, 2 tokens y 3 oraciones.

Atribuir influencia real exigiría intervenir la entrada con técnicas de XAI, es decir inteligencia artificial explicable, como la ablación de tokens, las perturbaciones léxicas, los gradientes e Integrated Gradients, o el análisis de importancia sobre un modelo con cabeza de tarea.
