# 2026-11-19 — Propuesta: JEPA para resúmenes de documentos largos

Recoge el estado del estudio con LLM-JEPA en NL-RX-Synth y la propuesta de llevar el regularizador a resúmenes de documentos largos, extractivos y abstractivos. Nada de lo que sigue sobre resúmenes está probado todavía: son hipótesis y un plan de trabajo.

## 1. Estado actual: LLM-JEPA en NL-RX-Synth

Detalle completo en [`experiments/2026-11-05-llm-jepa-nl-rx-synth.md`](../experiments/2026-11-05-llm-jepa-nl-rx-synth.md).

- **Montaje.** Llama-3.2-1B-Instruct con LoRA (r = 32), comparando fine-tuning normal (NTP) con NTP + pérdida JEPA (λ = 1, k = 1 `[PRED]`, coseno). Vista Text = descripción + `[PRED]`, vista Code = regex.
- **Resultado.** No hay mejora significativa: 87.4 % frente a 86.8 % de exact match en test (McNemar p = 0.26). JEPA entrena un ~53 % más lento.
- **Collapse.** La distancia JEPA llega a ~0 porque todos los embeddings apuntan en la misma dirección (coseno 0.9999 entre ejemplos distintos), no porque alinee los pares.
- **Baseline.** El nuestro es mucho más fuerte que el del paper (86.8 % frente a 57.3 %), en línea con el 88.9 % de Sengupta. Parte de la ganancia que reporta LLM-JEPA puede venir de un baseline poco entrenado.

**Conclusión.** Con la formulación del paper no hay ganancia. Antes de cambiar de tarea hay que conseguir una variante que no colapse: mean pooling (ya implementado), stop-gradient o EMA en la vista target, o un término de varianza/covarianza tipo VICReg. Se valida con `nlrx-screen` y `nlrx-diagnose` (coseno centrado y top-1 Text→Code).

## 2. ¿Por qué resúmenes?

- **Las vistas salen solas.** Text = documento + `[PRED]`, Code = resumen de referencia. Es la misma estructura de pares que en NL-RX.
- **Resumir admite muchas respuestas válidas.** La NTP penaliza cada diferencia de superficie respecto a la referencia. Predecir en el espacio latente puede premiar el contenido y no la forma, que es el argumento central de JEPA. En NL-RX casi no se aplica porque la regex es prácticamente única.
- **`[PRED]` como plan latente.** Obliga al modelo a condensar *qué va a decir* el resumen antes de generarlo, y eso podría ayudar a la cobertura y la fidelidad, no solo a la fluidez.

## 3. ¿Por qué documentos largos en particular?

- **Los fallos típicos son globales:** cobertura incompleta, sesgo de posición (se usa el principio y el final del documento y se ignora el medio) y alucinaciones. La NTP supervisa token a token. Predecir el latente del resumen completo obliga a agregar información de todo el documento.
- **El argumento de los "muchos resúmenes válidos"** pesa más con resúmenes de 200–500 tokens que con titulares o regex.
- **Riesgo: el cuello de botella.** Comprimir 4k–8k tokens en un único vector es mucho. Mitigaciones:
  - k > 1 tokens `[PRED]` (4–16);
  - comparar contra embeddings por segmento del resumen en lugar de un solo vector.

## 4. Abstractivo vs. extractivo

### Abstractivo

Es la traducción directa: JEPA como regularizador junto a la NTP, con el documento como vista Text y el resumen como vista Code.

### Extractivo

Aquí JEPA puede ser el **propio mecanismo de extracción**, no solo un regularizador:

1. Cada frase del documento se codifica sola, igual que la vista Code.
2. El `[PRED]` del documento predice el latente del resumen.
3. En inferencia se rankean las frases por cos(`[PRED]`, frase_i) y se seleccionan las top-k. No hay generación, así que es muy rápido.

Para que entrenamiento e inferencia vivan en el mismo espacio, el target debería ser el **oracle extractivo**: las frases elegidas de forma greedy por ROUGE frente a la referencia, que es el estándar. Si el target es el resumen abstractivo, se entrena contra un espacio y se evalúa contra otro.

La alternativa clásica es que el LLM genere los índices de las frases. Ahí JEPA volvería a ser un regularizador, igual que en el abstractivo.

### Ventaja de hacer ambos

El mismo modelo da las dos salidas: el `[PRED]` para extraer y la generación para abstraer. Además, la calidad extractiva sirve de **diagnóstico directo** de si el latente predicho contiene información útil, independiente de la calidad de generación.

## 5. Diseño de entrenamiento eficiente para contextos largos

### Problema del diseño actual

Ahora mismo `compute_losses` (`src/jepa_llm/model.py`) hace dos forwards: uno NTP (prompt + respuesta) y otro JEPA empaquetado `[Text + PRED | Code]` con máscara 4D causal por bloques. Con documentos largos esto tiene dos problemas:

1. **El documento se procesa dos veces**, así que el coste se dobla.
2. **La máscara 4D densa impide FlashAttention.** SDPA cae a un kernel que materializa la matriz L×L: con 8k tokens son 64M entradas por cabeza y capa. Esto domina la memoria, más que el modelo.

### Propuesta: un solo forward largo con máscara causal normal

```
[documento] [PRED]×k [resumen]   → NTP sobre el resumen + hidden states de los [PRED] (predictor)
[resumen]                        → target JEPA (forward corto, opcionalmente con stop-gradient)
```

- La NTP y el predictor salen del mismo pase.
- El target es un forward solo del resumen, que es corto (200–500 tokens) y por tanto barato. Con stop-gradient es aún más barato y además ayuda contra el collapse.
- El coste total es aproximadamente el del baseline más un forward corto, y permite usar `attn_implementation="flash_attention_2"` o SDPA flash.
- **Efecto secundario.** Los tokens del resumen atienden a los `[PRED]`, así que en inferencia también hay que añadirlos. Encaja con la idea del plan latente, pero obliga a añadir un baseline **NTP + `[PRED]` sin pérdida JEPA**. Sin él no se puede separar el efecto de JEPA del de tener tokens extra.
- Si hiciera falta mantener vistas separadas, la alternativa es empaquetar con FlashAttention varlen (position ids que vuelven a 0) en lugar de una máscara 4D densa.

Esto implica generalizar `batching.py` y `model.py`, que ahora están atados a NL-RX (`NLRXConfig`, `NLRXTokenizer`).

## 6. Hardware

Llama-3.2-1B + LoRA, bf16, gradient checkpointing, FlashAttention, batch 1 con gradient accumulation:

| GPU | Uso previsto |
|---|---|
| **RTX 4090 (24 GB)** | Principal. Documentos de 8k tokens sin problema y probablemente 16k. Es la más rápida, porque Ada tiene más throughput en bf16 que Ampere |
| **A40 (48 GB)** | Solo si hacen falta 16k–32k tokens o Llama-3.2-3B. Más memoria pero más lenta por paso |
| RTX 4070 SUPER (12 GB) | Desarrollo y tests con contextos cortos |

Estas cifras son estimaciones. El primer paso es un **benchmark de tokens/s y memoria pico** con el diseño de un solo forward, para calcular la duración de cada run antes de lanzar nada.

## 7. Datos

| Dataset | Documento | Resumen | Uso |
|---|---|---|---|
| **PubMed** | ~3–4k tokens | ~200–300 (abstract) | Piloto: subset de ~10k documentos truncados a 4k |
| **arXiv** | ~6k+ tokens | ~200–300 (abstract) | Escalado a 8k si el piloto da señal |
| **GovReport** | ~9k+ tokens | ~600 | Segunda fase, documentos largos de verdad |
| BookSum, QMSum | muy largo / diálogo | — | Fuera de alcance por ahora |

Para el extractivo hay que precomputar el oracle greedy por ROUGE de cada documento.

## 8. Evaluación

**Abstractivo**
- ROUGE-1/2/L. Es el estándar, pero mide superficie, justo lo que JEPA intenta no optimizar.
- BERTScore.
- Factualidad: SummaC o AlignScore.
- **Cobertura por posición:** de qué parte del documento salen los hechos del resumen. Es donde se esperaría ver el efecto de JEPA.
- Opcionalmente, un juez LLM sobre una muestra.

**Extractivo**
- ROUGE contra la referencia.
- Recall y precision frente al oracle.

**Comparaciones**

| Variante | Qué aísla |
|---|---|
| NTP | Baseline |
| NTP + `[PRED]` sin pérdida JEPA | El efecto de los tokens extra |
| NTP + JEPA | El efecto de la pérdida JEPA |

Al menos 3 semillas por variante y test pareados. En NL-RX el margen fue pequeño y con una sola ejecución no se puede concluir nada.

## 9. Riesgos

- **Collapse.** Sin resolverlo en NL-RX, en resúmenes ocurriría lo mismo.
- **Baseline fuerte.** La NTP con LoRA ya resume razonablemente, así que la ganancia esperable es pequeña. Lo mismo pasó en NL-RX.
- **Capacidad del modelo.** Un 1B puede quedarse corto en abstractivo de documentos largos. Si los resúmenes del baseline son flojos, el ruido tapa cualquier efecto. En ese caso, pasar a 3B en la A40 antes de concluir.
- **Cuello de botella del latente** con documentos muy largos (ver §3).
- **Métricas de superficie.** Si solo se mira ROUGE, una mejora real en contenido puede no verse.

## 10. Plan

1. **Resolver el collapse en NL-RX.** Es barato (screens de minutos). Criterio: coseno centrado y top-1 Text→Code claramente por encima del baseline.
2. **Refactor.** Generalizar el paquete más allá de NL-RX e implementar el diseño de un solo forward (§5).
3. **Benchmark** de memoria y tokens/s en la 4090 con 4k y 8k tokens.
4. **Piloto en PubMed** (~10k documentos, 4k tokens): las tres variantes de §8 con 3 semillas, abstractivo y extractivo.
5. **Si hay señal:** k > 1 `[PRED]`, targets por segmento, arXiv/GovReport a 8k+, y vistas formadas por dos resúmenes distintos del mismo documento en datasets con varias referencias.

## Preguntas abiertas

- ¿Qué datasets propios de los colaboradores podrían usarse?
- ¿Target del extractivo: oracle por ROUGE o resumen abstractivo con un término contrastivo?
- ¿Cuántos `[PRED]` hacen falta en función de la longitud del documento?
- ¿Mantener 1B por velocidad o ir directamente a 3B?
