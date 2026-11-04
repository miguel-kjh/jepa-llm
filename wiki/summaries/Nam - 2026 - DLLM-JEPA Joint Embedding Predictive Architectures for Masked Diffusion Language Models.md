# DLLM-JEPA: Joint Embedding Predictive Architectures for Masked Diffusion Language Models

## 0. Ficha

| Campo | Valor |
|---|---|
| Autores | Sangdae Nam (VESSL AI) |
| Venue y año | Preprint, arXiv 2026 (v1, 24 may 2026; el PDF pone "Preprint. June 2, 2026") |
| Enlace (arXiv / DOI) | https://arxiv.org/abs/2606.00091 |
| Código | No publicado ("will be released… upon publication") |
| Fecha de lectura | 2026-11-04 |

> Nota: el PDF se leyó como texto extraído (pypdf); las figuras (drift, KDE 3D, ablation) se interpretaron por sus etiquetas y valores.

---

## 1. El núcleo

**X — ¿Qué intentan hacer y por qué es relevante?**
Adaptar los objetivos JEPA a los *masked diffusion language models* (LLaDA, Dream). LLM-JEPA demostró que un término JEPA ayuda en LLMs autoregresivos, pero arrastra dos limitaciones: necesita datasets con pares de vistas explícitos y duplica el cómputo de entrenamiento. [§1]

**Y — ¿Por qué es difícil?**
- En LLMs autoregresivos, la atención causal obliga a usar una máscara por bloques y dos forwards con gradiente para obtener embeddings independientes de dos vistas (≈ +100 % de FLOPs). [§1, Tabla 1]
- En texto no hay un análogo natural a la data augmentation que genere vistas sin pares. [§1]
- Un objetivo con solo coseno puede colapsar a una solución constante. [§3.2]

**Z — ¿Cuál es su contribución?**
Aprovechar el *noise schedule* de difusión para generar las dos vistas a partir de la misma entrada:
- **Contexto:** $x_{t_L}$, con el 20 % de los tokens enmascarados.
- **Target:** $x_{t_H}$, con el 70 % enmascarado.

El backbone procesa el contexto en un único forward con gradiente, que da a la vez los logits de difusión y un embedding pooled. El target lo codifica una copia EMA sin gradiente. Un predictor de *k* capas transformer mapea el embedding del contexto al del target, con loss coseno. El coste es de 4F por paso, frente a 6F de LLM-JEPA (−33 %). [§3.2, §3.3, Tabla 1]

**1a — ¿Cómo lo verifican experimentalmente?**
- Fine-tuning de LLaDA-8B y Dream-7B en GSM8K, NL-RX-SYNTH, Spider y Django, comparado con fine-tuning solo de difusión.
- Titular: en LLaDA-8B GSM8K 4-shot pasa de 42.61 a 61.33 (+18.73) con el mejor (λ, k) y un solo seed; la media de 3 seeds es +2.98 (de 55.14 ± 8.90 a 58.12 ± 3.91). [Tablas 2, 10]
- Preservación de capacidades: loss de difusión en Wikitext-103 y MMLU (500 preguntas). [Tablas 3, 4]
- Análisis de drift por capa y ablations de vistas y predictor. [§5, Figs. 2–3]

**1b — ¿Hay respaldo teórico?**
No se aborda en el paper. Solo hay diagnósticos empíricos de collapse (rango efectivo, desviación estándar, diversidad de coseno). [Apéndice A.7, Tabla 13]

**2 — Contribuciones que reclaman los autores**
- Una formulación JEPA arquitectónicamente eficiente para masked-diffusion LMs: 33 % menos FLOPs que LLM-JEPA, sin máscara custom y aplicable a cualquier dataset de texto. [§1]
- Un patrón empírico consistente, la "disociación drift geométrico–funcional": más drift de las representaciones respecto al modelo base pero menos olvido funcional, concentrado en las capas medias. [§1, §5.1]
- Ganancia en la tarea y preservación del modelo base a la vez, frente a un ancla L2 en parámetros. [§1, §4.3]
- Mejoras en las 4 tareas × 2 backbones. [§1, Tabla 2]

---

## 2. Trabajos relacionados (hermanos)

| Trabajo | Diferencia (supuestos / método) | ¿Comparado en experimentos? | Si no, ¿por qué? |
|---|---|---|---|
| LLM-JEPA (Huang et al., 2025) | AR con atención causal, vistas Text/Code explícitas, 2 forwards con gradiente, sin EMA | No (solo números de referencia no emparejados) [Apéndice A.8, Tabla 14] | Sustratos distintos (causal frente a bidireccional); una comparación emparejada confundiría sustrato y objetivo [§6 L1] |
| I-JEPA / V-JEPA | JEPA en imagen y vídeo con EMA y predictor; inspiración del diseño EMA + predictor + coseno | No | Otra modalidad |
| BYOL / SimSiam | Stop-gradient + EMA/predictor contra el collapse | No (la ablation "no predictor" equivale a consistencia cosine con stop-grad) [§5.2] | — |
| LLaDA, MDLM, SEDD | Masked diffusion LMs; LLaDA y Dream son los backbones | Sí, como baseline: fine-tuning solo de difusión | — |
| EWC / anclas en espacio de parámetros | Regularización contra el olvido | Sí, un ancla L2 al modelo base [Tabla 3] | — |

---

## 3. Antecedentes (ancestros)

**Conceptos previos necesarios**
- Proceso forward de masked diffusion: cada token se enmascara con probabilidad *t*; la loss es la cross-entropy sobre las posiciones enmascaradas. [§3.1, Eqs. 1–2]
- JEPA con EMA target, stop-gradient y predictor (I-JEPA, BYOL). [§3.2]
- Catastrophic forgetting. [§4.3]

**Definición del problema**
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{diff}} + \lambda \big(1 - \cos(\text{sg}(z_{t_H}), g_\phi(z_{t_L}))\big)$$
donde:
- $z_{t_L} = \text{LN}(\text{MeanPool}(f_\theta(x_{t_L})))$ sobre los tokens no enmascarados ni de padding;
- $z_{t_H}$ es lo mismo con el encoder EMA $f_{\theta'}$ ($\tau = 0.996$);
- $g_\phi$ son *k* capas transformer. [Eqs. 3–7]

**Supuestos inusuales o restrictivos**
- Hace falta un backbone de difusión bidireccional ya preentrenado.
- El target se construye con la vista *más* enmascarada (70 %): se predice el embedding de una vista con menos información a partir de una con más, lo contrario de la intuición habitual de I-JEPA, donde el target ve más. [§3.2]
- El embedding es un vector pooled por secuencia, no un vector por token.

**¿El escenario del problema es una contribución nueva?**
Sí: aplicar JEPA a masked diffusion LMs es el escenario que plantean. [§1]

---

## 4. Método

**Qué hacen**
1. Dada $x_0$, muestrean $x_{t_L}$ ($t_L = 0.2$) y $x_{t_H}$ ($t_H = 0.7$); en la configuración Wide-t usan (0.1, 0.9). [§3.2, §4.1]
2. Forward con gradiente sobre $x_{t_L}$: logits para $\mathcal{L}_{\text{diff}}$ y embedding $z_{t_L}$ (mean pool + LayerNorm). [§3.2]
3. Forward sin gradiente del EMA sobre $x_{t_H}$ para obtener $z_{t_H}$. [§3.2]
4. Predictor $g_\phi$ (k ∈ {1..5} capas transformer, inicialización aleatoria) y loss $1 - \cos$. [§3.2, Apéndice A.10]
5. Loss total $\mathcal{L}_{\text{diff}} + \lambda\,\mathcal{L}_{\text{JEPA}}$, con λ ∈ {0.5, 1, 2}; el EMA se actualiza en cada paso. [§3.2]

**Por qué lo hacen así**
- **Vistas del noise schedule:** evitan necesitar pares y reutilizan el forward de difusión. [§3.2]
- **EMA + stop-grad + predictor:** dan un target estable y evitan el collapse, junto con $\mathcal{L}_{\text{diff}}$. [§3.2]
- **Vistas asimétricas:** con vistas simétricas (0.2, 0.2) el resultado es peor que el baseline. [§5.2, Fig. 3]
- **Predictor:** sin él se mantiene parte de la ganancia 0-shot pero desaparece la 4-shot (−16.9 pp). [§5.2]
- **Predictor de capas transformer:** "por consistencia arquitectónica con LLM-JEPA". [§3.2]

---

## 5. Configuración experimental

**Instancia del problema**
- GSM8K (7,473; accuracy), NL-RX-SYNTH (10K; exact y functional match contra 200 strings), Spider (7K; exact y execution match en validation), Django (18,805; prefix match normalizado). [§4.1]
- Preservación: loss de difusión en Wikitext-103 val (50 % de masking) y MMLU (500 preguntas estratificadas). [§4.3]

**Detalles de implementación**
- LLaDA-8B (32 capas, d = 4096) y Dream-7B (28 capas, d = 3584), checkpoints públicos de HF. [Apéndice A.10]
- Fine-tuning completo de 2 épocas en 8× A100 80GB, AdamW, cosine con 5 % de warmup, batch efectivo 32, DeepSpeed ZeRO-2. [§4.1, Apéndice A.10]
- Dos configuraciones: *aggressive* (LR 1e-5, (0.2, 0.7)) y *Wide-t* (LR 1.4e-6, (0.1, 0.9)). [Apéndice A.10]
- Evaluación con unmasking iterativo de 128 pasos, greedy. [Apéndice A.10]
- Seeds {42, 123, 777}; ≈800 A100-horas en total; ≈45 min por run de GSM8K. [Apéndice A.10]

---

## 6. Resultados y discusión

**Resultados principales frente a baselines** (el baseline es fine-tuning solo de difusión en el mismo backbone)

| Tarea / dataset | Métrica | Método | Mejor baseline |
|---|---|---|---|
| GSM8K 4-shot, LLaDA (best (λ, k), 1 seed) | acc. | 61.33 | 42.61 [Tabla 2] |
| GSM8K 4-shot, LLaDA (3 seeds) | acc. | 58.12 ± 3.91 | 55.14 ± 8.90 [Tabla 10] |
| GSM8K 0-shot Wide-t, LLaDA (3 seeds) | acc. | 67.07 ± 0.41 | 65.23 ± 0.93; ancla L2 65.18 ± 0.87 [Tabla 3] |
| GSM8K 4-shot, Dream | acc. | 46.25 (1 seed) / 40.21 ± 4.34 (3 seeds) | 34.87 / 37.65 ± 3.77 [Tablas 2, 10] |
| NL-RX 4-shot func, LLaDA | match | 58.20 (1 seed) / 48.52 ± 4.55 (3 seeds) | 47.50 / 45.93 ± 2.70 [Tablas 2, 10] |
| Spider exec (cleaned), LLaDA / Dream | match | 39.36 / 25.15 | 35.40 / 20.89 (1 seed) [Tabla 2] |
| Django ws-prefix, LLaDA (3 seeds) | match | 74.83 ± 0.56 | 74.63 ± 0.16 [Tabla 10] |
| Wikitext Δloss frente al base (Wide-t) | Δ ↓ | −0.0017 | −0.0004 (1 seed) [Tabla 3] |
| MMLU (500 preguntas) | acc. | 57.53 ± 0.23 | 57.93 ± 0.42 (base 57.40) [Tabla 4] |

**Rigor estadístico**
- 3 seeds en algunas celdas, con desviación estándar poblacional [Tabla 10].
- La tabla principal (Tabla 2) usa el mejor (λ, k) por celda; Spider en LLaDA y Spider/NL-RX en Dream tienen un solo seed. [§4.2]
- Correlación drift–olvido: r = +0.94 (baseline) frente a +0.75 (JEPA); el test de Fisher da p = 0.16 (no significativo), y el autor lo reconoce. [Apéndice A.1]
- Las filas de Wikitext de baseline y JEPA tienen un solo seed. [Tabla 3]

**¿Comparación justa?**
- Mismo LR, mismo schedule y mismos seeds para baseline y JEPA. [Tabla 2]
- Pero el JEPA elige el mejor de 15 configuraciones (λ, k) por celda y el baseline ninguna. Con seed 42, el baseline de GSM8K (42.61) está en la cola baja de su dispersión (±8.9).
- La configuración fija (λ = 1, k = 3) da ganancias positivas pero menores. [Tabla 8]
- No se indica que la selección se haga en un conjunto separado del de test.

**Ablaciones**
- Vistas simétricas (0.2, 0.2): 0-shot 38.9 (por debajo del baseline 39.5); asimétricas sin predictor: 4-shot 44.9 frente a 61.3 con todo. [Fig. 3]
- (t_L, t_H): (0.2, 0.7) domina con claridad en 4-shot (61.33 frente a 42.76–46.40). [Tabla 9]
- Grid λ × k en GSM8K 0-shot: todas las configuraciones superan al baseline (0.402); la mejor, λ = 2, k = 4, da 0.488. [Tabla 7]
- Grid λ × k en NL-RX: varias configuraciones quedan por debajo del baseline (0.475), p. ej. 0.414 o 0.432. [Tabla 11]
- Django 0-shot: de +21.89 pp, 18.52 se deben a menos artefactos de parada y solo 3.38 a mejor contenido. [Tabla 5]
- Collapse: el rango efectivo (≈42–44) es igual al del base y al del baseline. [Tabla 13]

**Limitaciones que reconocen los autores**
- No hay comparación emparejada con LLM-JEPA (L1). [§6]
- La evidencia de preservación descansa en Wikitext con un solo seed superviviente; MMLU es solo un sanity check (L2). [§6]
- 3 seeds, efecto grande solo en GSM8K, un único backbone adicional, solo dos puntos de operación, 2 épocas a escala 7–8B y sin preentrenamiento (L3). [§6]

---

## 7. Conclusión y trabajo futuro (descendientes)

**Trabajo futuro propuesto**
- Aplicar DLLM-JEPA en preentrenamiento. [§7]
- Análisis teórico de la disociación. [§7]
- Scheduling adaptativo de λ. [§7]
- Una comparación con LLM-JEPA bajo un protocolo emparejado. [§7]

**Preguntas abiertas**
- Por qué el drift extra se concentra en las capas medias y solo en GSM8K. [§5.1]

---

## 8. Lectura crítica

> Esta sección es la valoración del lector, no lo que dice el paper.

**¿Contribuciones claramente separadas del trabajo previo?**
Sí. El paper delimita bien qué toma de LLM-JEPA, de I-JEPA y de BYOL, y su aportación concreta: vistas desde el noise schedule y el ahorro de cómputo. Es honesto sobre el alcance de la comparación (L1) y sobre la tabla de referencia no emparejada (Tabla 14).

**Afirmaciones sin respaldo o lenguaje grandilocuente**
- **El titular de +18.7 pp** [Abstract, §7] mezcla el mejor (λ, k) con un solo seed cuyo baseline cae en la cola baja. Con 3 seeds la mejora es +2.98 y entra dentro de la desviación del baseline (±8.9).
- **"Improves every (task, architecture) cell"** depende de elegir el mejor de 15 configuraciones; en NL-RX, muchas configuraciones empeoran. [Tabla 11]
- **El ahorro del 33 %** es frente a LLM-JEPA. Frente al propio baseline de difusión, DLLM-JEPA cuesta +33 % [Tabla 1], y no hay comparación a cómputo igualado con el baseline (p. ej. entrenarlo 1.33× más).
- **La "disociación"** se apoya en diferencias de Wikitext del orden de 1e-3 y en un test no significativo (p = 0.16). El autor lo matiza ("descriptive sense"), pero ocupa un lugar central en el abstract.
- **Django:** gran parte de la ganancia viene de terminar mejor la generación, no de mejor contenido. [Tabla 5]
- **Tabla 14:** etiqueta los números de LLM-JEPA como "3-seed", cuando LLM-JEPA usa 5 seeds.

**Limitaciones no mencionadas por los autores**
- **Dirección de la predicción:** el target es la vista más ruidosa (70 % de masking). Esto contradice el principio de I-JEPA (predecir información ausente en el contexto) y no se discute por qué funciona así.
- **Baseline con cómputo igualado:** no hay un baseline con el mismo cómputo extra ni con regularizadores alternativos baratos (p. ej. consistencia de logits entre niveles de ruido).
- **Rango efectivo bajo de partida:** ≈42 sobre 4096 dimensiones ya es muy bajo en el modelo base; que no cambie no descarta un collapse dimensional parcial.
- **Validación:** sin un conjunto de validación separado declarado para elegir (λ, k).

**¿Es coherente la historia X → Y → Z → 1?**
Sí en X → Y → Z: es una motivación clara y una solución elegante al problema de las vistas y del cómputo. El paso a 1 es más débil: las ganancias robustas (multi-seed) son de 1–3 pp, y el análisis mecanístico es descriptivo.

**Relevancia para jepa-llm**
Muy relevante como diseño. Muestra cómo generar vistas "gratis" con distintos niveles de masking en modelos bidireccionales, lo que es directamente aplicable a encoders BERT-style. También ofrece un diagnóstico útil de collapse y de olvido. Sugiere ablations obligatorias: vistas simétricas frente a asimétricas y con predictor frente a sin predictor.

**TL;DR propio**
DLLM-JEPA añade al fine-tuning de LLMs de difusión enmascarada un término JEPA, con un teacher EMA y un predictor, entre una vista poco enmascarada y otra muy enmascarada de la misma frase. No necesita datos emparejados y cuesta un tercio menos que LLM-JEPA. Las ganancias reales (3 seeds) son modestas, de +1 a +3 pp, y la cifra de +18.7 pp es el mejor caso con un solo seed. Merece la pena leerlo con atención por su diseño de vistas y sus ablations, no por sus números titulares.
