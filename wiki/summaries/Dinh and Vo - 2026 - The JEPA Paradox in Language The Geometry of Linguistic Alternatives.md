# The JEPA Paradox in Language: The Geometry of Linguistic Alternatives

## 0. Ficha

| Campo | Valor |
|---|---|
| Autores | Anh Trac Duc Dinh (VinUniversity CAIR / HCMUT), Khang Nhat Hoang Vo (MBZUAI); contribución igual |
| Venue y año | Preprint, "under review", arXiv 2026 (v1, 26 jul 2026) |
| Enlace (arXiv / DOI) | https://arxiv.org/abs/2607.23531 |
| Código | No publicado (el PDF no da URL) |
| Fecha de lectura | 2026-11-04 |

> Nota: el PDF se leyó como texto extraído (pypdf); las figuras (curvas de MI, rango, varianza, t-SNE y heatmaps) se interpretaron a partir del texto, que da los valores clave.

---

## 1. El núcleo

**X — ¿Qué intentan hacer y por qué es relevante?**
Explicar por qué la predicción latente determinista de estilo JEPA (MSE en espacio latente, sin término generativo) funciona en imagen, vídeo y audio pero no se ha convertido en un objetivo estándar para entrenar encoders de texto. [Abstract, §1]

**Y — ¿Por qué es difícil?**
Según los autores, el obstáculo es estadístico y no de capacidad ni de tuning. Un predictor con MSE aprende la media condicional. En texto, un span enmascarado admite varias continuaciones válidas, cuyos embeddings apuntan en direcciones distintas, así que la media condicional es un centroide que no corresponde a ninguna completación ("centroid degeneracy"). El encoder puede reducir la loss fusionando esas alternativas, es decir, colapsando. [§1, §3]

**Z — ¿Cuál es su contribución?**
- Introducen la noción de **conditional concentration** y tres condiciones necesarias para que JEPA aprenda bien: predictibilidad (P), no-collapse (NC) y baja varianza condicional (LV). [§4]
- Formalizan cómo fallan en texto con proposiciones sobre el techo de información (Data Processing Inequality), el collapse low-rank con loss cero y la descomposición sesgo-varianza. [Apéndice B]
- Diagnostican empíricamente una "T-JEPA" (I-JEPA trasladado a texto con BERT-Large) frente a I-JEPA. [§5]

**1a — ¿Cómo lo verifican experimentalmente?**
- **I-JEPA:** ViT-H/16 sobre 100K imágenes de ImageNet. **T-JEPA:** BERT-Large sobre 100K frases de C4. Mismo esquema de entrenamiento (15 épocas).
- **Métricas:** proxy de MI (InfoNCE), effective rank, coseno entre pares, gap de loss entre train y validación, y varianza condicional del target.
- **Secuencia observada:** el MI satura temprano (< 0.35 nats), la varianza se mantiene alta, en el paso 9,340 la loss diverge entre train y validación, y el effective rank de validación colapsa de 4.66 a 1.57 mientras el coseno se acerca a 1. [§5.2–5.4]
- **Robustez:** la secuencia se replica con 5 semillas de datos. [Apéndice D]
- **Downstream:** con encoders congelados y preentrenamiento en 3M frases, T-JEPA queda a nivel de azar en IMDB (50.02) y SNLI (33.34) y da 0 en retrieval, frente a 81.91 / 59.72 de BERT. [Tabla 2]

**1b — ¿Hay respaldo teórico?**
Sí, aunque las proposiciones son elementales: DPI, la descomposición sesgo-varianza del MSE, una cota inferior de la varianza con alternativas separadas ($\text{Var} \ge \alpha^2 m(m-1)\Delta^2/2$), la existencia de soluciones low-rank con loss cero, y que SIGReg (una restricción marginal) no garantiza propiedades condicionales. [Apéndices B, K]

**2 — Contribuciones que reclaman los autores**
- La *conditional concentration* como condición para que la predicción latente determinista dé buenas representaciones. [§1]
- La *centroid degeneracy* como modo de fallo del MSE latente con continuaciones ambiguas. [§1]
- Un diagnóstico empírico de T-JEPA (información, rango, coseno, varianza y downstream) que muestra que la saturación de información precede al collapse. [§1]

---

## 2. Trabajos relacionados (hermanos)

| Trabajo | Diferencia (supuestos / método) | ¿Comparado en experimentos? | Si no, ¿por qué? |
|---|---|---|---|
| I-JEPA | La referencia en imagen; mismo marco aplicado a parches | Sí, como contraste en los diagnósticos [§5.2–5.4] | — (sin evaluación downstream de I-JEPA) |
| BERT (MLM) | Objetivo distribucional sobre tokens | Sí, downstream [Tabla 2] | — |
| BYOL, VICReg, Barlow Twins | No contrastivos de dos vistas (masking o reemplazo) | Sí, downstream [Tabla 2] | — |
| LLM-JEPA | JEPA auxiliar sobre un LM autoregresivo, con vistas Text/Code | No; se discute que su ablation γ = 0 (0 %) apoya la tesis [Apéndice J] | El JEPA nunca se entrena solo |
| DLLM-JEPA | JEPA auxiliar sobre masked diffusion | No; se discute [Apéndice J] | Ídem; $\mathcal{L}_{\text{diff}}$ nunca se elimina |
| data2vec | Predicción latente de representaciones contextualizadas del teacher sobre texto enmascarado, sin término generativo | No; solo se cita como "important precursor" [§2] | No se justifica (ver §8) |
| SIGReg / LeJEPA | Regularizador anti-collapse marginal | No; solo análisis teórico [Apéndice K] | Queda como trabajo futuro |
| Text-JEPA (NL→FOL), TI-JEPA, VL-JEPA | JEPA con texto en tareas específicas o multimodales | No | Ámbito específico |

---

## 3. Antecedentes (ancestros)

**Conceptos previos necesarios**
- I-JEPA: context encoder, target EMA, predictor con queries posicionales y MSE con stop-grad. [§3, Eqs. 1–5]
- Collapse total y dimensional; effective rank como entropía espectral. [§4.2, Apéndice C]
- Cotas de MI con InfoNCE (Poole et al.). [§4.1]
- Entropía del inglés (Shannon). [Apéndice B.1.2]

**Definición del problema**
$$\mathcal{L}_{\text{JEPA}} = \mathbb{E}\Big[\tfrac{1}{|T|}\sum_{j\in T} \|g_\phi(f_\theta(x_C), p_j) - \text{sg}(f_{\bar\theta}(x)^{(j)})\|_2^2\Big]$$
El predictor óptimo es la media condicional $\mathbb{E}[z_T^{(j)} \mid z_C, p_j]$, y la loss se descompone en error reducible más $\mathbb{E}[\text{Var}(z_T^{(j)} \mid z_C, p_j)]$. [Eqs. 5, 6, 10]

**Supuestos inusuales o restrictivos**
- Para imágenes asumen que el encoder es Lipschitz en el manifold de datos y que los targets se concentran. [Apéndice B.1.1, Assumption 3]
- Para texto asumen que el target encoder separa por al menos Δ las continuaciones plausibles. [Assumption 1]
- El análisis se limita a JEPA *determinista* con MSE y como *único* objetivo.

**¿El escenario del problema es una contribución nueva?**
Sí: plantear por qué JEPA puro falla en texto, en términos de concentración condicional, es la contribución.

---

## 4. Método

**Qué hacen**
1. Definen las tres condiciones (P, NC, LV) y sus validaciones empíricas: proxy de MI, effective rank y coseno, y la varianza estimada con K = 16 completaciones. [§4]
2. Entrenan I-JEPA (ViT-H/16, máscaras por bloques) y T-JEPA (BERT-Large, predictor transformer de 6 capas, 1–5 spans de 1–5 tokens) con el mismo esquema. [§5.1]
3. Monitorizan las 5 métricas en train y validación para establecer el orden temporal de los fallos. [Tabla 1]
4. Evalúan downstream con encoders congelados frente a BERT y SSL de dos vistas. [§5.5]

**Por qué lo hacen así**
- **Esquema de entrenamiento:** warmup lineal de 10 épocas (2e-4 → 1e-3), clipping a 0.3 y weight decay de 0.04 → 0.4, "para que la dinámica de collapse sea observable". Con LR fijo o warmup corto, T-JEPA colapsaba antes de poder resolver la transición. [§5.1, Apéndice F]
- **Varianza en texto:** completaciones muestreadas con un MLM congelado como oráculo, ponderadas por su probabilidad. **En imagen:** perturbaciones gaussianas en píxeles (σ = 0.2). [Apéndice C.3]

---

## 5. Configuración experimental

**Instancia del problema**
- Diagnóstico: 100K imágenes de ImageNet-1K y 100K frases de C4 (90/10), seed 42; T-JEPA además con seeds 0–4. [§5.1, Apéndice D]
- Downstream: preentrenamiento en 3M frases de C4 con el tokenizer bert-base-uncased; evaluación en IMDB, SNLI (acc / F1), MTEB FEVER y MSMARCO (nDCG@10 / R@100); media ± std de 5 ejecuciones. [§5.5, Apéndice F]

**Detalles de implementación**
- I-JEPA: ViT-H/16 (d = 1280, 32 capas), EMA de 0.996, escala de encoder 0.85–1.0, escala de predictor 0.15–0.2 y 4 bloques target. [§5.1]
- T-JEPA (diagnóstico): BERT-Large (d = 1024, 24 capas), EMA de 0.996, predictor de 6 capas (d = 384, 8 cabezas), longitud máxima 256. [§5.1]
- Downstream: 10 épocas, AdamW con LR 1e-4, 100 pasos de warmup; T-JEPA con EMA de 0.9996; VICReg (25/25/1), Barlow (λ = 0.005), BYOL (0.996 → 1). [Apéndice F]
- Hardware y tiempo: no se aborda en el paper.

---

## 6. Resultados y discusión

**Resultados principales frente a baselines**

| Tarea / dataset | Métrica | T-JEPA | Mejor baseline |
|---|---|---|---|
| IMDB | Acc / F1 | 50.02 ± 0.13 / 66.68 ± 0.41 | BERT 81.91 / 81.89; mejor SSL: Barlow-M 64.34 / VICReg-M 68.85 [Tabla 2] |
| SNLI | Acc / F1 | 33.34 ± 0.12 / 16.69 ± 0.62 | BERT 59.72 / 59.11; Barlow-M 45.55 / 45.99 [Tabla 2] |
| MTEB FEVER | nDCG@10 / R@100 | 0 / 0 | BERT 14.43 / 38.82; todos los SSL de dos vistas también dan 0 [Tabla 2] |
| MTEB MSMARCO | nDCG@10 / R@100 | 0 / 0 | BERT 9.49 / 33.67 [Tabla 2] |
| Diagnóstico | effective rank (val) | 4.66 → 1.57 tras el paso 9,340 (train: 494–545) | I-JEPA ≈ 4.7–4.85, estable [§5.3] |
| Diagnóstico | varianza condicional (paso 9,340, val) | 0.0116 | I-JEPA 0.0059 [§5.4] |

**Rigor estadístico**
- Downstream: media ± std sobre 5 ejecuciones. [Tabla 2]
- Diagnósticos: un solo seed para la comparación con I-JEPA; 5 seeds de datos solo para T-JEPA. [Apéndice D]
- No hay tests de significancia, pero las diferencias downstream son enormes.

**¿Comparación justa?**
- Downstream: el mismo presupuesto y esquema de entrenamiento para todos ("no configuration advantage"). [Apéndice F]
- Diagnóstico: el esquema se eligió expresamente para hacer *visible* el collapse de T-JEPA, no para intentar evitarlo. [§5.1]
- I-JEPA y T-JEPA comparten esquema pero no backbone, datos ni estimador de varianza.

**Ablaciones**
- BYOL con masking colapsa igual que T-JEPA (IMDB 50.23, SNLI 34.28); BYOL con reemplazo funciona (61.77 / 43.99). Los autores lo interpretan como evidencia del papel de la ambigüedad. [Tabla 2, §5.5]
- No hay ablations de T-JEPA con mecanismos anti-collapse (VICReg, SIGReg, normalización de targets); SIGReg queda como predicción para trabajo futuro. [Apéndice K]

**Limitaciones que reconocen los autores**
- Los resultados no implican que el aprendizaje predictivo sea imposible en lenguaje; los objetivos compatibles con texto deben preservar varias completaciones. [Abstract, §1, §6]
- SIGReg resolvería NC pero no P ni LV; no lo han probado. [Apéndice K]
- La escala es pequeña (100K–3M frases); no hay una sección explícita de limitaciones.

---

## 7. Conclusión y trabajo futuro (descendientes)

**Trabajo futuro propuesto**
- Objetivos distribucionales, contrastivos, de mezcla (*mixture-valued predictors*) o a nivel semántico para JEPA en texto. [§1, §6]
- Entrenar T-JEPA con SIGReg; predicen que el rango se mantendría sano pero el MI y el downstream seguirían mal. [Apéndice K]

**Preguntas abiertas**
- Qué distinciones lingüísticas debe preservar la representación; el objetivo con MSE no tiene forma de decidirlo. [Apéndice B.2.4]

---

## 8. Lectura crítica

> Esta sección es la valoración del lector, no lo que dice el paper.

**¿Contribuciones claramente separadas del trabajo previo?**
Sí respecto a LLM-JEPA y DLLM-JEPA: el Apéndice J es un análisis útil y bien argumentado de por qué ambos funcionan (anclaje generativo). La parte teórica es en gran medida conocida (el MSE da la media condicional y la DPI); la novedad es el encuadre ("conditional concentration") y el protocolo de diagnóstico.

**Afirmaciones sin respaldo o lenguaje grandilocuente**
- **"T-JEPA fails not from insufficient capacity or tuning"** [§1]: no hay ningún intento de tuning para que funcione. Faltan los ingredientes estándar de data2vec (targets promediados en varias capas con instance/layer norm), un regularizador de varianza o SIGReg, y una búsqueda de EMA y LR. El esquema se eligió para *observar* el collapse. La afirmación excede la evidencia.
- **Contraejemplo de data2vec, ignorado:** data2vec entrena un encoder de texto con predicción latente determinista de los tokens enmascarados (Smooth L1, sin término generativo) y obtiene en GLUE resultados comparables o algo mejores que RoBERTa. Según la tesis del paper, debería sufrir la misma *centroid degeneracy*. El paper lo cita como "important precursor" sin reconciliarlo. Es la objeción más seria a la generalidad del argumento: sugiere que el fallo de T-JEPA se debe más a la receta (target de la última capa sin normalizar, poco dato, esquema de entrenamiento) que a una imposibilidad estadística.
- **Effective rank de I-JEPA ≈ 4.7–4.85** con d = 1280: es un valor bajísimo, propio de una representación casi colapsada, y en cambio se presenta como "non-degenerate". Esto pone en duda la escala o el cálculo de la métrica, o indica que I-JEPA tampoco está aprendiendo bien con 100K imágenes y 15 épocas.
- **Estimadores de varianza no comparables:** en imagen, ruido gaussiano en píxeles; en texto, completaciones semánticamente distintas de un MLM. El gap de la Fig. 4 (≈2×) puede deberse al estimador y no a la modalidad.
- **"Necessary conditions":** las condiciones P, NC y LV se presentan como necesarias, pero no se demuestra que lo sean, solo que su violación encarece el MSE.

**Limitaciones no mencionadas por los autores**
- **Escala diminuta:** BERT-Large sobre 100K frases y un BERT de referencia débil (IMDB 82 % con features congeladas). Los SSL de dos vistas dan 0 en retrieval, lo que sugiere que el protocolo es duro para cualquier método no-MLM.
- **I-JEPA sin evaluación downstream:** no se demuestra que el lado "que funciona" funcione en este setup.
- **Ambigüedad y nivel del target:** la ambigüedad depende del nivel del target. Si el target es una representación contextualizada (que ya "sabe" el token real) o un embedding a nivel de frase/span semántico, la concentración condicional puede ser mucho mayor. El análisis lo menciona como alternativa, pero no lo prueba.

**¿Es coherente la historia X → Y → Z → 1?**
Internamente sí: la hipótesis da un orden temporal de fallos y el experimento lo reproduce. Pero el paso 1 → generalidad ("deterministic JEPA es inadecuado para texto") no está justificado sin explicar data2vec y sin intentar ningún mecanismo anti-collapse estándar.

**Relevancia para jepa-llm**
Muy alta como marco crítico y como lista de riesgos:
1. Un JEPA puro con MSE sobre tokens enmascarados probablemente colapsa sin un anclaje (MLM o NTP) o un regularizador.
2. Conviene monitorizar el proxy de MI, el effective rank, el coseno y la varianza condicional durante el entrenamiento; su protocolo es reutilizable.
3. Vías prometedoras: predictores distribucionales o de mezcla, targets a nivel de span o frase, SIGReg más predictor multimodal.
4. Un experimento natural para el proyecto: reproducir T-JEPA con la receta de data2vec y SIGReg para contrastar su tesis.

**TL;DR propio**
El paper argumenta que JEPA puro con MSE falla en texto porque los spans enmascarados tienen varias completaciones válidas: el predictor aprende un centroide que no corresponde a ninguna y el encoder colapsa para reducir la varianza. Lo ilustra con una T-JEPA que colapsa y da resultados de azar. El diagnóstico es valioso y reutilizable, pero la generalización es excesiva: no intenta ningún mecanismo anti-collapse ni explica por qué data2vec sí funciona en texto. Lectura obligatoria para el proyecto, aunque con lectura crítica.
