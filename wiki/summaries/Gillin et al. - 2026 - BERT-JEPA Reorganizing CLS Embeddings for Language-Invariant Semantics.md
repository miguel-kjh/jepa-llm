# BERT-JEPA: Reorganizing CLS Embeddings for Language-Invariant Semantics

## 0. Ficha

| Campo | Valor |
|---|---|
| Autores | Taj Gillin, Adam Lalani, Kenneth Zhang, Marcel Mateos Salles (contribución igual; Brown University) |
| Venue y año | Preprint, arXiv 2026 (v1, 1 ene 2026) |
| Enlace (arXiv / DOI) | https://arxiv.org/abs/2601.00366 |
| Código | https://anonymous.4open.science/r/bert-jepa-translation-3EB8/README.md (repositorio anónimo) |
| Fecha de lectura | 2026-11-04 |

> Nota: el PDF se leyó como texto extraído (pypdf); las figuras (PCA, t-SNE, matriz G-XLT de MLQA, curvas de RankMe) no se vieron como imagen.

---

## 1. El núcleo

**X — ¿Qué intentan hacer y por qué es relevante?**
Hacer que el embedding [CLS] de modelos BERT-style represente el significado con independencia del idioma (un "thought space"): "El gato es rojo" y "The cat is red" deberían caer en el mismo punto. El [CLS] es muy usado en tareas downstream, pero captura mal la semántica de la frase. [§1]

**Y — ¿Por qué es difícil?**
- El espacio [CLS] de RoBERTa y XLM-RoBERTa está colapsado: cualquier par de frases tiene coseno de ≈0.99, estén relacionadas o no y sean del mismo idioma o no. [§4, Apéndice A.6, Tablas 4–5]
- Los modelos de lenguaje siguen codificando rasgos específicos de cada idioma. [§2]

**Z — ¿Cuál es su contribución?**
BEPA: fine-tuning de XLM-RoBERTa-base con $\mathcal{L}_{\text{MLM}} + \lambda\,\mathcal{L}_{\text{Alignment}}$. Se empaquetan dos frases con el mismo significado (traducciones) en una sola entrada para el MLM. Para la alineación se hacen dos forwards, cada uno con una de las frases enmascarada, y se alinean los dos [CLS] resultantes con un predictor identidad y una loss InfoNCE. [§3]

**1a — ¿Cómo lo verifican experimentalmente?**
- Geometría: coseno entre pares relacionados y no relacionados, PCA, t-SNE por idioma y RankMe. [§4, Apéndices A.7–A.8]
- XNLI zero-shot (fine-tuning en inglés): media de 0.744 (bilingual) frente a 0.728 de XLM-RoBERTa, con mejora en los 15 idiomas. [Tabla 1]
- MLQA (fine-tuning en SQuAD inglés y evaluación en 7×7 pares de idiomas): mejor transferencia cross-lingual. [Fig. 3, Tabla 10]
- GLUE: 88.64 frente a 88.07 (sin pérdida en inglés). [Tabla 8]

**1b — ¿Hay respaldo teórico?**
No se aborda en el paper.

**2 — Contribuciones que reclaman los autores**
- El fine-tuning BEPA reorganiza el espacio [CLS] hacia una estructura "semantic-first". [§1]
- La representación pasa de rango bajo a rango más completo según la PCA. [§1]
- La reorganización apenas cuesta rendimiento en inglés. [§1]
- Mejora el rendimiento en tareas multilingües. [§1]

---

## 2. Trabajos relacionados (hermanos)

| Trabajo | Diferencia (supuestos / método) | ¿Comparado en experimentos? | Si no, ¿por qué? |
|---|---|---|---|
| LLM-JEPA (Huang et al., 2025) | JEPA sobre decoders con pares Text/Code y tokens [PRED]; BEPA es su inspiración directa, adaptada a encoders y traducciones | No | No se justifica |
| XLM-RoBERTa / RoBERTa | Modelos base sin objetivo de alineación | Sí, en XNLI, MLQA y GLUE [Tablas 1, 8–10] | — |
| mBERT, XLM-15, XLM-R (números publicados) | Modelos multilingües de referencia | Sí, solo en MLQA y con números del paper original [Tabla 10] | — |
| Sentence-BERT (Reimers & Gurevych, 2019) | Fine-tuning siamés para embeddings de frase; documenta el problema del [CLS] | No | No se discute como baseline |
| I-JEPA, V-JEPA, V-JEPA 2 | JEPAs de visión y vídeo | No | Otra modalidad |
| SIGReg / LeJEPA (Balestriero & LeCun, 2025) | Regularización anti-collapse | No en el texto principal (queda como trabajo futuro) | Ver la inconsistencia señalada en §8 |

---

## 3. Antecedentes (ancestros)

**Conceptos previos necesarios**
- BERT y MLM; RoBERTa y XLM-RoBERTa (sin NSP ni TLM). [§2]
- JEPA (LeCun, 2022) y LLM-JEPA. [§2]
- InfoNCE. [§3.2]
- Métricas de embeddings: isotropía, t-SNE, RankMe (rango efectivo). [§2, Apéndice A.7]

**Definición del problema**
Dado un par de frases $(s_1, s_2)$ con el mismo significado (posiblemente en distintos idiomas):
$$\mathcal{L}_{\text{BEPA}} = \mathcal{L}_{\text{MLM}}([s_1; s_2]) + \lambda\, \mathcal{L}_{\text{Alignment}}\big(\text{CLS}(s_1), \text{CLS}(s_2)\big)$$
$\text{CLS}(s_i)$ es el [CLS] final obtenido enmascarando la otra frase, y $\mathcal{L}_{\text{Alignment}}$ es InfoNCE con la traducción como positivo y el resto del batch como negativos. [§3.2, Apéndice A.7]

**Supuestos inusuales o restrictivos**
- Requiere datos paralelos (traducciones) o al menos pares semánticamente equivalentes. [Apéndice A.3]
- Solo es fine-tuning sobre un modelo ya preentrenado; no hay preentrenamiento completo, por limitaciones de GPU. [Apéndice A.3]

**¿El escenario del problema es una contribución nueva?**
No. El objetivo (embeddings multilingües alineados) es conocido; lo nuevo es plantearlo como JEPA sobre el [CLS].

---

## 4. Método

**Qué hacen**
1. Parten de xlm-roberta-base. [§3.1]
2. Empaquetan dos frases en una entrada con dos [CLS] y un [SEP]: modo *monolingual* (las dos frases en el mismo idioma) o *bilingual* (pares en idiomas distintos con el mismo significado). [Apéndice A.5, Fig. 4]
3. Aplican MLM clásico al 15 % sobre la entrada empaquetada. [§3.1]
4. Hacen dos forwards adicionales, cada uno enmascarando por completo una de las frases, para obtener el [CLS] de cada frase por separado. [Fig. 5]
5. Usan un predictor identidad y alinean los dos [CLS] con InfoNCE; λ = 1. [§3.2, Apéndice A.3]

**Por qué lo hacen así**
- **MLM:** conserva las capacidades lingüísticas de BERT. [§3.1]
- **InfoNCE:** MSE y coseno "no combaten el collapse del [CLS]"; hacen falta negativos que separen. [§3.2, Apéndice A.5]
- **Predictor identidad:** es la elección más simple; otros predictores quedan para trabajo futuro. [§3.1]

---

## 5. Configuración experimental

**Instancia del problema**
- Datos de fine-tuning: Opus-100 (10K pares) más Eng-Swa-Dataset para suajili; ablations con Flores (≈2K pares, 5× épocas) y "Small" (2K de Opus, 5× épocas); WMT19 para el análisis de cosenos. [Apéndices A.3, A.9]
- Evaluación: XNLI (15 idiomas, zero-shot desde inglés, 2 épocas de fine-tuning), MLQA (7 idiomas de contexto × 7 de pregunta, fine-tuning en SQuAD v1.1) y GLUE (3 épocas por tarea, cabeza lineal). [§5]

**Detalles de implementación**
- 1× NVIDIA RTX A5500, 10 épocas, batch 16, longitud máxima 256, AdamW (LR 2e-5, weight decay 0.01, 500 pasos de warmup), λ = 1, masking del 15 %. [Apéndice A.3]

---

## 6. Resultados y discusión

**Resultados principales frente a baselines**

| Tarea / dataset | Métrica | Método | Mejor baseline |
|---|---|---|---|
| XNLI zero-shot, media de 15 idiomas | acc. | 0.744 (Bilingual-Opus) | 0.728 XLM-RoBERTa base [Tabla 1] |
| XNLI, sw (mayor ganancia) | acc. | 0.673 | 0.610 [Tabla 1] |
| XNLI, ur (menor ganancia) | acc. | 0.657 | 0.656 [Tabla 1] |
| GLUE (media) | acc. / Pearson | 88.64 (Bilingual) | 88.07 XLM-R reentrenado por ellos; 91.8 XLM-R publicado [Tabla 8] |
| MLQA, media de 7 idiomas (diagonal) | F1 / EM | 63.2 / 47.7 | 62.1 / 46.8 XLM-R (suyo); 63.7 / 46.3 XLM-R Base (publicado) [Tabla 10] |
| Coseno entre pares no relacionados (WMT19) | cos ↓ | ≈0.31–0.68 | ≈0.98–0.998 [Tablas 4–7] |
| Varianza en la 1.ª componente de la PCA | % ↓ | 34 % | 78 % XLM-R; 47 % RoBERTa [Fig. 1] |

**Rigor estadístico**
XNLI y GLUE se promedian sobre 5 semillas [Tablas 1, 8, 9], pero no se reporta desviación estándar ni test de significancia. MLQA no indica el número de semillas.

**¿Comparación justa?**
- El baseline principal es XLM-RoBERTa base *sin* fine-tuning en los mismos datos paralelos. No hay un baseline con MLM solo sobre los mismos pares empaquetados, ni con InfoNCE solo, así que no se puede atribuir la mejora al término JEPA.
- El XLM-R reentrenado por ellos rinde bastante por debajo del publicado (88.07 frente a 91.8 en GLUE), así que la configuración de evaluación no es óptima.

**Ablaciones**
- Monolingual frente a bilingual: bilingual es mejor en XNLI (0.744 frente a 0.732). [Tabla 9]
- Datos: Opus (10K) 0.744 > Small (2K) 0.741 > Flores (2K, mayor calidad) 0.739; las diferencias son pequeñas. [Tabla 9]
- Loss: MSE y coseno son "menos efectivos" que InfoNCE, pero no se muestran números. [§3.2]
- RankMe: el rango efectivo sube brevemente y luego se estabiliza; monolingual mantiene un RankMe mayor. [Fig. 8]

**Limitaciones que reconocen los autores**
- Sin preentrenamiento completo por limitaciones de GPU. [Apéndice A.3]
- Solo un predictor identidad y un único λ; quedan pendientes otros predictores, SIGReg y variables latentes. [Apéndice A.10]
- La loss MLM del modo bilingual es más baja porque la traducción puede contener la palabra enmascarada. [Fig. 7]

---

## 7. Conclusión y trabajo futuro (descendientes)

**Trabajo futuro propuesto**
- Probar SIGReg como loss de alineación, distintos λ, predictores parametrizados y variables latentes. [Apéndice A.10]
- Extender el framework a modelos autoregresivos, "as autoregression implicitly prevents collapse". [Apéndice A.10]
- Preentrenamiento completo con BEPA. [Apéndice A.3]

**Preguntas abiertas**
- Qué parte de la ganancia viene del término de alineación y qué parte del MLM sobre pares paralelos. No se aborda en el paper.

---

## 8. Lectura crítica

> Esta sección es la valoración del lector, no lo que dice el paper.

**¿Contribuciones claramente separadas del trabajo previo?**
Poco. El método es esencialmente un fine-tuning contrastivo (InfoNCE) sobre traducciones con un predictor identidad. Eso es muy cercano a trabajos de embeddings de frase multilingües (LaBSE, SBERT multilingüe, distilación multilingüe), que ni se citan ni se comparan. Además, el MLM sobre pares bilingües empaquetados es básicamente TLM (de XLM), y el propio paper menciona TLM como algo que RoBERTa eliminó [§2]. Llamarlo "JEPA" se apoya solo en que alinea en espacio latente: no hay predictor, ni teacher EMA, ni enmascaramiento parcial de la vista objetivo.

**Afirmaciones sin respaldo o lenguaje grandilocuente**
- "Thought space" y "the abstraction above language that humans have, our thoughts" [§1]: lenguaje especulativo sin respaldo.
- "Drastically reorganizes" [§1, §6]: se apoya en cosenos y t-SNE, sin una métrica de alineación cross-lingual estándar (p. ej. Tatoeba o BUCC retrieval).
- "Consistently outperforms" en XNLI [Tabla 1]: la mejora media es de +1.6 puntos, sin desviación estándar; en ur es +0.001.
- **Inconsistencia:** §3.2 sitúa SIGReg como trabajo futuro, pero el Apéndice A.5 dice que "InfoNCE and SigREG were the most effective".
- **Errores en la Tabla 10:** dos filas se llaman "BEPA Monolingual" (presumiblemente una es Bilingual), y zh tiene F1 40.3 frente a EM 39.6 en el baseline propio, muy inferior al XLM-R publicado (68.0), lo que sugiere un problema de evaluación en chino.
- "Strong improvement" en MLQA cross-lingual: solo se apoya en una figura (Fig. 3) y las medias de la Tabla 10 cambian ~1 punto.

**Limitaciones no mencionadas por los autores**
- No hay baseline de ablation (MLM solo en los mismos pares; InfoNCE sin MLM).
- No hay comparación con métodos de alineación multilingüe específicos.
- Coste de 3 forwards por paso (uno MLM y dos para los [CLS]), no discutido.
- Un solo modelo (base) y una sola configuración de λ.
- La reducción del coseno entre pares no relacionados es casi automática con InfoNCE; no demuestra por sí misma una mejor semántica.

**¿Es coherente la historia X → Y → Z → 1?**
Lo es a nivel narrativo ([CLS] colapsado → alinear traducciones → espacio menos colapsado y algo de mejora cross-lingual), pero el paso Z→1 no aísla la contribución JEPA de la contrastiva ni de la de datos paralelos.

**Relevancia para jepa-llm**
Aporta un caso de JEPA en encoders con vistas multilingües y un diagnóstico útil (el collapse del [CLS] y RankMe como métrica). Como evidencia de que "JEPA" ayude en encoders es débil: es más un buen punto de partida para diseñar las ablations que faltan que un resultado del que fiarse.

**TL;DR propio**
BEPA ajusta XLM-RoBERTa con MLM sobre pares de traducciones más un InfoNCE entre los [CLS] de cada frase, lo que descolapsa el espacio [CLS] y da mejoras pequeñas (+1.6 en XNLI, +0.6 en GLUE) sin varianza reportada. El método es más contrastivo que JEPA, faltan las ablations clave y hay errores en las tablas. Basta con leerlo en diagonal: sirve como idea de setup multilingüe y de diagnóstico de collapse, no como resultado sólido.
