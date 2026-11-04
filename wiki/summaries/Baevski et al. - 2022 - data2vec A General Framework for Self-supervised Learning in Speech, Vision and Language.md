# data2vec: A General Framework for Self-supervised Learning in Speech, Vision and Language

## 0. Ficha

| Campo | Valor |
|---|---|
| Autores | Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, Michael Auli (Meta AI; Q. Xu en SambaNova) |
| Venue y año | arXiv 2022 (v3, 25 oct 2022). El PDF no indica venue; se publicó en ICML 2022 (dato externo al PDF) |
| Enlace (arXiv / DOI) | https://arxiv.org/abs/2202.03555 |
| Código | https://github.com/pytorch/fairseq/tree/master/examples/data2vec |
| Fecha de lectura | 2026-11-04 |

> Nota: el PDF se leyó como texto extraído (pypdf). Las figuras 2 y 3 no se vieron como imagen; sus resultados se describen a partir del texto y de los ejes.

---

## 1. El núcleo

**X — ¿Qué intentan hacer y por qué es relevante?**
Proponer un único método de aprendizaje auto-supervisado que funcione igual para speech, visión y NLP. Hasta entonces cada modalidad tenía su propio objetivo e inductive biases (palabras en NLP, unidades discretas aprendidas en speech, visual tokens/píxeles/invarianza a augmentations en visión), y no está claro que generalicen a otras modalidades. Un algoritmo común facilitaría el aprendizaje multimodal en el futuro. [§1]

**Y — ¿Por qué es difícil?**
- Las entradas son muy distintas (waveform de 16 kHz frente a palabras frente a píxeles) y no existe un "vocabulario" natural de unidades en speech ni en visión. Por eso los métodos previos usan mecanismos específicos: cuantización o clustering de unidades de speech, tokenizadores visuales entrenados aparte, normalización de píxeles. [§1, §2]
- Si los targets se aprenden (en lugar de ser fijos), aparece el riesgo de *representation collapse*: el modelo produce la misma representación para todo y la tarea se vuelve trivial. [§3.3, §6]

**Z — ¿Cuál es su contribución?**
Un Transformer en modo student recibe una versión enmascarada de la entrada y regresa, en las posiciones enmascaradas, las representaciones que produce un teacher (EMA de los pesos del student) a partir de la entrada completa. El target es la media de las salidas normalizadas de los K bloques superiores del teacher. Así los targets son **continuos y contextualizados** (incorporan información de toda la entrada vía self-attention), no unidades locales y discretas. El objetivo es idéntico en las tres modalidades; solo cambian el feature encoder y la estrategia de masking. [§3, Fig. 1]

**1a — ¿Cómo lo verifican experimentalmente?**
- Visión: ImageNet-1K, top-1 de 84.2 (ViT-B) y 86.6 (ViT-L), el mejor entre modelos individuales. [Tabla 1]
- Speech: Librispeech/Libri-light. Con el modelo Base y 10 min de datos etiquetados, WER test-other de 12.3 frente a 15.3 de HuBERT (≈20 % relativo). [Tabla 2]
- NLP: GLUE dev, media de 82.7 frente a 82.5 de un RoBERTa reentrenado en el mismo setup; 82.9 con span masking. [Tabla 4]
- Audio: AudioSet, mAP de 34.5 frente a 32.3 de MaskSpec. [Tabla 3]
- Ablations sobre K, el contexto del teacher, el tipo de feature usada como target, la loss y el masking. [§5.4, Apéndices B–C]

**1b — ¿Hay respaldo teórico?**
No se aborda en el paper. La motivación teórica se limita a citar teorías biológicas (predictive coding, free-energy principle) [§1]; no hay teoremas ni análisis formal.

**2 — Contribuciones que reclaman los autores**
- Un framework general (data2vec) con el mismo objetivo de aprendizaje para speech, visión y NLP. [Abstract, §1]
- Targets contextualizados formados por la media de varias capas del teacher, lo que generaliza a BYOL/DINO (que solo usan la capa superior). [§1, §2]
- Nuevo state of the art en ImageNet-1K para ViT-B (modelo individual) y ViT-L, mejoras en ASR de pocos recursos y superar a RoBERTa en GLUE. [§1, §7]
- "El primer modelo de NLP preentrenado con éxito que no usa unidades discretas como target." [§5.3, §6]

---

## 2. Trabajos relacionados (hermanos)

| Trabajo | Diferencia (supuestos / método) | ¿Comparado en experimentos? | Si no, ¿por qué? |
|---|---|---|---|
| BYOL (Grill et al., 2020) | También regresa representaciones de un teacher con momentum, pero usa dos augmentations (no masking), regresa solo la capa superior y busca invarianza a transformaciones | No directamente; la ablation K=1 se presenta como "similar a BYOL" [§5.4] | Es de visión con CNNs; no se justifica explícitamente |
| DINO (Caron et al., 2021) | Self-distillation con augmentations y solo la última capa | Sí, ImageNet ViT-B: 82.8 [Tabla 1] | — |
| MoCo v3 | Contrastivo con augmentations | Sí: 83.2 / 84.1 [Tabla 1] | — |
| BEiT, PeCo | Predicen visual tokens de un tokenizador entrenado aparte (PeCo además destila MoCo v3) | Sí, separados como "multiple models" [Tabla 1] | — |
| MAE, SimMIM | Reconstruyen píxeles (targets locales) | Sí: 83.6 / 83.8 en ViT-B [Tabla 1] | — |
| MaskFeat | Predice features ingenieriles (HOG) | Sí: 84.0 / 85.7 [Tabla 1] | — |
| iBOT | Tokenizador online durante el preentrenamiento | Sí: 83.8 en ViT-B [Tabla 1] | — |
| wav2vec 2.0 | Contrastivo sobre unidades de speech cuantizadas | Sí [Tablas 2, 6] | — |
| HuBERT | Predice unidades discretas obtenidas por clustering iterativo de capas | Sí [Tablas 2, 6] | — |
| WavLM | Variante de HuBERT con más datos (MIX-94K) en Large | Sí [Tabla 2] | — |
| BERT / RoBERTa | Masked LM sobre sub-words discretos | Sí, BERT publicado y RoBERTa reentrenado [Tabla 4] | — |
| VICReg (Bardes et al., 2021) | Regularización explícita de varianza/covarianza contra el collapse | No | Prefieren normalizar los targets porque no añade hiperparámetros [§3.3] |
| Perceiver / Perceiver IO | Arquitectura general sin encoders específicos, pero supervisada | No | Lo consideran complementario [§6] |
| Multimodal pre-training (CLIP, FLAVA, VATT…) | Entrena con datos multimodales emparejados | No | data2vec no hace entrenamiento multimodal [§2] |

---

## 3. Antecedentes (ancestros)

**Conceptos previos necesarios**
- Masked prediction: BERT (tokens), wav2vec 2.0 (spans de speech), BEiT (bloques de patches). [§1, §3.2]
- Teacher con EMA / mean teacher / self-distillation: Tarvainen & Valpola, MoCo, BYOL, DINO. [§3.3]
- Transformer y ViT (patches de 16×16). [§3.1]
- Feature encoder convolucional de wav2vec 2.0 (de 16 kHz a 50 Hz). [§3.1, §4.2]
- Instance normalization y layer normalization. [§3.3]

**Definición del problema**
Dada una entrada $x$ embebida como secuencia de time-steps, el student recibe $x$ con parte de las posiciones sustituidas por un embedding MASK aprendido, y predice $f_t(x)$ para cada time-step enmascarado $t$. El target es
$$y_t = \frac{1}{K}\sum_{l=L-K+1}^{L} \hat a_t^l,$$
donde $\hat a_t^l$ es la salida normalizada del bloque $l$ del teacher (red de $L$ bloques) sobre la entrada sin enmascarar. Los parámetros del teacher se actualizan como $\Delta \leftarrow \tau\Delta + (1-\tau)\theta$. [§3.3]

**Supuestos inusuales o restrictivos**
- Se asume un encoder bidireccional que ve todo el contexto (no autoregresivo).
- Se aprende una representación por modalidad: no es un único modelo multimodal. [§1]
- El feature encoder, el masking y varios hiperparámetros (K, β, τ, tipo de normalización, incluso la loss) siguen siendo específicos de cada modalidad. [§4, §6]

**¿El escenario del problema es una contribución nueva?**
Parcialmente: el escenario de "un mismo objetivo self-supervised para varias modalidades" se presenta como motivación propia, pero cada tarea (preentrenar y luego hacer fine-tuning) es estándar. [§1]

---

## 4. Método

**Qué hacen**
1. Embeben la entrada con un encoder específico de la modalidad: patches lineales (visión), CNN 1-D (speech), embeddings BPE (texto). [§3.1]
2. **Modo teacher:** pasan la entrada completa por el Transformer con pesos EMA $\Delta$ y toman la salida de la FFN (antes de la última conexión residual) de los K bloques superiores. [§3.3, nota 5]
3. Normalizan cada bloque (instance norm sin parámetros en speech; layer norm sin parámetros en NLP y visión) y promedian para obtener $y_t$. [§3.3]
4. **Modo student:** enmascaran la entrada (bloques de patches en visión, spans en speech, tokens al estilo BERT en texto) y la pasan por el Transformer con pesos $\theta$. [§3.2]
5. Regresan $y_t$ solo en los time-steps enmascarados con Smooth L1 (parámetro β); en speech usan L2. [§3.4, §4.2]
6. Actualizan el teacher con EMA, con τ creciendo linealmente de $\tau_0$ a $\tau_e$ durante $\tau_n$ pasos. El feature encoder y el positional encoder se comparten entre teacher y student. [§3.3]

**Por qué lo hacen así**
- **Targets contextualizados:** un experimento preliminar mostró que enmascarar en el teacher empeoraba los resultados; más contexto da targets más ricos. [§3.3 nota 4, Fig. 3]
- **Media de K capas:** las capas superiores de wav2vec 2.0 rinden peor que las intermedias, y promediar rinde igual que predecir cada capa con su propia proyección, pero es más eficiente. [§3.3, §5.4]
- **Normalización de targets:** evita el collapse y que dominen las capas con norma alta. Instance norm en speech porque las representaciones vecinas están muy correladas. [§3.3]
- **Teacher EMA con τ creciente:** el teacher se actualiza más al principio, cuando el modelo es aleatorio, y menos después. [§3.3]
- **Smooth L1:** es menos sensible a outliers (aunque obliga a ajustar β). [§3.4]
- **Feature encoders y masking específicos:** la resolución y la correlación de las entradas son muy distintas; el texto es muy semántico y basta enmascarar el 15 %, mientras que en speech hay que enmascarar spans. [§6, Apéndice C]
- **Salida de la FFN como target:** la salida de self-attention da un modelo inservible porque está sesgada hacia otros time-steps. [§5.4, Tabla 5]

---

## 5. Configuración experimental

**Instancia del problema**
- Visión: preentrenamiento en ImageNet-1K (train) y fine-tuning de clasificación en ImageNet-1K; métrica top-1 en val. [§5.1]
- Speech: preentrenamiento en LS-960 (Base) o LL-60K (Large); fine-tuning de ASR con 10 min, 1 h, 10 h, 100 h y 960 h etiquetadas; WER en dev/test clean/other con LM 4-gram. [§5.2, Tablas 2, 6]
- Audio: AudioSet, preentrenamiento en AS y fine-tuning en el subset balanceado de 20K; métrica mAP. [§5.2, Tabla 3]
- NLP: preentrenamiento en Books Corpus + English Wikipedia (setup BERT); GLUE dev (MNLI, QNLI, RTE, MRPC, QQP, STS-B, CoLA, SST-2, sin WNLI). [§5.3]

**Detalles de implementación**
- Base: L=12, H=768; Large: L=24, H=1024; FFN de 4H. EMA en fp32. [§4]
- Visión: masking del 60 % por bloques (BEiT usa 40 %), β=2, K=6, τ=0.9998 constante, stochastic depth 0.2, batch 2048 (B) / 8192 (L), 800 épocas (B). ViT-L: 1600 épocas, reiniciando el LR y el teacher a la época 800 con τ=0.9999. Basado en el código de BEiT. [§4.1]
- Speech: masking p=0.065 × 10 pasos (≈49 % enmascarado), τ de 0.999 a 0.9999 en 30K pasos, K=8, loss L2, LR pico 5e-4, scheduler tri-stage, 400K updates, batch de 63 min de audio; fine-tuning como wav2vec 2.0. fairseq. [§4.2]
- AudioSet: K=12, 200K updates, DeepNorm y layer norm en los targets, mixup y masking durante el fine-tuning. [§5.2]
- NLP: RoBERTa en fairseq, BPE de 50K, masking BERT del 15 % (80/10/10) sin NSP, τ de 0.999 a 0.9999 en 100K pasos, K=10, β=4, 1M updates, LR 2e-4, 16 GPUs, batch de 256 secuencias × 512 tokens. Fine-tuning con 4 LRs, eligiendo el mejor en conjunto para todas las tareas. [§4.3]
- Cómputo total (horas de GPU) en visión y speech: no se aborda en el paper.

---

## 6. Resultados y discusión

**Resultados principales frente a baselines**

| Tarea / dataset | Métrica | Método | Mejor baseline |
|---|---|---|---|
| ImageNet-1K, ViT-B | top-1 ↑ | 84.2 | 84.0 MaskFeat (modelo individual); 84.5 PeCo (múltiples modelos) [Tabla 1] |
| ImageNet-1K, ViT-L | top-1 ↑ | 86.6 | 86.5 PeCo; 85.9 MAE [Tabla 1] |
| LS test-other, Base, 10 min | WER ↓ | 12.3 | 15.3 HuBERT [Tabla 2] |
| LS test-other, Base, 1 h / 10 h / 100 h | WER ↓ | 9.1 / 8.1 / 6.8 | 10.8 / 9.2 / 7.7 WavLM [Tabla 2] |
| LS test-other, Base, 960 h | WER ↓ | 5.5 | 6.1 wav2vec 2.0 [Tabla 2] |
| LS test-other, Large, 10 min | WER ↓ | 8.4 | 10.1 HuBERT [Tabla 2] |
| LS test-other, Large, 100 h / 960 h | WER ↓ | 4.6 / 3.7 | 4.5 HuBERT / 3.6 wav2vec 2.0 (data2vec no gana) [Tabla 2] |
| AudioSet | mAP ↑ | 34.5 | 32.3 MaskSpec [Tabla 3] |
| GLUE dev (media) | score ↑ | 82.7 (82.9 con span masking) | 82.5 RoBERTa reentrenado [Tabla 4] |

**Rigor estadístico**
GLUE es la media de 5 ejecuciones de fine-tuning [§5.3]. No hay desviación estándar ni intervalos en ninguna tabla. El preentrenamiento usa una sola semilla (no se indica lo contrario) y no se dan semillas para visión ni speech.

**¿Comparación justa?**
- NLP: el RoBERTa reentrenado en el mismo setup es un baseline *like-for-like* [§5.3]. El LR de fine-tuning se elige sobre el propio dev de GLUE con el que se reporta [§4.3].
- Visión y speech: los números de los baselines se toman de sus papers, con distinto número de épocas y cómputo. ViT-L usa 1600 épocas con un reinicio especial [§4.1]. WavLM Large usa más datos (MIX-94K) [Tabla 2].
- No se compara el coste de entrenamiento (el teacher añade un forward extra por paso).

**Ablaciones**
- **K capas:** promediar varias capas mejora sobre K=1 (estilo BYOL) en las tres modalidades; el efecto es muy marcado en speech y NLP y leve en visión. Usar todas las capas queda cerca del K óptimo. [§5.4, Fig. 2]
- **Contexto del teacher:** restringir la atención del teacher a una fracción de la entrada empeora el rendimiento de forma monótona; el mejor resultado se obtiene con el 100 % de contexto. Solo se mide en speech y visión. [Fig. 3]
- **Tipo de feature:** self-attention da WER 100 (no funciona); FFN 13.1; FFN + residual 14.8; final del bloque 14.5 (setup reducido de speech). [Tabla 5]
- **Loss:** L2, L1 y Smooth L1 quedan entre 16.8 y 17.3 de WER; el efecto es pequeño. [Apéndice B, Tabla 7]
- **Masking en speech:** la probabilidad de masking apenas importa (17.1–17.3); la longitud del span sí (10 → 17.1; 5 → 22.1; 15 → 18.9). [Apéndice C, Tabla 8]
- **NLP:** span masking de 4 tokens con p=0.35 mejora a 82.9 (con un schedule coseno distinto). [Tabla 4, nota 7]

**Limitaciones que reconocen los autores**
- Siguen usando feature encoders y estrategias de masking específicos de cada modalidad. [§6, §7]
- Aprenden representaciones por separado para cada modalidad; no hay entrenamiento multimodal. [§1, §2]
- Smooth L1 obliga a ajustar β. [§3.4]
- El collapse aparece con LR alto o warmup corto, con τ bajo, y en modalidades con targets adyacentes muy correlados (speech); se mitiga ajustando hiperparámetros y normalizando. [§6]
- En speech con muchos datos etiquetados (100 h / 960 h) el rendimiento es comparable, no superior. [§5.2]

---

## 7. Conclusión y trabajo futuro (descendientes)

**Trabajo futuro propuesto**
- Aprendizaje cross-modal con un método único, p. ej. audio-visual speech recognition o cross-modal retrieval. [§7]
- Unificar también los encoders de entrada y las estrategias de masking. [§7]

**Preguntas abiertas**
- Si el método funciona sin encoders específicos de modalidad (lo relacionan con Perceiver). [§6]
- Cómo evitar el collapse de forma robusta sin tanto ajuste de hiperparámetros: el paper lo resuelve de forma empírica. [§6]

---

## 8. Lectura crítica

> Esta sección es la valoración del lector, no lo que dice el paper.

**¿Contribuciones claramente separadas del trabajo previo?**
Sí, en general. El paper deja claro que combina piezas existentes: masked prediction (BERT/wav2vec 2.0/BEiT), el teacher EMA (BYOL/DINO/mean teacher), y encoders y masking tomados de la literatura [§1, §3]. La aportación concreta (targets como media normalizada de K capas, contextualizados, con un objetivo común) está bien aislada y respaldada por ablations (Fig. 2, Fig. 3, Tabla 5).

**Afirmaciones sin respaldo o lenguaje grandilocuente**
- "Same learning method for either speech, NLP or computer vision" se matiza poco: K, β, τ, el tipo de normalización, la loss (L2 en speech frente a Smooth L1), DeepNorm en AudioSet y el ratio de masking cambian por modalidad [§4]. Es el mismo *tipo* de objetivo, no la misma receta.
- La motivación biológica (Friston) [§1] no se conecta con ningún experimento.
- "First successful pre-trained NLP model which does not use discrete units as the training target" [§5.3] es una afirmación fuerte, apoyada solo en un "to our knowledge", y el resultado en NLP es marginal.
- "Outperforming a like for like RoBERTa baseline" [§1]: la media es 82.7 frente a 82.5 sin desviación estándar, y data2vec pierde en MNLI (83.2 frente a 84.1), RTE (67.0 frente a 69.3), STS-B y SST-2. La mejora en la media se debe sobre todo a CoLA (62.2 frente a 56.8). Con 5 ejecuciones sin varianza, no es una mejora demostrada.
- "New state of the art for ViT-L": 86.6 frente a 86.5 de PeCo, también sin varianza.

**Limitaciones no mencionadas por los autores**
- **Coste:** no se reporta el cómputo; el teacher exige un forward completo extra por paso, y ViT-L usa 1600 épocas con un reinicio ad hoc del teacher [§4.1].
- **Selección de hiperparámetros:** en GLUE el LR se elige sobre el mismo dev en el que se reporta [§4.3].
- **Ablations reducidas:** la ablation de contexto del teacher no incluye NLP [Fig. 3], y la de tipo de feature solo cubre speech [Tabla 5].
- **Solo encoders bidireccionales:** no se explora el caso generativo/autoregresivo, ni si las representaciones sirven para generación.
- **NLP pequeño:** solo hay modelo Base en NLP, y solo se evalúa GLUE (sin SQuAD ni tareas de generación).

**¿Es coherente la historia X → Y → Z → 1?**
Sí. X (unificar el self-supervised learning) → Y (cada modalidad necesita targets ad hoc y existe el riesgo de collapse) → Z (regresar representaciones latentes contextualizadas de un teacher EMA) → 1 (resultados en tres modalidades y ablations que aíslan los componentes clave). El punto más débil es que la evidencia en NLP es la más floja de las tres, y es la que más nos interesa.

**Relevancia para jepa-llm**
data2vec es, en la práctica, un precursor directo de JEPA en texto: predice en espacio latente, con un teacher EMA y masking. Aunque el paper nunca usa el término "JEPA", sirve de referencia para lo que funciona en encoders BERT-style:
- targets de varias capas normalizadas;
- salida de la FFN como target;
- span masking en texto.

También avisa de que, en texto, la ganancia sobre MLM es pequeña. Los decoders quedan fuera de su alcance.

**TL;DR propio**
data2vec entrena un Transformer a predecir, desde una entrada enmascarada, la media normalizada de las K capas superiores de un teacher EMA que ve la entrada completa, con el mismo objetivo en speech, visión y texto. Los resultados son claramente buenos en speech con pocos recursos y en visión, pero en NLP la mejora sobre RoBERTa es marginal y sin rigor estadístico. Merece la pena leerlo a fondo para el proyecto, sobre todo §3.3 y §5.4, porque es la base práctica sobre la que se apoyan los JEPA para lenguaje y sus ablations dicen qué elegir como target.
