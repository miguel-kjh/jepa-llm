# Efficient Rationale-based Retrieval: On-policy Distillation from Generative Rerankers based on JEPA

## 0. Ficha

| Campo | Valor |
|---|---|
| Autores | Teng Chen, Sheng Xu, Feixiang Guo, Xiaoyu Wang, Qingqing Gu, Hongyan Li, Luo Ji (Geely AI Lab) |
| Venue y año | ICMR '26 (International Conference on Multimedia Retrieval), Ámsterdam, 2026 |
| Enlace (arXiv / DOI) | https://doi.org/10.1145/3805622.3810780 |
| Código | No publicado (el PDF no da URL) |
| Fecha de lectura | 2026-11-04 |

> Nota: el PDF se leyó como texto extraído (pypdf); las Figs. 4–6 salieron parcialmente ilegibles en el texto.

---

## 1. El núcleo

**X — ¿Qué intentan hacer y por qué es relevante?**
Construir un retriever para *rationale-based retrieval*: recuperación en la que la relación entre query y documento depende de razonamiento o lógica implícita, no de similitud semántica (niveles L3–L4 de Zhao et al.). Debe tener la precisión de un reranker LLM (cross-encoder) con la velocidad de un bi-encoder. [§1]

**Y — ¿Por qué es difícil?**
- Los bi-encoders codifican query y documento por separado y no capturan su interacción cruzada; fallan en este tipo de retrieval. [§1, §3.3]
- Los rerankers LLM sí la capturan, pero cuestan $n\,\mathcal{O}((L_D+L_Q)^2)$ online por query, lo que es inviable a gran escala. [§3.4]

**Z — ¿Cuál es su contribución?**
Rabtriever, un proceso en tres pasos:
1. **Teacher:** un reranker generativo (LaHoRe) que pone el documento *antes* que la query y responde <T>/<F>. Gracias a la atención causal, la parte del documento se puede precalcular. [§4.1, Eq. 8]
2. **Student:** el mismo LLM congelado, que codifica la query sola. Un predictor ligero (MLP de 2 capas multiplicado elemento a elemento por el último embedding del documento) proyecta el embedding de la query hacia el embedding "condicionado al documento" del teacher. [§4.2, Eq. 12]
3. **Entrenamiento del predictor:** MSE entre embeddings + $w \cdot \mathrm{KL}_{\text{rev}}$ entre los logits del student y del teacher. Los autores lo enmarcan como JEPA: $x = e_Q$, $y = e^{ce}_{QD}$, y $e_D$ como variable latente $z$. [§4.2, Eq. 11]

**1a — ¿Cómo lo verifican experimentalmente?**
- Retrieval de rationale: ESConv y PsyQA (estrategias de apoyo emocional) y SayCan (selección de skills robóticos). En SayCan, R@1 de 67.3 frente a 44.2 de OneGen (mejor retriever) y 71.2 del teacher. [Tabla 2]
- Latencia: 95 ms frente a 980 ms del teacher. [Tabla 2]
- Generalización: MRR@10 de 49.30 en MS MARCO dev [Fig. 6]; nDCG@10 medio de 58.2 en BEIR frente a 59.2 de GritLM y 65.8 del teacher. [Tabla 4]

**1b — ¿Hay respaldo teórico?**
Solo un análisis de complejidad: online pasa de $n\,\mathcal{O}((L_D+L_Q)^2)$ a $\mathcal{O}(L_Q^2)+\mathcal{O}(L_Q)+n\,\mathcal{O}(1)$. [§4.3, Eq. 13] No hay garantías sobre la calidad de la destilación.

**2 — Contribuciones que reclaman los autores**
- Un retriever LLM eficiente con codificación independiente que rinde como un reranker en rationale-based retrieval. [§1]
- Un paradigma de destilación desde un reranker generativo usando JEPA para reconstruir el embedding de la query condicionado al documento. [§1]
- Evaluación en escenarios variados (conversación empática, robótica, MS MARCO, BEIR). [§1]

---

## 2. Trabajos relacionados (hermanos)

| Trabajo | Diferencia (supuestos / método) | ¿Comparado en experimentos? | Si no, ¿por qué? |
|---|---|---|---|
| RepLLaMA, E5(-Mistral), TART-Contriever | Bi-encoders densos con InfoNCE, orientados a hechos | Sí [Tablas 2, 4] | — |
| InstructOR, Promptriever, LLaMA2Vec | Retrievers instruccionales o con prompt "NEXT:" | Sí [Tablas 2, 4] | — |
| GritLM, OneGen | Retrievers híbridos representación + generación | Sí [Tablas 2, 4] | — |
| RankLLaMA | Reranker cross-encoder con proyección lineal | Sí [Tablas 2, 4] | — |
| LaHoRe (Ji et al., 2025; mismos autores) | Reranker generativo con documento antes de la query; es el teacher | Sí [Tablas 2, 4] | — |
| RankGPT (Sun et al., 2023) | Paradigma de *relevance generation* adoptado | No | Se usa como diseño, no como baseline |
| MiniLLM (Gu et al., 2024) | Destilación con reverse KL | No | Inspira la loss; no se compara |
| Otros métodos de destilación cross→bi-encoder (p. ej. ColBERT, late interaction) | — | No | No se discuten |

---

## 3. Antecedentes (ancestros)

**Conceptos previos necesarios**
- JEPA como modelo basado en energía: $\min F(P_z(x), y)$. [§3.1, Eq. 1]
- Bi-encoder frente a cross-encoder y su complejidad. [§3.3–3.4]
- Knowledge distillation basada en features y en logits; forward frente a reverse KL. [§1, Fig. 1]
- On-policy distillation (blog de Thinking Machines). [§1, nota 1]

**Definición del problema**
Dada una query $Q$ y un pool $\{D_i\}_{i=1}^n$, calcular scores $s_i(Q, D_i)$ y devolver el top-*k*. El student calcula
$$e^{\text{student}}_Q = \text{MLP}_\theta(e_Q) \odot e_D[-1], \qquad \hat s = \frac{\log P(\langle T\rangle)}{\log P(\langle T\rangle) + \log P(\langle F\rangle)}$$
con la cabeza del LM aplicada sobre $e_D \oplus e^{\text{student}}_Q$. [Eqs. 9, 10, 12]

**Supuestos inusuales o restrictivos**
- Los documentos son cortos ($L_D \le 32$ tokens) [§5.1]; los pools de rationale son minúsculos (8 y 7 estrategias en ESConv y PsyQA).
- El backbone del student está congelado; solo se entrena el MLP. [§4.2]
- Depende de tener ya un reranker generativo entrenado como teacher.

**¿El escenario del problema es una contribución nueva?**
No: rationale-based retrieval y el teacher LaHoRe vienen de trabajo previo de los mismos autores [8]. Lo nuevo es la destilación a bi-encoder.

---

## 4. Método

**Qué hacen**
1. Hacen SFT del teacher (Qwen2-7B-Instruct) con el template "Document / Query / ¿se puede responder? <T>/<F>", con cross-entropy sobre la respuesta y todos los pares negativos posibles. [§4.1]
2. Inicializan el student desde el teacher y lo congelan; insertan el predictor $P$ entre las capas y la LM head. [§4.2]
3. El student codifica $D$ (offline) y $Q$ (online) por separado; el predictor combina $\text{MLP}(e_Q)$ con $e_D[-1]$. [Eq. 12]
4. Loss $F_{\text{MSE}}(e^{\text{student}}_Q, e^{\text{teacher}}_Q) + 0.25 \cdot \mathrm{KL}(P_{\text{student}} \| P_{\text{teacher}})$, con rollouts del student ("on-policy"). [§4.2]
5. En inferencia el score sale de los log-probs de <T>/<F>, con vLLM y prefix cache. [§5.1]

**Por qué lo hacen así**
- **Documento antes de la query:** con atención causal, $\text{LLM}(D, Q)[:L_D] = \text{LLM}(D)$, así que no hace falta destilar el encoder del documento. [Eq. 8]
- **MSE y no contrastiva:** "as suggested by JEPA", por convergencia y estabilidad. [§4.2]
- **Reverse KL:** el teacher es bimodal (<T>/<F>); el reverse KL busca la moda en vez de promediar. [Fig. 1, §4.2]
- **Multiplicación elemento a elemento:** es la única vía por la que el documento condiciona a la query; sin ella el método falla. [Tabla 3]

---

## 5. Configuración experimental

**Instancia del problema**
- ESConv (1.2K conversaciones; 8 estrategias), PsyQA (4,012 pares QA; 7 estrategias), SayCan (≈276K trayectorias; 551 skills). [§5.2]
- MS MARCO passage (≈500K de train; MRR@10 en dev) y BEIR (13 tareas, zero-shot, nDCG@10). [§5.3]
- Métricas: R@k, p-MRR@10, nDCG@10 y tiempo de respuesta online (RT) por par query-documento. [§5.1]

**Detalles de implementación**
- Qwen2-7B-Instruct, LlamaFactory, 8× A100, menos de 10 h, LR 1e-6, batch 16, $w = 0.25$, $L_D$ máx. 32 con left-padding, ventana de 4096. El número de épocas no está ("the epoch number of epochs" queda incompleto en el texto). [§5.1]
- Escalabilidad con backbones de 0.5B y 1.5B. [Fig. 5]

---

## 6. Resultados y discusión

**Resultados principales frente a baselines**

| Tarea / dataset | Métrica | Método | Mejor baseline |
|---|---|---|---|
| ESConv | R@1 / MRR | 32.8 / 54.7 | OneGen 29.0 / LlaMA2Vec 45.5 (retrievers); teacher 34.4 / 54.8 [Tabla 2] |
| PsyQA | R@1 / MRR | 32.8 / 55.1 | LlaMA2Vec 34.3 / 57.5 (supera a Rabtriever); teacher 35.6 / 58.2 [Tabla 2] |
| SayCan | R@1 / MRR | 67.3 / 73.7 | OneGen 44.2 / 56.8; teacher 71.2 / 79.8 [Tabla 2] |
| Latencia | RT (ms) ↓ | 95 | LlaMA2Vec 93; teacher 980 [Tabla 2] |
| MS MARCO dev | MRR@10 | 49.30 | teacher 52.4 [§5.3, Fig. 6] |
| BEIR (media de 13) | nDCG@10 | 58.2 | GritLM 59.2 (retriever); teacher 65.8 [Tabla 4] |

**Rigor estadístico**
No se aborda en el paper: no hay semillas, desviación estándar ni tests.

**¿Comparación justa?**
- Los baselines tienen tamaños distintos (110M–7B); la mayoría son de 7B. [Tabla 2]
- No se indica si los retrievers baseline se ajustaron en los datasets de rationale ni con qué protocolo; parece que solo Rabtriever y su teacher se entrenaron en ellos.
- La latencia solo se compara con LlaMA2Vec y con el teacher.

**Ablaciones**
- Sin la multiplicación: SayCan R@1 9.6, PsyQA R@1 2.0. [Tabla 3]
- Sin MSE: PsyQA R@1 0.0, SayCan 23.1. [Tabla 3]
- Sin reverse KL: PsyQA R@1 24.1, SayCan 21.2. Con forward KL en su lugar: 28.4 y 46.8. [Tabla 3]
- La latencia crece de forma lineal con $L_D$ en el student y cuadrática en el teacher. [Fig. 4]
- Backbones de 0.5B y 1.5B rinden "parecido" a LlaMA2Vec-7B en SayCan. [Fig. 5]

**Limitaciones que reconocen los autores**
- Online sigue siendo más lento que un bi-encoder clásico (tiene términos $\mathcal{O}(L_Q^2)$ y $\mathcal{O}(L_Q)$). [§4.3]
- Por lo demás, no se aborda en el paper.

---

## 7. Conclusión y trabajo futuro (descendientes)

**Trabajo futuro propuesto**
No se aborda en el paper.

**Preguntas abiertas**
- Si es posible bajar a backbones más pequeños a cambio de precisión. [§5.2, Fig. 5]

---

## 8. Lectura crítica

> Esta sección es la valoración del lector, no lo que dice el paper.

**¿Contribuciones claramente separadas del trabajo previo?**
Regular. El teacher, la formulación del problema y los datasets vienen del trabajo previo de los mismos autores (LaHoRe). Lo nuevo es un predictor MLP ⊙ $e_D$ entrenado con MSE + reverse KL, es decir, destilación de features y logits. Llamarlo "JEPA" es un reencuadre terminológico: no es auto-supervisado, no hay dos vistas de la misma entrada ni mecanismo anti-collapse, y la "variable latente" es el embedding del documento.

**Afirmaciones sin respaldo o lenguaje grandilocuente**
- "Comparable performance to the reranker" en BEIR [Abstract]: 58.2 frente a 65.8 no es comparable, y además queda por debajo de GritLM (59.2).
- "State-of-the-art performances on both MS MARCO and BEIR" [§6]: en BEIR no es el mejor retriever.
- **Números de BEIR anómalos:** el teacher obtiene nDCG@10 de 91.4 en FiQA y 92.0 en NQ, muy por encima de cualquier valor publicado habitual. Además, teacher y student coinciden exactamente en Arguana (36.8), NFCorpus (40.6) y SciFact (78.9). Esto sugiere un posible problema de evaluación o de transcripción.
- **Contradicción en la complejidad:** "linear dependency on the document length". La Eq. 13 indica que el coste online no depende de $L_D$, y la frase de la Fig. 4 dice "linear dependency of teacher and quadratic dependency of teacher" (errata).
- "Orders-of-magnitude reduction in inference latency" [§1]: es ≈10× (980 frente a 95 ms), un solo orden de magnitud.

**Limitaciones no mencionadas por los autores**
- **Documentos muy cortos:** con $L_D \le 32$ y pools de 7–8 documentos, el problema de eficiencia que motiva el paper apenas existe en los datasets de rationale, y en MS MARCO/BEIR truncar a 32 tokens es una restricción fuerte no discutida.
- **Interacción query-documento mínima:** se reduce a un producto elemento a elemento con el último embedding del documento. Es dudoso que esto "reconstruya" la comprensión cruzada en general; los resultados pueden deberse a que el pool es pequeño y fijo.
- **Sin rigor estadístico ni detalles de entrenamiento:** no hay varianza ni semillas, y el número de épocas falta en el texto.
- **Valores extremos en las ablations:** R@1 = 0.0 sugiere fallos de entrenamiento más que el efecto del componente.

**¿Es coherente la historia X → Y → Z → 1?**
X → Y → Z se sigue bien (rerankers caros → destilar a bi-encoder). En el paso a 1 hay inconsistencias numéricas (BEIR) y la ventaja de eficiencia se demuestra en un régimen (documentos de 32 tokens) donde importa poco.

**Relevancia para jepa-llm**
Baja. Es un ejemplo de uso laxo de la etiqueta "JEPA" (destilación con MSE entre embeddings). La única idea aprovechable es poner el documento antes de la query para reutilizar la caché causal, y el uso de reverse KL al destilar distribuciones bimodales.

**TL;DR propio**
Rabtriever destila un reranker LLM generativo en un retriever bi-encoder entrenando un MLP que, multiplicado por el embedding del documento, imita el embedding de la query condicionado al documento (MSE + reverse KL). Lo llaman JEPA, pero es destilación supervisada. Funciona bien en tareas de retrieval de "rationale" con pools diminutos, pero los resultados en BEIR presentan anomalías y no hay rigor estadístico. No merece la pena leerlo a fondo para el proyecto.
