# Notas de lectura

Visión de conjunto de los papers de `papers/`. Los resúmenes detallados están en `summaries/`.

> Los PDFs se leyeron como texto extraído; las figuras se interpretaron a partir del texto y de los ejes, no como imagen. Excepciones: I-JEPA y Balestriero & LeCun, leídos con figuras.

## Resumen por paper

| Paper | TL;DR | ¿Leer a fondo? |
|---|---|---|
| [I-JEPA](summaries/Assran%20et%20al.%20-%202023%20-%20Self-Supervised%20Learning%20from%20Images%20with%20a%20Joint-Embedding%20Predictive%20Architecture.md) | JEPA canónico en imagen: un contexto grande predice las representaciones EMA de 4 bloques objetivo grandes. Supera a MAE y data2vec en linear probing con mucho menos cómputo, pero lo decisivo es el masking multi-block (54.2 frente a 17–20 con block o random masking). Una sola semilla; sin ablation de EMA o collapse. | Sí, §3, §9 y Apéndice C (referencia de JEPA) |
| [data2vec](summaries/Baevski%20et%20al.%20-%202022%20-%20data2vec%20A%20General%20Framework%20for%20Self-supervised%20Learning%20in%20Speech,%20Vision%20and%20Language.md) | Predice la media de varias capas de un teacher EMA a partir de la entrada enmascarada. Funciona bien en voz e imagen; en texto apenas mejora a RoBERTa (82.7 frente a 82.5 en GLUE). | Sí, sobre todo §3.3 y §5.4 |
| [Balestriero & LeCun (reconstrucción)](summaries/Balestriero%20and%20LeCun%20-%202024%20-%20Learning%20by%20Reconstruction%20Produces%20Uninformative%20Features%20For%20Perception.md) | Reconstruir con MSE en píxeles centra la capacidad en el subespacio de alta varianza, el menos útil para clasificar (TinyImagenet: ~55 % con el bottom subspace frente a ~45 % con el top). Las features discriminativas se aprenden tarde y la loss de reconstrucción no las distingue. Teoría lineal limpia; la parte de masking está sobrevendida y no hay semillas. | Sí, §3 y Fig. 1 (motivación de predecir en latente) |
| [LLM-JEPA](summaries/Huang%20et%20al.%20-%202025%20-%20LLM-JEPA%20Large%20Language%20Models%20Meet%20Joint%20Embedding%20Predictive%20Architectures.md) | Alinea con coseno las representaciones de Text y Code (las dos vistas) dentro del propio LLM, junto a la loss de next-token prediction. Mejora de forma consistente, pero se benefició de un grid de hiperparámetros extra y no tiene mecanismo anti-collapse. | Obligatorio (baseline de decoders) |
| [DLLM-JEPA](summaries/Nam%20-%202026%20-%20DLLM-JEPA%20Joint%20Embedding%20Predictive%20Architectures%20for%20Masked%20Diffusion%20Language%20Models.md) | Saca las dos vistas del noise schedule de difusión, así que no necesita pares y cuesta un 33 % menos que LLM-JEPA. Con 3 semillas la mejora real es de +1 a +3 pp; el +18.7 pp del titular es el mejor caso con una sola semilla. | Sí, por su diseño de vistas y sus ablations |
| [JEPA Paradox](summaries/Dinh%20and%20Vo%20-%202026%20-%20The%20JEPA%20Paradox%20in%20Language%20The%20Geometry%20of%20Linguistic%20Alternatives.md) | Sostiene que un JEPA puro con MSE colapsa en texto porque un hueco enmascarado admite varias completaciones válidas, y el predictor aprende un centroide que no corresponde a ninguna. Su diagnóstico es reutilizable, pero no explica por qué data2vec sí funciona en texto. | Obligatorio, con espíritu crítico |
| [Sengupta (auditoría)](summaries/Sengupta%20-%202026%20-%20Representation%20Without%20Reward%20A%20JEPA%20Audit%20for%20LLM%20Fine-Tuning.md) | Prueba más de 20 objetivos auxiliares de tipo JEPA: todos cambian la geometría de los hidden states, pero ninguno mejora el exact match tras corregir por comparaciones múltiples. | Obligatorio (modelo de cómo evaluar) |
| [BERT-JEPA](summaries/Gillin%20et%20al.%20-%202026%20-%20BERT-JEPA%20Reorganizing%20CLS%20Embeddings%20for%20Language-Invariant%20Semantics.md) | En la práctica es aprendizaje contrastivo (InfoNCE) sobre el [CLS], entrenado con traducciones. Mejoras pequeñas, sin varianza, sin las ablations clave y con errores en las tablas. | En diagonal |
| [JEPA4Rec](summaries/Nguyen%20and%20Le%20-%202025%20-%20JEPA4Rec%20Learning%20Effective%20Language%20Representations%20for%20Sequential%20Recommendation%20via%20Joint%20Embedd.md) | RecFormer más una loss de predicción latente con teacher EMA, aplicado a recomendación. Le falta justo la ablation que demuestre que el término JEPA aporta algo. | En diagonal |
| [Rabtriever (Chen)](summaries/Chen%20et%20al.%20-%202026%20-%20Efficient%20Rationale-based%20Retrieval%20On-policy%20Distillation%20from%20Generative%20Rerankers%20based%20on%20JEPA.md) | Es destilación supervisada que solo se llama JEPA. Sus resultados en BEIR tienen valores anómalos. | No |

## Hallazgos que cruzan varios papers

- **El baseline de LLM-JEPA podría estar poco entrenado.** Con el mismo modelo (Llama-3.2-1B-Instruct) y el mismo dataset (NL-RX-SYNTH), Sengupta obtiene un 88.9 % de exact match sin término JEPA (LoRA r = 32). LLM-JEPA reporta un baseline del 57.3 % con fine-tuning completo y del 50.2 % con LoRA de rango 512. Hay que comprobarlo antes de reproducir sus ganancias.
- **JEPA solo, frente a JEPA como término auxiliar.** LLM-JEPA y DLLM-JEPA funcionan con un término generativo de anclaje (NTP o difusión); sin él, LLM-JEPA cae al 0 % (γ = 0). El JEPA Paradox muestra un T-JEPA puro que colapsa. La excepción a explicar es data2vec, que hace predicción latente pura en texto y funciona (con targets promediados en varias capas y normalizados). En imagen, I-JEPA también funciona sin término generativo, pero con un contexto muy informativo (85–100 % de la imagen) que reduce la multimodalidad del target, justo lo que el JEPA Paradox señala como causa del collapse.
- **Por qué predecir en latente y no reconstruir.** Balestriero & LeCun dan el argumento formal en imagen: con MSE en píxeles la reconstrucción prioriza las direcciones de alta varianza, no las discriminativas. Su análisis depende del espectro de píxeles, así que no cubre la cross-entropy sobre tokens; en texto falta demostrar que MLM/NTP sufre el mismo desajuste, que es justo la premisa de añadir un término JEPA.
- **El collapse es el riesgo central.** LLM-JEPA y BERT-JEPA no tienen EMA ni stop-gradient; DLLM-JEPA, data2vec, I-JEPA y JEPA4Rec sí usan teacher EMA, pero ninguno demuestra con una ablation que el EMA sea lo que evita el collapse. Ningún paper prueba SIGReg en serio en texto: el JEPA Paradox lo deja como predicción y Sengupta lo usa solo como auxiliar.
- **Rigor estadístico desigual.** Solo Sengupta corrige por comparaciones múltiples. LLM-JEPA y DLLM-JEPA eligen el mejor (λ, k) de un grid que el baseline no tiene. BERT-JEPA, JEPA4Rec y Rabtriever no reportan varianza.

## Ideas para el proyecto

- **Ablations obligatorias:** sin término JEPA; con vistas simétricas frente a asimétricas; con predictor frente a sin predictor; con EMA frente a sin EMA.
- **Métricas que monitorizar** (protocolo del JEPA Paradox y de Sengupta):
  - effective rank o RankMe;
  - coseno entre pares;
  - proxy de MI (InfoNCE);
  - varianza condicional del target;
  - coseno entre el gradiente del auxiliar y el de la CE.
- **Evaluación:** varias semillas, tests pareados y corrección Bonferroni o Holm, con baselines bien ajustados.
- **Experimentos candidatos:**
  - un T-JEPA con la receta de data2vec y SIGReg, para contrastar la tesis del JEPA Paradox;
  - un JEPA multi-vista completo en decoders, con target encoder (el hueco que deja abierto Sengupta);
  - vistas generadas con distintos niveles de masking en encoders, al estilo de DLLM-JEPA;
  - un análogo en texto del multi-block de I-JEPA (varios spans objetivo largos y contexto amplio), replicando sus ablations de tamaño de target, tamaño de contexto y número de targets (Tablas 8–10).
