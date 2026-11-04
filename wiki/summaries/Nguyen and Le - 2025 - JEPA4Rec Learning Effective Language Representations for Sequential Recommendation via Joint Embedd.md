# JEPA4Rec: Learning Effective Language Representations for Sequential Recommendation via Joint Embedding Predictive Architecture

## 0. Ficha

| Campo | Valor |
|---|---|
| Autores | Minh-Anh Nguyen, Dung D. Le (VinUniversity, Vietnam) |
| Venue y año | COLM 2025 |
| Enlace (arXiv / DOI) | https://arxiv.org/abs/2504.10512 |
| Código | No publicado (el PDF no da URL) |
| Fecha de lectura | 2026-11-04 |

> Nota: el PDF se leyó como texto extraído (pypdf); las Figs. 3–6 y 9 (curvas) no se vieron como imagen.

---

## 1. El núcleo

**X — ¿Qué intentan hacer y por qué es relevante?**
Recomendación secuencial (predecir el siguiente ítem a partir del historial) representando los ítems como texto, para transferir conocimiento entre dominios y plataformas sin IDs compartidos y con poco preentrenamiento. Los métodos basados en IDs no transfieren y sufren con el cold-start. [§1]

**Y — ¿Por qué es difícil?**
- Los PLMs generales no están alineados con las descripciones de ítems y dan embeddings de frase poco útiles para modelar preferencias. [§1]
- Hacen falta representaciones flexibles más allá de concatenar atributos, captar preferencias de "sentido común" (marca, categoría) y entrenar de forma eficiente con datos escasos. [§1]
- Los métodos text-based previos (UniSRec, VQ-Rec, RecFormer) separan la comprensión textual del modelado secuencial y dependen de objetivos contrastivos y de grandes cantidades de datos de preentrenamiento. [§3.5]

**Z — ¿Cuál es su contribución?**
Añadir a RecFormer, un Longformer bidireccional sobre ítems-como-texto, un objetivo de tipo JEPA:
- **Context encoder:** recibe el historial con el 15 % de los tokens enmascarados y el siguiente ítem con el 50 % enmascarado.
- **Predictor MLP:** a partir de $h_{CLS}$ (más el embedding de posición del ítem, o el embedding del siguiente ítem enmascarado), predice la representación completa de un ítem del historial y del siguiente ítem.
- **Target:** la representación que da un target encoder EMA sobre el ítem sin enmascarar.
- **Loss de mapping L2,** que se combina con MLM y con una loss contrastiva secuencia-ítem. [§3.3]

**1a — ¿Cómo lo verifican experimentalmente?**
- Preentrenamiento en 3 categorías de Amazon (≈35 % de los datos de RecFormer) y fine-tuning en 5 categorías de Amazon y en Online Retail (cross-platform). [§4.1]
- Mejora media sobre RecFormer: +6.22 % en Recall@10 y +10.06 % en NDCG@10. [Tabla 2, §4.2]
- Zero-shot, bajos recursos, robustez a metadatos incompletos y ablations. [§4.3–4.4, Apéndices E–G]

**1b — ¿Hay respaldo teórico?**
No se aborda en el paper.

**2 — Contribuciones que reclaman los autores**
- Un encoder bidireccional con embeddings modificados para frases de ítems (token, posición de token, tipo de token e posición de ítem), usado como context y target encoder, y una estrategia de masking a distintas tasas. [§1]
- Entrenamiento en dos fases (preentrenamiento y fine-tuning) con objetivos self-supervised. [§1]
- Supera al estado del arte en todos los datasets usando una fracción de los datos de preentrenamiento. [§1]

---

## 2. Trabajos relacionados (hermanos)

| Trabajo | Diferencia (supuestos / método) | ¿Comparado en experimentos? | Si no, ¿por qué? |
|---|---|---|---|
| RecFormer (Li et al., 2023a) | Mismo backbone (Longformer sobre ítems-como-texto) con MLM + contrastiva; JEPA4Rec es "building upon" RecFormer | Sí; es el baseline principal [Tabla 2, Figs. 3–4, Tablas 4–5] | — |
| UniSRec, VQ-Rec | Features de un PLM congelado + adaptador MoE o códigos discretos | Sí [Tabla 2] | — |
| SASRec, BERT4Rec | Solo IDs | Sí [Tabla 2] | — |
| S3-Rec, LlamaRec | IDs + texto; LlamaRec rankea con un LLM | Sí [Tabla 2] | — |
| P5 | Recomendación como procesamiento de lenguaje | No | No se justifica |
| I-JEPA, S-JEPA | JEPA en imagen y esqueletos; inspiración | No | Otra modalidad |

---

## 3. Antecedentes (ancestros)

**Conceptos previos necesarios**
- JEPA: context encoder, target encoder EMA y predictor, con asimetría contra el collapse (SimSiam). [§1, Apéndice A]
- MLM al estilo BERT (80/10/10). [§3.3.1]
- Longformer (atención local más [CLS] global). [§3.2]
- Loss contrastiva con in-batch negatives (sequence-item). [§3.3.2]

**Definición del problema**
Cada ítem es una frase $S_i = \{a_{i1}, v_{i1}, \dots\}$ (atributo-valor). Dado el historial $\{S_1..S_n\}$, se predice $S_{n+1}$. Con $X = \{[CLS], S_n, \dots, S_1\}$ y $h_{CLS} = \text{Encoder}(X)$:
$$\hat h_n = \text{FFN}_1(h_{CLS} \oplus D_n), \quad \hat h_{n+1} = \text{FFN}_2(h_{CLS} \oplus h_{n+1,M})$$
$$\mathcal{L}_{\text{map}} = \|h_n - \hat h_n\|^2 + \|h_{n+1} - \hat h_{n+1}\|^2$$
$$\mathcal{L}_{PT} = \mathcal{L}_{S\text{-}I} + \lambda_1 \mathcal{L}_{MLM} + \lambda_2 \mathcal{L}_{\text{map}}$$
[§3.3.2]

**Supuestos inusuales o restrictivos**
- Los ítems deben tener metadatos textuales (título, marca, categoría o descripción). [§4.1]
- Sobre el historial se muestrea *un* ítem enmascarado por secuencia como target. [§3.3.1]

**¿El escenario del problema es una contribución nueva?**
No; el escenario (recomendación secuencial text-only con transferencia entre dominios) viene de UniSRec y RecFormer.

---

## 4. Método

**Qué hacen**
1. Aplanan los metadatos de cada ítem en una frase atributo-valor e invierten el orden del historial (el más reciente primero). [§3.1, §3.2]
2. Suman 4 embeddings (token, posición de token, tipo de token e posición de ítem) y los codifican con Longformer. [§3.2]
3. Enmascaran el 15 % de los tokens del historial y el 50 % del siguiente ítem (BERT 80/10/10). [§3.3.1]
4. El target encoder (EMA del context encoder) codifica, sin enmascarar, un ítem del historial y el siguiente ítem. [§3.3.1]
5. Los predictores MLP reconstruyen ambos embeddings; se aplica la loss L2 junto con MLM y la contrastiva S-I. [§3.3.2]
6. En el fine-tuning solo se usa el context encoder, con una loss contrastiva *full softmax* sobre todos los ítems; la matriz de ítems se actualiza cada época. [§3.4]

**Por qué lo hacen así**
- **Masking del 50 % en el siguiente ítem:** obliga a inferirlo a partir del historial y de información parcial; la ablation muestra el óptimo en torno al 50 %. [§3.3.1, Fig. 6]
- **15 % en el historial:** más masking degradaría $h_{CLS}$. [§3.3.1]
- **Reconstruir ítems del historial:** mitiga la pérdida de información en $h_{CLS}$ con secuencias largas. [§3.3.2]
- **MLM:** cierra la brecha semántica entre el conocimiento del PLM y el texto de recomendación. [§3.3.2]
- **Target EMA:** estabilidad y evitar el collapse. [§3.2]

---

## 5. Configuración experimental

**Instancia del problema**
- Preentrenamiento: Amazon Automotive, Grocery y Movies&TV (115K usuarios, 158K ítems, 1.25M interacciones). [Tabla 1]
- Fine-tuning: Scientific, Instruments, Arts, Office, Pet (Amazon) y Online Retail (Kaggle, Reino Unido); five-core. [§4.1, Tabla 1]
- Leave-one-out (último ítem para test, penúltimo para validación), ranking contra todos los ítems; métricas NDCG@10, Recall@10 y MRR. [§4.1]

**Detalles de implementación**
- Fine-tuning: batch 16, LR 5e-5, 32 tokens por atributo, 1024 por secuencia, máx. 50 ítems, τ = 0.05, λ₁ = 0.1. Preentrenamiento: batch 32, λ₂ = 0.1. Mismo setup que RecFormer. [Apéndice D]
- Hardware y tiempo de entrenamiento: no se aborda en el paper.

---

## 6. Resultados y discusión

**Resultados principales frente a baselines**

| Tarea / dataset | Métrica | Método | Mejor baseline |
|---|---|---|---|
| Scientific | N@10 | 0.1282 | 0.1198 RecFormer [Tabla 2] |
| Pet | N@10 / MRR | 0.1210 / 0.1157 | 0.1086 / 0.0940 RecFormer [Tabla 2] |
| Instruments | N@10 | 0.1057 | 0.1001 RecFormer [Tabla 2] |
| Arts | N@10 | 0.1442 | 0.1249 RecFormer [Tabla 2] |
| Office | N@10 | 0.1276 | 0.1151 RecFormer [Tabla 2] |
| Online Retail | R@10 / N@10 / MRR | 0.2429 / 0.1266 / 0.0985 | 0.2361 LlamaRec / 0.1249 / 0.0985 RecFormer (empate en MRR) [Tabla 2] |
| Zero-shot frente a RecFormer oficial (7 dominios) | N@10 | gana en Pet y Office; pierde en Instruments (0.0360 frente a 0.0416) y Arts (0.0610 frente a 0.0722) | [Tabla 5] |

**Rigor estadístico**
La tabla principal no tiene desviación estándar ni número de semillas. Solo el análisis del masking ratio se promedia sobre 5 ejecuciones [§4.4.2], también sin desviación estándar.

**¿Comparación justa?**
- Se adopta el setup de RecFormer para compararse con él. [§4.1]
- En el zero-shot de la Fig. 4 se reentrena RecFormer en los mismos 3 datasets. [§4.3]
- Los demás baselines siguen "prior work settings"; no se aclara si sus números son reproducidos o copiados.

**Ablaciones**
- Sin MLM: Scientific N@10 0.1170; Online Retail 0.1118. [Tabla 3]
- Sin preentrenamiento: 0.0935 / 0.1198. [Tabla 3]
- Sin token type: 0.1251 / 0.1231 (y Recall@10 en Online Retail baja de 0.2429 a 0.1755). [Tabla 3]
- Sin MLM ni posición de token: 0.1102; sin MLM, posición ni tipo: 0.0967. [Tabla 6]
- Sin la contrastiva en el preentrenamiento: 0.1216 / 0.1064; BPR en el fine-tuning: 0.1245 / 0.0934. [Tabla 7]
- Masking ratio del siguiente ítem: el óptimo está en torno al 50 % (probado de 10 % a 80 %). [Fig. 6]
- Robustez a metadatos incompletos (20–60 % de tokens eliminados): JEPA4Rec Scientific 0.1282 → 0.1201; RecFormer 0.1198 → 0.1088. [Tabla 4]

**Limitaciones que reconocen los autores**
No se aborda en el paper.

---

## 7. Conclusión y trabajo futuro (descendientes)

**Trabajo futuro propuesto**
No se aborda en el paper.

**Preguntas abiertas**
No se aborda en el paper.

---

## 8. Lectura crítica

> Esta sección es la valoración del lector, no lo que dice el paper.

**¿Contribuciones claramente separadas del trabajo previo?**
En parte. El backbone, el formato de ítems-como-texto, los cuatro embeddings, la loss contrastiva y el setup experimental vienen de RecFormer. La aportación real es el masking asimétrico (15 % / 50 %) y la loss L2 de mapping con target EMA, con un peso pequeño (λ₂ = 0.1). La lista de contribuciones presenta como propias piezas heredadas (los embeddings modificados).

**Afirmaciones sin respaldo o lenguaje grandilocuente**
- **Falta la ablation central:** no hay ninguna fila "sin $\mathcal{L}_{\text{map}}$" ni "sin target EMA". Todas las ablations quitan otras piezas (MLM, preentrenamiento, embeddings, contrastiva). Por tanto, no se demuestra que el componente JEPA cause la mejora; podría venir del masking del 50 % sobre el siguiente ítem (vía MLM) o de diferencias de entrenamiento con RecFormer.
- **"Outperforms other text-only models by 1–5% across all five datasets" en zero-shot** [§4.3, Fig. 4]: contrasta con la Tabla 5, donde frente al RecFormer oficial pierde en Instruments y Arts. Son comparaciones con checkpoints distintos, pero el mensaje del texto principal es más fuerte que la evidencia.
- **"Unlike contrastive learning, JEPA eliminates negative samples"** [§1], pero el método depende de una loss contrastiva, y quitarla empeora los resultados. [Tabla 7]
- **"Highlighting the robustness of contrastive language modeling"** [§4.3]: la propia frase atribuye el éxito a lo contrastivo, no a JEPA.
- **"Common-sense user preferences"** no se define operativamente ni se mide.

**Limitaciones no mencionadas por los autores**
- **Sin varianza:** las mejoras sobre RecFormer son de 0.005–0.02 en N@10 y no hay varianza.
- **Rareza en la Tabla 4:** RecFormer en Office *mejora* al eliminar el 20 % de los tokens (0.1151 → 0.1207), lo que sugiere ruido considerable entre ejecuciones.
- **Coste no reportado:** el target encoder y el predictor tienen un coste que no se reporta.
- **Nomenclatura inconsistente:** el dataset aparece como "Crafts" en §4.1 y como "Arts" en las tablas.

**¿Es coherente la historia X → Y → Z → 1?**
X → Y → Z es razonable. El salto Z → 1 no está cerrado: sin la ablation de la loss JEPA, el experimento no aísla la contribución que da nombre al paper.

**Relevancia para jepa-llm**
Media-baja. Es un ejemplo de JEPA en un encoder bidireccional (Longformer) con masking asimétrico, target EMA y predictor MLP sobre [CLS], combinado con MLM: una receta parecida a la que podría usarse en encoders, pero en un dominio aplicado (recomendación). Útil sobre todo como advertencia de diseño: incluir siempre la ablation "sin término JEPA".

**TL;DR propio**
JEPA4Rec extiende RecFormer (Longformer sobre ítems descritos como texto) con una loss L2 que predice, desde el [CLS] del historial, el embedding EMA de ítems parcialmente enmascarados, combinada con MLM y una loss contrastiva. Mejora unos puntos sobre RecFormer en 6 datasets de recomendación, pero sin varianza y sin la ablation que aísle el término JEPA. Para el proyecto basta con leer §3.3 en diagonal.
