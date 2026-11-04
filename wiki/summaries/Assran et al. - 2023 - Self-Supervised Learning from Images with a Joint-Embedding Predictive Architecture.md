# Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture (I-JEPA)

## 0. Ficha

| Campo | Valor |
|---|---|
| Autores | Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas (Meta AI FAIR, McGill, Mila, NYU) |
| Venue y año | CVPR 2023 (arXiv v3, abril de 2023) |
| Enlace (arXiv / DOI) | arXiv:2301.08243 |
| Código | No se indica en el paper |
| Fecha de lectura | 2026-11-04 |

---

## 1. El núcleo

**X — ¿Qué intentan hacer y por qué es relevante?**
Quieren aprender representaciones de imagen de alto nivel semántico sin usar data augmentations hechas a mano. Los métodos de invarianza (DINO, iBOT, SimCLR) dan buenas representaciones *off-the-shelf*, pero meten sesgos fuertes que no sirven para todas las tareas y no se trasladan fácilmente a otras modalidades. Los métodos generativos (MAE, BEiT) necesitan menos conocimiento previo, pero sus representaciones son de nivel más bajo y requieren fine-tuning completo para rendir [§1].

**Y — ¿Por qué es difícil?**
- Reconstruir en píxeles obliga a modelar detalles irrelevantes, y las representaciones salen poco semánticas: MAE rinde mal en linear probing [§1, Tabla 1].
- Predecir en espacio latente abre la puerta al collapse de representaciones, igual que en las arquitecturas joint-embedding [§2].
- Sin augmentations, algo tiene que empujar al modelo hacia la semántica. Los autores muestran que el diseño del enmascarado es decisivo: con *block* o *random* masking el Top-1 cae de 54.2 a 20.2 o 17.6 [Tabla 6].

**Z — ¿Cuál es su contribución?**
I-JEPA es una instancia para imágenes de la JEPA de LeCun. A partir de **un único bloque de contexto**, un predictor estrecho, condicionado en mask tokens con positional embeddings, predice las **representaciones** de **varios bloques objetivo** de la misma imagen. Los targets salen de un target encoder EMA, y se enmascara la *salida* del target encoder, no su entrada. La pieza clave es la estrategia **multi-block**: 4 targets grandes (escala 0.15–0.2) y un contexto informativo, espacialmente distribuido (escala 0.85–1.0), del que se quitan las zonas que solapan con los targets [§3, Fig. 3–4].

**1a — ¿Cómo lo verifican experimentalmente?**
- Preentrenan ViT-B, L, H y G en ImageNet-1K e IN-22K.
- Evalúan con:
  - linear probing en ImageNet-1K;
  - low-shot en ImageNet-1%;
  - transfer lineal a CIFAR100, Places205 e iNat18;
  - tareas de bajo nivel (Clevr/Count y Clevr/Dist);
  - fine-tuning completo.
- Miden el coste en GPU-horas.
- Resultado titular: ViT-H/16 a 448 px llega a **81.1 %** en linear eval de ImageNet, frente a 81.0 de iBOT ViT-L/16 (que sí usa augmentations) y 77.2 de MAE ViT-H/14. Esa ViT-H/14 se entrena en menos de 72 h con 16 A100 [Tabla 1, Fig. 1, Abstract].

**1b — ¿Hay respaldo teórico?**
No. Solo un encuadre conceptual en Energy-Based Models (JEA, generativas y JEPA, Fig. 2). No hay análisis formal de por qué no colapsa [§2].

**2 — Contribuciones que reclaman los autores**
- I-JEPA aprende representaciones *off-the-shelf* fuertes sin augmentations de vistas y supera a MAE en linear probing de ImageNet-1K, en ImageNet-1% semisupervisado y en transfer semántico [§1].
- Es competitivo con los métodos de invarianza en tareas semánticas y mejor en tareas de bajo nivel (conteo y profundidad) [§1, §5, §6].
- Escala y es eficiente: una ViT-H/14 necesita menos de 1200 GPU-horas, más de 2.5× menos que una ViT-S/16 con iBOT y más de 10× menos que una ViT-H/14 con MAE [§1, §7].

---

## 2. Trabajos relacionados (hermanos)

| Trabajo | Diferencia (supuestos / método) | ¿Comparado en experimentos? | Si no, ¿por qué? |
|---|---|---|---|
| MAE | Reconstruye píxeles enmascarados; el encoder solo procesa los parches visibles (I-JEPA hereda esa eficiencia) | Sí (Tablas 1–5, 15; Fig. 5) | — |
| data2vec | Predice representaciones de un target encoder EMA a partir de la entrada enmascarada, sin augmentations, en varias modalidades. I-JEPA se presenta como más eficiente y más semántico | Sí (Tablas 1–4; Fig. 1, 5) | — |
| Context Autoencoders (CAE) | Encoder–decoder con suma de una loss de reconstrucción y una restricción de alineamiento en espacio latente | Sí (Tabla 1) | — |
| data2vec-v2 | Concurrente; arquitecturas eficientes multimodales | No | Se cita como concurrente |
| BEiT / SimMIM / MaskFeat | Reconstruyen tokens discretos de un dVAE o features HOG, no un espacio aprendido | No | No se justifica |
| DINO / iBOT / MSN | Joint-embedding con augmentations de vistas; iBOT añade una loss de reconstrucción por parche al estilo data2vec | Sí (Tablas 1–4) | — |
| SimCLR v2 / BYOL | Contrastivo / no contrastivo con augmentations, sobre ResNet | Sí (Tablas 1–2) | — |
| CPC (contrastive predictive coding) | Contrastivo entre parches solapados con CNN; I-JEPA no es contrastivo y predice desde un único contexto | No | Se discute solo en el Apéndice B |
| VICReg / Barlow Twins | Regularización explícita anti-collapse (redundancia y varianza) | No | Se mencionan solo como familia de métodos |

---

## 3. Antecedentes (ancestros)

**Conceptos previos necesarios**
- Energy-Based Models aplicados a SSL: energía baja para pares compatibles $(x, y)$ [§2].
- Joint-Embedding Architectures y el problema del collapse; mecanismos que lo evitan: contrastivo, reducción de redundancia, clustering y asimetría arquitectónica (EMA, predictor) [§2, Apéndice B].
- JEPA, propuesta por LeCun ("A path towards autonomous machine intelligence") [§2].
- ViT, masked image modelling y teacher EMA (mean teacher, BYOL, DINO) [§3].

**Definición del problema** (escenario formal y notación clave)
- Una imagen $y$ se divide en $N$ parches y pasa por el target encoder $f_{\bar\theta}$: $s_y = \{s_{y_1}, \dots, s_{y_N}\}$.
- Se muestrean $M$ bloques objetivo con máscaras $B_i$, cuyos targets son $s_y(i) = \{s_{y_j}\}_{j \in B_i}$.
- El bloque de contexto $x$ (máscara $B_x$) pasa por el context encoder $f_\theta$ y da $s_x$.
- El predictor $g_\phi(s_x, \{m_j\}_{j \in B_i})$ produce $\hat s_y(i)$.
- La loss es
$$\frac{1}{M}\sum_{i=1}^{M}\sum_{j\in B_i}\lVert \hat s_{y_j} - s_{y_j}\rVert_2^2$$
[§3].

**Supuestos inusuales o restrictivos**
- Las vistas son siempre recortes espaciales de la misma imagen; el muestreo de bloques asume estructura 2D [§3, Fig. 4].
- Se evalúa siempre con la salida del target encoder y average pooling, porque no hay token [cls] [Apéndice A.1].

**¿El escenario del problema es una contribución nueva?**
No. Es el SSL de imágenes estándar con preentrenamiento en ImageNet. Lo nuevo es la instancia JEPA y la estrategia de masking.

---

## 4. Método

**Qué hacen** (paso a paso)
1. Pasan la imagen completa por el target encoder $f_{\bar\theta}$ (ViT) y obtienen representaciones por parche [§3].
2. Muestrean $M = 4$ bloques objetivo, posiblemente solapados, con escala en (0.15, 0.2) y aspect ratio en (0.75, 1.5). Los targets se toman **de la salida** del target encoder [§3].
3. Muestrean un bloque de contexto con escala en (0.85, 1.0) y aspect ratio 1. Le quitan los parches que solapan con cualquier target, de modo que queda en torno al 25 % de la imagen [§3, Tabla 6].
4. El context encoder $f_\theta$ procesa solo los parches visibles del contexto [§3].
5. Para cada target, el predictor (un ViT estrecho) recibe $s_x$ más un mask token compartido con el positional embedding de cada parche objetivo, y predice sus representaciones. Se ejecuta $M$ veces [§3].
6. La loss es la L2 media entre predicción y target. $\theta$ y $\phi$ se actualizan por gradiente; $\bar\theta$ se actualiza por EMA de $\theta$ [§3].

**Por qué lo hacen así**
- **Predecir en espacio latente:** el target encoder puede eliminar detalle irrelevante de nivel píxel y producir targets abstractos [§1, §9]. Respaldo: predecir píxeles da 40.7 frente a 66.9 [Tabla 7].
- **Targets grandes:** bloques de escala 0.15–0.2 son "semánticos". Respaldo: con escalas más pequeñas o más grandes el resultado empeora [Tabla 8].
- **Contexto grande y distribuido:** hace la predicción factible y es barato de procesar. Respaldo: reducir el contexto degrada el resultado [Tabla 9].
- **Varios targets:** pasar de 1 a 4 targets lleva el Top-1 de 9.0 a 54.2 [Tabla 10].
- **Enmascarar la salida del target encoder, no la entrada:** produce targets más semánticos; 67.3 frente a 56.1 [Tabla 11, Apéndice C]. Siguen el argumento de data2vec.
- **EMA y predictor asimétrico:** evitan el collapse, con la justificación de que el EMA "ha demostrado ser esencial" para JEAs con ViT en trabajos previos [§3]. No lo comprueban con una ablation propia.
- **Predictor estrecho y profundo:** un cuello de botella de 384 canales da 70.7 frente a 68.4 con 1024 [Tabla 14]; profundidad 12 da 66.9 frente a 64.0 con profundidad 6 [Tabla 12].

---

## 5. Configuración experimental

**Instancia del problema**
- Preentrenamiento en ImageNet-1K (y en IN-22K para el escalado) [§5, §7].
- Evaluación:
  - linear probing en ImageNet-1K;
  - ImageNet-1 % (12–13 imágenes por clase);
  - linear transfer a CIFAR100, Places205 e iNat18;
  - Clevr/Count y Clevr/Dist;
  - fine-tuning completo en ImageNet [§5, §6, Apéndice D].

**Detalles de implementación**
- **Arquitectura:**
  - Encoders ViT-B/16, L/16, H/14, H/16 a 448 px y G/16.
  - Predictor con 384 dimensiones y profundidad 6 (B), 12 (L y H) o 16 (G).
  - Sin token [cls] [Apéndice A.1].
- **Optimización:**
  - AdamW, batch de 2048.
  - Learning rate de 1e-4 a 1e-3 con warmup de 15 epochs, después coseno hasta 1e-6.
  - Weight decay de 0.04 a 0.4.
  - Momentum del EMA de 0.996 a 1.0 [Apéndice A.1].
- **Epochs:** 300 para H, 600 para B y L, frente a 1600 en MAE y data2vec [Tablas 1–2].
- **Linear eval (VISSL):** reportan el mejor resultado entre varias combinaciones de representación (última capa o concatenación de las últimas 4) y de cabeza (lineal o BN más lineal). En ImageNet usan LARS con batch de 16384, 50 epochs y un barrido de 3 learning rates × 2 weight decays [Apéndice A.2].
- **Low-shot:** fine-tuning de 50 epochs al estilo MAE, con RandAugment, sin mixup ni cutmix. Para cada método se usa el protocolo, fine-tuning o linear probing, que mejor le funcione [Tabla 2, Apéndice A.2].
- **Cómputo:** la ViT-H/14 se entrena con 16 A100 en menos de 72 h (menos de 1200 GPU-horas). Cada iteración es un 7 % más lenta que en MAE, pero converge en unas 5× menos iteraciones [Abstract, §7].

---

## 6. Resultados y discusión

**Resultados principales frente a baselines**

| Tarea / dataset | Métrica | Método | Mejor baseline |
|---|---|---|---|
| ImageNet-1K linear | Top-1 | 79.3 (H/14, 300 ep); 81.1 (H/16 a 448 px) | Sin augmentations: CAE ViT-L 78.1. Con augmentations: iBOT ViT-L 81.0 [Tabla 1] |
| ImageNet-1% | Top-1 | 73.3 (H/14); 77.3 (H/16 a 448 px) | data2vec ViT-L 73.3; MSN ViT-B/4 75.7 [Tabla 2] |
| CIFAR100 linear | Top-1 | 87.5 | iBOT 88.3; MAE 77.3 [Tabla 3] |
| Places205 linear | Top-1 | 58.4 | iBOT 60.4; MAE 55.0 [Tabla 3] |
| iNat18 linear | Top-1 | 47.6 | iBOT 57.3; MAE 32.9 [Tabla 3] |
| Clevr/Count | Acc. | 86.7 | MAE 90.5; DINO 86.6 [Tabla 4] |
| Clevr/Dist | Acc. | 72.4 | MAE 72.4; iBOT 62.8 [Tabla 4] |
| Fine-tuning ImageNet | Top-1 | 87.1 (H/16 a 448 px, 300 ep) | MAE 87.8 (H/14 a 448 px, 1600 ep) [Tabla 15] |

- Escalado:
  - con IN-22K, la ViT-H/14 mejora en CIFAR100 (89.5), iNat18 (50.5) y Clevr (88.6 y 75.0);
  - la ViT-G/16 sube en Places e iNat (59.1 y 55.3), pero no en las tareas de bajo nivel [Tabla 5].
- Visualizaciones con RCDM: el predictor captura la incertidumbre posicional y partes de objetos con la pose correcta, y descarta el fondo y los detalles finos. Es un resultado cualitativo [§8, Fig. 6–8].

**Rigor estadístico**
No hay desviación estándar, intervalos de confianza ni número de semillas. Cada número viene de una sola ejecución.

**¿Comparación justa?**
- **Distintos protocolos de evaluación:**
  - en low-shot, cada método se evalúa con el protocolo que mejor le va [Tabla 2];
  - en linear eval, se toma el máximo entre varias representaciones y cabezas para todos los métodos [Apéndice A.2].
- **Arquitecturas no emparejadas:** las tablas comparan arquitecturas y epochs distintos, por ejemplo I-JEPA ViT-H/14 frente a data2vec ViT-L/16 o frente a iBOT ViT-L/16.
- **Baselines evaluados por los autores:** las comparaciones de transfer (Tablas 3–4) son evaluaciones propias con VISSL de los checkpoints de cada método.

**Ablaciones** (todas con linear probe en ImageNet-1 %, una sola ejecución)
- **Estrategia de masking:** multi-block 54.2; rasterized 15.5; block 20.2; random 17.6 (ViT-B/16, 300 ep) [Tabla 6].
- **Target:** representación 66.9 (500 ep) frente a píxeles 40.7 (800 ep), con ViT-L/16 [Tabla 7].
- **Escala de los targets:** el óptimo, 54.2, está en (0.15, 0.2); cae a 19.2 con (0.075, 0.2) y a 33.6 con (0.2, 0.3) [Tabla 8].
- **Escala del contexto:** 31.2 con (0.40, 1.0) frente a 54.2 con (0.85, 1.0) [Tabla 9].
- **Número de targets:** 9.0, 22.0, 48.5 y 54.2 con 1, 2, 3 y 4 targets [Tabla 10].
- **Dónde se enmascara el target:** en la salida 67.3, en la entrada 56.1 (ViT-H/16) [Tabla 11].
- **Predictor:** con profundidad 12, 66.9 frente a 64.0 con 6 [Tabla 12]; con ancho 384, 70.7 frente a 68.4 con 1024 (fine-tuning) [Tabla 14].
- **Weight decay:** un weight decay creciente, de 0.04 a 0.4, mejora el linear eval (77.8 frente a 76.4) pero empeora el fine-tuning en el 1 % (69.4 frente a 70.7) [Tabla 13].

**Limitaciones que reconocen los autores**
- La ViT-G/16 no mejora en tareas de bajo nivel, posiblemente por usar parches más grandes [§7].
- Con el weight decay hay que elegir entre rendimiento en linear eval y en fine-tuning [Apéndice C].
- Al fine-tunear en todo ImageNet queda por debajo de MAE, aunque con 5.3× menos epochs [Apéndice D].

---

## 7. Conclusión y trabajo futuro (descendientes)

**Trabajo futuro propuesto**
- No se aborda en el paper. La conclusión solo dice que I-JEPA "señala un camino" para aprender representaciones generales con joint-embedding sin augmentations [§10].

**Preguntas abiertas**
- No se aborda en el paper.

---

## 8. Lectura crítica

> Esta sección es la valoración del lector, no lo que dice el paper.

**¿Contribuciones claramente separadas del trabajo previo?**
En general sí.
- Atribuyen la JEPA a LeCun, el teacher EMA a BYOL, DINO y MSN, el encoder que solo procesa parches visibles a MAE, y el enmascarado de la salida del target encoder a data2vec [§3, §4].
- La contribución real es estrecha: la estrategia multi-block y el predictor condicionado en posición. La distancia con data2vec es pequeña y no está bien aislada. Los cambios son tres: target de una sola capa frente a la media de las últimas $K$, predictor separado frente a cabeza lineal, y multi-block frente a masking por bloques. Ninguna ablation los separa uno a uno.

**Afirmaciones sin respaldo o lenguaje grandilocuente**
- "Highly scalable" [Abstract] se apoya en 3–4 puntos por familia de modelos (Fig. 5, Tabla 5), sin curvas de escalado propiamente dichas.
- "Simpler model with less rigid inductive bias, applicable to a wider set of tasks" [§1]. Pero el multi-block es un prior espacial 2D muy específico: con *random* o *block* masking el método casi no funciona (17.6 y 20.2 frente a 54.2, Tabla 6). Se sustituye un prior de augmentations por otro de masking, y no se prueba ninguna otra modalidad.
- "The use of an EMA target-encoder has proven essential" [§3]: no hay ablation sin EMA ni con un predictor trivial. No se demuestra qué evita el collapse en I-JEPA.
- **Bajo nivel:** en Clevr/Count dicen "outperforms" frente a los métodos de invarianza [§6], pero MAE lo supera claramente (90.5 frente a 86.7, Tabla 4). La frase solo es cierta frente a DINO e iBOT.
- **Pixel frente a representación:** la comparación de la Tabla 7 usa 500 frente a 800 epochs. Esa diferencia no favorece a I-JEPA, pero sigue siendo un par no controlado.

**Limitaciones no mencionadas por los autores**
- Una sola semilla en todo el paper. Algunas diferencias de las ablations son de 2–3 puntos (Tablas 12, 14) y tienen ruido desconocido.
- **Elección del mejor protocolo por método:** el "best of" en evaluación (representación × cabeza en linear eval; fine-tuning o linear en low-shot) favorece a quien más combinaciones prueba y dificulta comparar entre métodos.
- **Hiperparámetros de masking ajustados sobre la propia evaluación:** se eligen con el linear probe de ImageNet-1 %, que luego es métrica principal (Tabla 2). Hay riesgo de sobreajuste al benchmark.
- **Tabla 14 inconsistente con la Tabla 13:** el modelo por defecto, con predictor de 384, da 70.7 en la Tabla 14, pero ese valor corresponde al weight decay fijo de 0.05 en la Tabla 13; la configuración por defecto (0.04 → 0.4) da 69.4. No queda claro con qué configuración está hecha la ablation de ancho.
- **Erratas menores:** el pie de la Tabla 9 dice "target size" cuando ablaciona el contexto; "(015, 0.2)" en el Apéndice A.1.
- **Sin análisis del collapse ni del espectro:** no se mide el effective rank ni la varianza de las representaciones. Es justo lo que el JEPA Paradox y Sengupta señalan como necesario en texto.
- **Relevancia para este proyecto:** no hay ningún experimento en texto. En texto, el análogo de "bloques grandes y semánticos" serían spans largos, y el JEPA Paradox argumenta que ahí la predicción con L2 converge a un centroide (multimodalidad). I-JEPA no aborda la multimodalidad del target; la sortea con contexto muy informativo (85–100 %).

**¿Es coherente la historia X → Y → Z → 1?**
Sí. X (semántica sin augmentations) → Y (los píxeles son demasiado bajo nivel y el latente arriesga collapse) → Z (predicción latente con multi-block) → 1 (linear probe, low-shot, transfer y ablations de masking). El hueco está en Y: el riesgo de collapse se presenta como central, pero nunca se mide ni se ablaciona.

**TL;DR propio**
I-JEPA predice, a partir de un contexto grande, las representaciones EMA de 4 bloques grandes de la misma imagen. Con eso supera a MAE y data2vec en linear probing con mucho menos cómputo y alcanza a iBOT a escala H/448. El ingrediente decisivo es el diseño de las máscaras (multi-block, targets grandes, contexto informativo), no la predicción latente por sí sola, y faltan semillas y una ablation que muestre qué evita el collapse. Merece la pena leerlo a fondo (§3, §9 y Apéndice C) como referencia canónica de JEPA y como plantilla de qué ablations de masking replicar en texto.
