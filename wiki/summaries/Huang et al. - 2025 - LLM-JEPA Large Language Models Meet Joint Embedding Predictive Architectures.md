# LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures

## 0. Ficha

| Campo | Valor |
|---|---|
| Autores | Hai Huang (Atlassian), Yann LeCun (NYU), Randall Balestriero (Brown University) |
| Venue y año | Preprint, arXiv 2025 (v2, 7 oct 2025) |
| Enlace (arXiv / DOI) | https://arxiv.org/abs/2509.14252 |
| Código | https://github.com/rbalestr-lab/llm-jepa |
| Fecha de lectura | 2026-11-04 |

> Nota: el PDF se leyó como texto extraído (pypdf); las figuras (t-SNE, heatmaps de la Fig. 7) no se vieron como imagen.

---

## 1. El núcleo

**X — ¿Qué intentan hacer y por qué es relevante?**
Llevar a los LLMs (decoders autoregresivos) un objetivo de tipo JEPA, es decir, de predicción en espacio de embeddings, que en visión ha mostrado ser superior a los objetivos de reconstrucción en el espacio de entrada. Los LLMs se entrenan y evalúan casi exclusivamente con reconstrucción de tokens. [Abstract, §1]

**Y — ¿Por qué es difícil?**
- Los LLMs se juzgan por su capacidad de generar texto, así que un objetivo JEPA no puede sacrificar la generación. [§1]
- En texto no existe un equivalente obvio a la data augmentation de visión para fabricar "vistas" no triviales del mismo contenido. [§1]
- Con atención causal no se pueden obtener los embeddings de dos vistas independientes en un único forward sin que una dependa de la otra. [§3.2]

**Z — ¿Cuál es su contribución?**
Añadir a la loss de next-token prediction un término JEPA: $\lambda \cdot d(\text{Pred}(\text{Enc}(\text{Text})), \text{Enc}(\text{Code}))$. Text y Code son dos vistas naturales del mismo conocimiento (p. ej. una descripción en lenguaje natural y la regex o el SQL correspondiente). Detalles:
- **Encoder:** el hidden state del último token de la última capa del propio LLM.
- **Predictor:** el mismo LLM, con *k* tokens [PRED] añadidos al final ("tied-weights predictor").
- **Distancia:** coseno.
- **Implementación:** una máscara de atención causal por bloques permite calcularlo con un forward extra. [§3.1, §3.2]

**1a — ¿Cómo lo verifican experimentalmente?**
- Fine-tuning de 4 familias (Llama-3.2-1B, gemma-2-2b, OpenELM-1.1B, OLMo-2-1B) en NL-RX-SYNTH, NL-RX-TURK, GSM8K y Spider. Resultado titular: Llama-3.2-1B en NL-RX-SYNTH pasa de 57.29 ± 5.32 a 71.46 ± 1.34. [Tablas 12, 13]
- Escalas de 1B a 8B, LoRA, 6 épocas (resistencia a overfitting). [Tablas 8, 15; Figs. 1, 8]
- Preentrenamiento desde cero en NL-RX-SYNTH (54.38 → 60.59) y en paraphrase, evaluado después en Rotten Tomatoes y Yelp. [Tablas 2, 9]
- Datasets sin dos vistas naturales (NQ-Open, HellaSwag) y modelos de razonamiento (Qwen3-1.7B, R1-Distill-1.5B en GSM8K). [Tablas 4, 5]
- Siempre 5 semillas y t-test pareado de una cola. [§4]

**1b — ¿Hay respaldo teórico?**
No se aborda en el paper. Solo hay análisis empírico de la geometría: SVD de $\text{Enc(Text)} - \text{Enc(Code)}$ y error de una regresión lineal entre ambos embeddings. [§4.2, Tabla 14]

**2 — Contribuciones que reclaman los autores**
- El primer objetivo JEPA para LLMs en espacio de embeddings con vistas distintas, "siguiendo perfectamente los JEPA de visión", sin sacrificar la generación. [§1]
- Mejoras sobre el fine-tuning estándar ("Improved SOTA") y resultados prometedores en preentrenamiento. [§1]
- Validación extensa sobre familias de modelos, datasets y tamaños. [§1]

---

## 2. Trabajos relacionados (hermanos)

| Trabajo | Diferencia (supuestos / método) | ¿Comparado en experimentos? | Si no, ¿por qué? |
|---|---|---|---|
| SimCSE (Gao et al., 2021) | Contrastivo en espacio latente con vistas por dropout; solo produce embeddings, no genera | No | Producen embeddings sin capacidad generativa [§2] |
| E5 (Wang et al., 2022), Sentence-T5 (Ni et al., 2021) | Joint embedding con pares débilmente supervisados o modelos preentrenados; no generativos | No | Ídem [§2] |
| Sentence-BERT (Reimers & Gurevych, 2019) | MLM más una loss semántica latente sobre BERT; no generativo | No | Al construirse sobre BERT no permite evaluación generativa [§2] |
| Large Concept Models (Barrault et al., 2024); Wang & Sun (2025) | Objetivos o regularizadores en espacio latente con restricciones estructurales (jerarquías, clusters) | No | "Quedan fuera del alcance JEPA" [§3.2] |
| data2vec, I-JEPA, V-JEPA | JEPAs de visión y multimodales, inspiración directa | No | No se discute; son de otras modalidades o encoders |
| Trabajos NL→Regex, NL→SQL, issue→diff | Usan las mismas "vistas" pero solo para generar una a partir de la otra, sin loss JEPA | Sí, implícitamente: el baseline es este fine-tuning generativo | — |
| InfoNCE (como alternativa a la distancia) | Loss contrastiva en lugar de coseno | Sí, como ablation: 34.40 ± 6.10 [Tabla 3] | — |

---

## 3. Antecedentes (ancestros)

**Conceptos previos necesarios**
- JEPA: predecir el embedding de una vista a partir de otra evitando el collapse (I-JEPA, V-JEPA, data2vec). [§1]
- Loss autoregresiva de next-token prediction. [Eq. 1]
- Máscaras de atención aditivas en HuggingFace. [§3.1]
- Resultado previo de Balestriero & LeCun (2024): aprender por reconstrucción da features poco informativas para percepción. [§3.3]

**Definición del problema**
$$\mathcal{L}_{\text{LLM-JEPA}} = \sum_{\ell=2}^{L} \mathcal{L}_{\text{LLM}}(\text{Text}_{1:\ell-1}, \text{Text}_\ell) + \lambda \cdot d\big(\text{Pred}(\text{Enc}(\text{Text})), \text{Enc}(\text{Code})\big)$$
Aquí Enc es el último hidden state del último token, Pred son *k* tokens [PRED] procesados por el mismo LLM y $d$ es la distancia coseno. [Eq. 2] En el apéndice se añade un peso $\gamma$ al término LLM. [Eq. 3]

**Supuestos inusuales o restrictivos**
- Requiere datasets con dos "vistas" no triviales del mismo contenido (Text/Code). Los autores lo señalan como crucial y limitante. [§1]
- El encoder es el propio LLM; no hay un teacher EMA separado ni stop-gradient explícito (el paper no menciona ninguno de los dos).
- El target es un solo vector por secuencia (el último token), no representaciones por token.

**¿El escenario del problema es una contribución nueva?**
Sí, en parte: interpretar los pares (texto, código) de un dataset de fine-tuning como dos vistas para un JEPA es la idea central que reclaman. [§1]

---

## 4. Método

**Qué hacen**
1. Empaquetan Text y Code en una única ventana de contexto, con una máscara causal por bloques para que no se atiendan entre sí. [§3.1, §3.2]
2. $\text{Enc(Text)}$ y $\text{Enc(Code)}$ son los hidden states del último token de cada bloque en la última capa. [§3.1]
3. Añaden *k* ∈ {0,…,4} tokens [PRED] tras Text; el embedding del último es $\text{Pred(Enc(Text))}$. Con *k* = 0, Pred es la identidad. [§3.1]
4. Loss total = next-token prediction (forward normal sobre el prompt completo) + λ·(1 − cos) entre la predicción y $\text{Enc(Code)}$. Son dos forwards en lugar de tres gracias a la máscara. [§3.2]
5. Variante *loss dropout* (LD): con probabilidad α se omite el término JEPA en el batch y se ahorra el forward extra; coste por época de $(2-\alpha)\times$. [§5.2]

**Por qué lo hacen así**
- **Mantener $\mathcal{L}_{\text{LLM}}$:** preserva la generación. Con γ = 0 el modelo solo genera salidas vacías. [§3.1, Tabla 10]
- **Coseno:** es lo habitual en visión. L2 colapsa (2.22 %), MSE da 70.64 % e InfoNCE 34.40 %. [§3.1, Tabla 3]
- **Predictor con pesos atados:** reduce el overhead y las decisiones de diseño. [§3.1]
- **Dirección Text→Code y [PRED] añadido al final:** son mejores que Code→Text (65.70) y que añadirlo al principio (68.07). [Tabla 3]
- **Necesidad del término JEPA:** minimizar solo $\mathcal{L}_{\text{LLM}}$ no reduce la distancia JEPA, y añadirlo no empeora la cross-entropy. [§3.3, Fig. 3]

---

## 5. Configuración experimental

**Instancia del problema**
- NL-RX-SYNTH y NL-RX-TURK (NL→regex; exact match), GSM8K (exact match del resultado final), Spider (NL→SQL; exact match de la ejecución). [§4.1]
- Preentrenamiento: NL-RX-SYNTH desde cero (métrica *startswith*); cestwc/paraphrase (5 paráfrasis, la *i*-ésima predice la *i*+1), con fine-tuning posterior sin JEPA en Rotten Tomatoes y Yelp (*prefix match* de la etiqueta). [§4.1]
- NQ-Open (pregunta → respuesta; substring match) y HellaSwag (contexto → continuación; probabilidad de las opciones). [§5.1]

**Detalles de implementación**
- Modelos: Llama-3.2-1B/3B-Instruct, Llama-3.1-8B-Instruct, gemma-2-2b-it, OpenELM-1.1B-Instruct, OLMo-2-1B/7B-Instruct, Qwen3-1.7B, DeepSeek-R1-Distill-Qwen-1.5B. [§4.1, §5.1, Tabla 15]
- LR elegido con el baseline en {1e-5, 2e-5, 4e-5, 8e-5} tras 4 épocas; luego un grid (k, λ) ∈ {0..4} × {0.5, 1, 2, 4} para LLM-JEPA con el mismo LR. [§4.1]
- 5 semillas fijas {82, 23, 37, 84, 4}. [§4]
- Fine-tuning completo y LoRA (rangos de 32 a 512). [Tabla 8]
- Hardware y tiempo de entrenamiento: no se aborda en el paper. Solo hay PFLOPs para un caso. [Fig. 5]

---

## 6. Resultados y discusión

**Resultados principales frente a baselines** (el baseline es siempre el fine-tuning con $\mathcal{L}_{\text{LLM}}$ en el mismo modelo)

| Tarea / dataset | Métrica | Método | Mejor baseline |
|---|---|---|---|
| NL-RX-SYNTH, Llama-3.2-1B | acc. % | 71.46 ± 1.34 | 57.29 ± 5.32 [Tabla 12] |
| NL-RX-SYNTH, gemma-2-2b | acc. % | 43.12 ± 2.61 | 33.65 ± 3.24 [Tabla 12] |
| NL-RX-SYNTH, OpenELM-1.1B | acc. % | 25.40 ± 2.40 | 12.07 ± 1.81 [Tabla 12] |
| NL-RX-SYNTH, OLMo-2-1B | acc. % | 87.52 ± 0.29 | 87.09 ± 0.36 [Tabla 12] |
| NL-RX-TURK, Llama-1B | acc. % | 30.94 ± 1.13 | 22.49 ± 1.91 [Tabla 13] |
| GSM8K, Llama-1B | acc. % | 36.36 ± 0.20 | 32.36 ± 0.58 [Tabla 13] |
| Spider, Llama-1B | acc. % | 50.55 ± 2.08 | 47.52 ± 2.44 [Tabla 13] |
| NL-RX-SYNTH, Llama-3.1-8B | acc. % | 63.57 ± 16.81 | 35.77 ± 6.60 [Tabla 15] |
| NL-RX-SYNTH, Llama-3B / OLMo-7B | acc. % | 77.16 / 87.75 | 74.55 / 87.26 [Tabla 15] |
| Preentrenamiento NL-RX-SYNTH | acc. % (startswith) | 60.59 ± 1.01 | 54.38 ± 1.70 [Tabla 2] |
| Paraphrase → Rotten Tomatoes / Yelp | acc. % | 57.76 / 27.15 | 56.57 / 26.46 [Tabla 9] |
| NQ-Open / HellaSwag | acc. % | 21.59 / 70.51 | 20.12 / 69.40 [Tabla 4] |
| GSM8K, Qwen3-1.7B / R1-Distill-1.5B | acc. % | 45.00 / 15.04 | 44.32 / 13.87 [Tabla 5] |

**Rigor estadístico**
5 semillas por configuración, media ± desviación estándar y t-test pareado de una cola con p-valores (todos < 0.05). En preentrenamiento + fine-tuning: 5 × 5 ejecuciones. [§4, Tablas 2, 9]

**¿Comparación justa?**
- El LR se ajusta para el baseline y se reutiliza para LLM-JEPA. [§4.1]
- Pero LLM-JEPA recibe además un grid de 20 configuraciones (k, λ), y las tablas reportan "the best possible accuracy over a set of configurations". [Tablas 12, 13] El paper no indica si esa selección se hace en un conjunto separado del de evaluación.
- El cómputo no se iguala salvo en la Fig. 5 y la Tabla 6 (loss dropout, con PFLOPs igualados en un solo setup).

**Ablaciones**
- Distancia: coseno 71.46 > MSE 70.64 > InfoNCE 34.40 > L2 2.22. [Tabla 3]
- Posición de [PRED]: al final (71.46) es mejor que al principio (68.07); la dirección Text→Code es mejor que Code→Text (65.70). [Tabla 3]
- Peso de $\mathcal{L}_{\text{LLM}}$ (γ/λ): γ = 0 da 0 % (salidas vacías); γ/λ = 1 es lo óptimo (70.42); sin JEPA, 57.29. [Tabla 10]
- No hay un patrón claro del (k, λ) óptimo, aunque las celdas adyacentes rinden parecido. [Fig. 7]
- LoRA: la ganancia se mantiene en todos los rangos; con rango 512 iguala al fine-tuning completo. [Tabla 8]
- Loss dropout: con α = 0.5–0.75 se obtiene más accuracy al mismo número de PFLOPs; la heurística es mantener $\lambda(1-\alpha)$ constante. [Fig. 5, Tabla 6]
- En HellaSwag, λ puede escalar hasta 1024 sin llegar a un plateau. [§5.1, Fig. 9]

**Limitaciones que reconocen los autores**
- El coste de entrenamiento se duplica (mitigado con loss dropout). [§6]
- Añade dos hiperparámetros (k, λ) y el óptimo puede estar en cualquier punto del grid, lo que encarece el tuning. [Limitations]
- Necesita datasets con vistas no triviales; haría falta un equivalente a la data augmentation para aplicarlo a cualquier dataset. [§1]
- El preentrenamiento es preliminar y a pequeña escala; el modelo no aprende a terminar la generación. [§4.1, §6]

---

## 7. Conclusión y trabajo futuro (descendientes)

**Trabajo futuro propuesto**
- Escalar y probar más a fondo el preentrenamiento con LLM-JEPA. [§6]
- Encontrar un mecanismo análogo a la data augmentation para crear vistas en cualquier dataset. [§1]
- Un algoritmo de tuning más eficiente para (k, λ). [Limitations]
- Combinar loss dropout y la máscara de atención para preentrenar a escala completa con poco overhead. [§5.2]

**Preguntas abiertas**
- Por qué funciona: los autores lo atribuyen a una transformación casi lineal entre Enc(Text) y Enc(Code), pero lo presentan como hipótesis ("may underlie"). [§4.2]

---

## 8. Lectura crítica

> Esta sección es la valoración del lector, no lo que dice el paper.

**¿Contribuciones claramente separadas del trabajo previo?**
En parte. La idea de usar pares (texto, código) como vistas y el predictor de pesos atados con tokens [PRED] están bien identificados como propios. Pero falta discutir data2vec, que ya hizo predicción latente en texto, y otros trabajos de JEPA en NLP: data2vec aparece en la bibliografía pero no se compara ni se discute. La afirmación de ser "el primer" objetivo JEPA para LLMs depende de restringir "LLM" a decoders generativos.

**Afirmaciones sin respaldo o lenguaje grandilocuente**
- "Exact replicate of the JEPA objective extensively used in vision" [§6]: no hay teacher EMA, stop-gradient ni mecanismo anti-collapse explícito, y el target es un único vector de secuencia. Es un regularizador de alineamiento entre dos vistas, más cercano a un objetivo siamés no contrastivo que a I-JEPA.
- "Improved SOTA" [§1]: no se compara con ningún SOTA externo, solo con el fine-tuning estándar del mismo modelo.
- "Far superior" (los JEPA en visión) [Abstract]: es una generalización fuerte, sin cita concreta.
- "Robust to overfitting" se apoya en curvas de 6 épocas en un solo setup. [Fig. 1, Fig. 8]
- La explicación de la mejora (el término pred "should be the main reason") es correlacional. [Fig. 3]

**Limitaciones no mencionadas por los autores**
- **Sesgo de selección:** LLM-JEPA elige el mejor de 20 configuraciones (k, λ) adicionales mientras el baseline solo ajusta el LR, y no queda claro que la selección se haga fuera del conjunto de test. Parte de la ganancia podría deberse a esta búsqueda extra.
- **Collapse:** con L2 el modelo colapsa (2.22 %) [Tabla 3], y el único freno parece ser $\mathcal{L}_{\text{LLM}}$. No se estudia la geometría global (rango efectivo, isotropía) más allá de $\text{Enc(Text)} - \text{Enc(Code)}$. Que esa diferencia tenga valores singulares pequeños es compatible con un colapso parcial.
- **Varianza alta en algunos casos:** con Llama-3.1-8B la desviación estándar es de ±16.81 [Tabla 15].
- **Ganancias pequeñas fuera del setup canónico:** en Qwen3 (+0.7), OLMo (+0.4), HellaSwag (+1.1) y Rotten Tomatoes (+1.2) son significativas pero modestas.
- **Escala:** modelos de 1–8B y datasets pequeños; nada en preentrenamiento real.
- **Coste no contabilizado:** no se reporta el impacto de la máscara empaquetada en la longitud de secuencia ni en la memoria.

**¿Es coherente la historia X → Y → Z → 1?**
Sí. Los JEPA funcionan en visión → en LLMs faltan vistas y hay que conservar la generación → usar pares Text/Code como vistas más NTP → mejoras consistentes en muchos modelos y datasets. El salto más débil es Y→Z respecto al collapse: se menciona como reto en la introducción y luego no se trata en el método.

**Relevancia para jepa-llm**
Es la referencia central para el caso decoder: da una receta concreta e implementable (máscara por bloques, tokens [PRED], coseno, λ y loss dropout) y deja abiertos exactamente los huecos que el proyecto podría atacar: vistas sin pares naturales, teacher EMA o anti-collapse explícito, targets por token y preentrenamiento real.

**TL;DR propio**
LLM-JEPA añade al fine-tuning de un LLM un término que alinea (por coseno) el embedding de una vista (Text más tokens [PRED]) con el de otra (Code), usando el propio LLM como encoder y predictor. Mejora de forma consistente y estadísticamente significativa en tareas NL→regex/SQL/math con modelos de 1–8B, aunque con un grid de hiperparámetros extra que favorece al método y sin mecanismo anti-collapse explícito. Lectura obligatoria para el proyecto: es el baseline natural de JEPA en decoders.
