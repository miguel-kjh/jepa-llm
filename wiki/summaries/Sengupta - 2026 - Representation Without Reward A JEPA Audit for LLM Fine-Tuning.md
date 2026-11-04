# Representation Without Reward: A JEPA Audit for LLM Fine-Tuning

## 0. Ficha

| Campo | Valor |
|---|---|
| Autores | Biswa Sengupta (LLM Suite Team, JPMorgan Chase & Co.) |
| Venue y año | Preprint, arXiv 2026 (v1, 14 may 2026) |
| Enlace (arXiv / DOI) | https://arxiv.org/abs/2605.15394 |
| Código | No publicado (el PDF no da URL) |
| Fecha de lectura | 2026-11-04 |

> Nota: el PDF se leyó como texto extraído (pypdf); las figuras (data-efficiency, diagramas) se interpretaron por su texto y por la Tabla V. Los apéndices B y C (variantes Fisher y DV-JEPA) se leyeron solo en sus partes clave (resultados y definición).

---

## 1. El núcleo

**X — ¿Qué intentan hacer y por qué es relevante?**
Comprobar si los objetivos auxiliares de estilo JEPA, añadidos al fine-tuning de un LLM, no solo cambian la geometría de los hidden states sino que mejoran la métrica de la tarea decodificada (exact match). El autor separa dos hipótesis: **H1**, el auxiliar *actúa* y cambia la geometría; **H2**, ese cambio se traduce en mejor exact match. [§I]

**Y — ¿Por qué es difícil?**
- En un LM autoregresivo la representación solo llega a la salida a través de la LM head y de un argmax. Una representación puede volverse más "suave", isotrópica o predictiva sin que cambie la secuencia elegida. [§I]
- Trabajos previos como STP (Semantic Tube Prediction, 16× de eficiencia en datos) y LLM-JEPA reportan ganancias sin tests corregidos por comparaciones múltiples. [§I, §II]

**Z — ¿Cuál es su contribución?**
Una auditoría controlada: modelo, LoRA, datos, punto de inserción, decodificador y evaluador fijos, y solo cambia el auxiliar.
- **22 auxiliares de entrenamiento y 1 de inferencia,** agrupados en: forma de trayectoria (STP, T1–T6), contrastivos y predictivos (T7, CPC), distribucionales (SIGReg-State/Tangent, VICReg, sliced-Wasserstein, score matching), asimetría predictor/target (BYOL-LLM, I-JEPA-LLM), residuos de Jacobi con métrica Fisher y un "decoder-visible JEPA" (DV-JEPA) cuyo gradiente cae en el cono de la cross-entropy. [§III, Tablas I–II, Apéndices B–C]
- **Estadística:** tests de Welch pareados y no pareados con corrección Bonferroni y Holm-Bonferroni. [§IV-A]

**1a — ¿Cómo lo verifican experimentalmente?**
- Llama-3.2-1B-Instruct con LoRA r = 32 en NL-RX-TURK (benchmark principal, no saturado) y NL-RX-SYNTH, con n = 3 semillas por celda. [§IV]
- **Resultado titular, un "null estructurado":** el mejor auxiliar es T3-Local (+2.53 pp, p pareado 0.003), pero ninguno sobrevive a la corrección por comparaciones múltiples. [Tabla III]
- **BYOL-LLM:** +0.60 pp en TURK y 0.00 en SYNTH. [§V-C]
- **DV-JEPA con fine-tuning completo** (n = 5): TURK +0.04 pp (p = 0.96) y SYNTH +0.52 pp (p = 0.28). [§VI-E]
- **Diagnósticos:** curvatura, anisotropía y coseno entre el gradiente del auxiliar y el de la CE. [§VI-A, Apéndice A]

**1b — ¿Hay respaldo teórico?**
Limitado. Hay análisis de escalado de las losses de curvatura ($\mathcal{L}_{T1}/\mathcal{L}_{STP} = O(\Delta^2)$), construcciones con garantías (métrica SPD por construcción, CF distance → 0 si y solo si la distribución es normal) y el argumento de que las losses distribucionales caen fuera del sesgo implícito de la CE. No hay teoría sobre el acoplamiento entre representación y métrica. [§III]

**2 — Contribuciones que reclaman los autores**
- Un test "decoder-visible" de JEPA en fine-tuning: ¿el cambio de geometría sobrevive a la LM head? [§I-a]
- Un mapa de hipótesis controlado con 22 auxiliares más un proyector de inferencia. [§I-a]
- Un null estructurado para el exact match asintótico. [§I-a]
- Diagnósticos que separan la inactividad en rendimiento de la inactividad representacional. [§I-a]

---

## 2. Trabajos relacionados (hermanos)

| Trabajo | Diferencia (supuestos / método) | ¿Comparado en experimentos? | Si no, ¿por qué? |
|---|---|---|---|
| STP (Semantic Tube Prediction) | Alineación coseno de velocidades de la trayectoria de hidden states en un solo forward; reporta 16× de eficiencia en datos en SYNTH | Sí, replicado como celda STP y extendido (T1–T6) [Tablas III–IV] | — |
| LLM-JEPA | Dos vistas (Text/Code), predictor y fine-tuning completo | Parcialmente: BYOL-LLM e I-JEPA-LLM prueban la asimetría en una sola pasada, no la receta de dos vistas [§V-C] | Lo marca como trabajo futuro ("test full multi-view JEPA") [§VI-E] |
| I-JEPA / V-JEPA | JEPA de visión con target EMA | Como inspiración de L14 (I-JEPA-LLM con target aleatorio congelado) | Aproximación de una sola pasada |
| BYOL, SimSiam, VICReg, Barlow Twins, SimCLR | SSL no contrastivo y contrastivo | Sí, adaptados (L13, L5, T7) | — |
| SIGReg / LeJEPA | Regularizador de isotropía (Cramér-Wold, Epps-Pulley) | Sí (L1, L2) | — |
| data2vec | Predicción latente en texto | No | Solo se cita |
| Wang et al. (straightening en world models) | El mismo regularizador de curvatura mejora la planificación por gradiente | No (otra tarea) | Lo usan para apoyar su interpretación [§VI-D] |

---

## 3. Antecedentes (ancestros)

**Conceptos previos necesarios**
- La idea JEPA (predecir representaciones) y la asimetría predictor + EMA + stop-grad. [§II]
- Geometría de trayectorias: curvatura geodésica, campos de Jacobi, métrica Fisher de la LM head. [§III, Apéndice B]
- Tests de Welch y corrección por comparaciones múltiples. [§IV-A]

**Definición del problema**
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{LM}} + \lambda(t)\,\mathcal{L}_{\text{aux}}, \qquad \mathcal{L}_{\text{aux}} \in \{\mathcal{L}_{\text{traj}}, \mathcal{L}_{\text{dist}}, \mathcal{L}_{\text{pred}}\}$$
El auxiliar actúa sobre los hidden states de la última capa del span del asistente (con 2 tokens finales recortados), en paralelo a la LM head y con un único forward. [Eqs. 1–5, Fig. 1]

El criterio "representationally active but task-metric inert" se cumple cuando $\Delta G > 0$, $|\rho_{\text{aux,CE}}| \ll 1$ y $|\Delta EM| \le \epsilon$. [Eq. 31]

**Supuestos inusuales o restrictivos**
- Solo un forward: no hay segunda vista ni encoder target real. Se excluye a propósito la receta JEPA completa. [§II-c]
- LoRA con el modelo base congelado, batch 4 y spans de regex cortos (mediana de 12 tokens). [§IV]

**¿El escenario del problema es una contribución nueva?**
Sí: el planteamiento H1 frente a H2 ("¿la geometría llega al decoder?") es el marco que propone el paper.

---

## 4. Método

**Qué hacen**
1. Fijan el harness: Llama-3.2-1B-Instruct, LoRA r = 32 / α = 64 en todas las proyecciones, AdamW con LR 2e-4, 4 épocas, batch 4, greedy y exact match sobre 500 ejemplos de test. [§IV]
2. Ponderan el auxiliar con un schedule de λ: warmup del 25 %, meseta y caída al 10 % en el último 25 %. [Eq. 30]
3. Aplican un "EOS clip": excluyen del auxiliar los dos últimos hidden states. Sin él, todos los auxiliares daban un 0 % de exact match por suprimir el EOS. [§IV-e, Fig. 2]
4. Evalúan cada auxiliar con n = 3 semillas ("escalation protocol": 1 semilla para decidir si se expande a 3). [§IV-A]
5. Rutas de falsificación pre-registradas: variantes con métrica Fisher (Apéndice B) y DV-JEPA, una KL multi-horizonte sobre distribuciones de la LM head más un hinge de margen (Apéndice C), con el umbral pre-registrado $\rho_{\text{aux,CE}} > 0.20$. [§VI-D]
6. Curva de eficiencia en datos en TURK: 5 fracciones × 4 variantes × 3 semillas. [§V-F]

**Por qué lo hacen así**
- **Fijar el harness y variar solo el auxiliar:** permite atribuir los efectos a la loss. [§I]
- **Tests pareados y corregidos:** las convenciones habituales (media ± std) habrían declarado ganadoras a celdas que luego desaparecen. L5 dio +2.73 pp con una semilla y +0.40 con tres. [§IV-A, §V-C]
- **Losses distribucionales:** caen fuera de la dirección de sesgo implícito de la CE, para descartar que los auxiliares geométricos fallen solo por estar alineados con la CE. [§III-K]

---

## 5. Configuración experimental

**Instancia del problema**
- NL-RX-TURK y NL-RX-SYNTH (8,000 de train y 2,000 de test cada uno; se evalúan los primeros 500), formato de chat ("Convert natural language to regular expression"). [§IV-a]
- Métricas: exact match y prefix accuracy. [§IV-f]

**Detalles de implementación**
- bf16 con gradient checkpointing; ≈50–60 min por celda en una workstation; el barrido completo cabe en ≈1 día-máquina. [§IV-b, §IV-c]
- Pesos λ₀: 1.0 para STP, T1, T2, T5 y T7; {3e-4, 1e-3} para T3. [§IV-d]
- Fine-tuning completo para DV-JEPA: LR 2e-5, n = 5. [§VI-E]

---

## 6. Resultados y discusión

**Resultados principales frente a baselines** (el baseline es LoRA sin auxiliar)

| Tarea / dataset | Métrica | Método | Baseline |
|---|---|---|---|
| TURK, T3-Local (mejor media) | EM % | 53.20 ± 1.91 (p pareado 0.003; no pareado 0.16) | 50.67 ± 1.68 [Tabla III] |
| TURK, T6 / T3 (λ = 3e-4) | EM % | 52.80 / 52.93 | 50.67 [Tabla III] |
| TURK, BYOL-LLM (L13) | EM % | 51.27 ± 0.70 (p 0.64) | 50.67 [Tabla III] |
| TURK, I-JEPA-LLM (L14) | EM % | 49.53 ± 1.29 | 50.67 [Tabla III] |
| TURK, SIGReg-State (L1) | EM % | 51.40 ± 0.53 | 50.67 [Tabla III] |
| TURK, T1 (λ = 1) | EM % | 2.27 (collapse) | 50.67 [Tabla III] |
| TURK, T9 (decoding proyectado) | EM % | 0.00 (bucles de tokens) | 50.67 [Tabla III] |
| SYNTH, L1 / L4 | EM % | 89.80 / 89.73 (p pareado 0.10 / 0.06) | 88.93 ± 0.42 [Tabla IV] |
| SYNTH, L9 score matching | EM % | 86.53 (−2.40; p pareado 0.06) | 88.93 [Tabla IV] |
| TURK, data-efficiency al 5 %, T6 | EM % | 31.33 (p 0.013) | 30.33 [Tabla V] |
| TURK, data-efficiency al 50 %, T3-Local | EM % | 48.40 (p 0.029) | 46.73 [Tabla V] |
| TURK, DV-JEPA con fine-tuning completo (n = 5) | EM % | 48.76 ± 1.42 | 48.72 ± 0.61 [§VI-E] |

**Rigor estadístico**
- Ejemplar para el área: n = 3 (n = 5 en el fine-tuning completo), Welch pareado y no pareado, Bonferroni y Holm por familias, y protocolo de escalado.
- El propio autor reconoce que n = 3 da poca potencia para efectos ≤ 2 pp. [§IV-A, §VI-F]

**¿Comparación justa?**
- Sí dentro del harness: todo fijo salvo el auxiliar.
- Pero la mayoría de auxiliares se prueban con un único λ (solo T3 tiene un barrido de 2 puntos). [§VI-F]

**Ablaciones**
- **Familia Jacobi:** profundidad (T5, +1.53) < una capa (T3, +1.86) < multiescala (T6, +2.13); T3-Local es el mejor (+2.53). [§V-B]
- **Métrica Fisher:** pierde en todos los enfrentamientos contra su versión euclídea (−1.20 a −2.47 pp) y anula la firma geométrica. [§VI-D, Apéndice B]
- **DV-JEPA:** primer coseno positivo entre gradientes ($\rho = +0.186 \pm 0.050$, frente a $[-0.69, +0.01]$ del resto), pero exact match +0.93 pp con p = 0.48. [§VII]
- **Diagnósticos:** la familia JFR reduce la curvatura de ≈2.00 a 1.21–1.45 rad; L1, L6 y L14 colapsan la anisotropía por debajo del baseline; la mayoría reduce la varianza entre semillas. [§VI-A, §VII]

**Limitaciones que reconocen los autores**
- n = 3 es poco para efectos ≤ 2 pp.
- Solo un modelo (1B), LoRA y la familia NL-RX.
- La mayoría de auxiliares, con un único λ.
- Batch 4, lo que invalida conclusiones sobre métodos contrastivos. [§VI-F]
- No se probó la receta JEPA multi-vista completa ni STP con fine-tuning completo. [§II-c, §VI-E]

---

## 7. Conclusión y trabajo futuro (descendientes)

**Trabajo futuro propuesto**
- Ampliar las celdas candidatas a n ≥ 5–10 semillas y completar la curva de eficiencia en SYNTH. [§VI-E]
- Métricas que puedan revelar ganancias representacionales: equivalencia semántica de regex, calibración y OOD. [§VI-E]
- Contrastivos con batches grandes o memory banks. [§VI-E]
- **Probar un JEPA multi-vista completo,** con encoder target y construcción de vistas no trivial. [§VI-E]

**Preguntas abiertas**
- Bajo qué métricas la geometría útil de los hidden states se convierte en señal visible para el decoder; el autor reformula la evaluación como un "coupling problem". [Abstract, §VI-D]

---

## 8. Lectura crítica

> Esta sección es la valoración del lector, no lo que dice el paper.

**¿Contribuciones claramente separadas del trabajo previo?**
Sí. Es un paper de auditoría honesto: distingue con claridad qué replica (STP), qué adapta (BYOL, SIGReg, VICReg) y qué propone (las variantes Jacobi, Fisher y DV-JEPA), y declara explícitamente que *no* prueba la receta de LLM-JEPA. [§II-c]

**Afirmaciones sin respaldo o lenguaje grandilocuente**
- **"Hidden-state representation work and decoded-task accuracy are weakly coupled":** la conclusión es razonable, pero se generaliza desde un único modelo de 1B, una sola familia de tareas (regex, de salida muy corta y simbólica) y una métrica binaria muy dura (exact match). La propia lista de trabajo futuro admite que otras métricas podrían revelar valor.
- **Números inconsistentes en el conteo:** "twenty-two training-time auxiliaries" en el abstract frente a "eighteen training-time auxiliaries" en la Tabla II (más las 4 rutas de los apéndices).
- **Interpretación del data-efficiency curve:** "first concrete instance of representational work translating into exact-match accuracy" para celdas que el propio autor dice que no sobreviven a la corrección. Es coherente con su marco, pero algo ambiguo.

**Limitaciones no mencionadas por los autores**
- **Baselines muy distintos a los de LLM-JEPA:** el baseline LoRA en SYNTH es de 88.93 %, mientras que LLM-JEPA reporta 57.29 % (fine-tuning completo) y 50.18 % (LoRA rango 512) con el mismo modelo y dataset. Esa diferencia enorme sugiere que la ganancia de LLM-JEPA podría depender de un baseline subentrenado o de un protocolo distinto. Es un hallazgo implícito muy relevante que el paper no explota.
- **Las vistas de L13/L14 no son las de LLM-JEPA:** las mitades del span de la regex y bloques enmascarados de la propia salida no equivalen a vistas Text↔Code. El null de "asimetría JEPA" no aplica directamente a LLM-JEPA.
- **Targets de L14 aleatorios:** un target congelado con inicialización aleatoria es una aproximación débil de I-JEPA.
- **Margen de detección:** con 500 ejemplos de test y n = 3, el umbral práctico es ≈2.5 pp; efectos menores no se pueden resolver.

**¿Es coherente la historia X → Y → Z → 1?**
Sí, y es la más rigurosa de la colección. Formula las hipótesis, diseña rutas de falsificación pre-registradas y reporta un null con todos sus matices (geometría activa, métrica inerte).

**Relevancia para jepa-llm**
Muy alta como guía metodológica y como señal de alerta:
1. Cualquier ganancia JEPA en decoders debe reportarse con varias semillas, tests pareados y corrección por comparaciones múltiples.
2. Conviene medir a la vez la geometría (curvatura, anisotropía, RankMe) y la métrica de la tarea, además del coseno entre el gradiente del auxiliar y el de la CE.
3. Hay que revisar si el baseline de LLM-JEPA está bien entrenado antes de reproducir sus ganancias.
4. El hueco que deja abierto, un JEPA multi-vista completo con target encoder, es exactamente un experimento candidato para el proyecto.

**TL;DR propio**
El autor prueba más de 20 auxiliares de estilo JEPA (geométricos, distribucionales, BYOL/I-JEPA de una sola pasada, Fisher y DV-JEPA) en el fine-tuning LoRA de Llama-3.2-1B para NL→regex. Todos cambian la geometría de los hidden states, pero ninguno mejora el exact match tras corregir por comparaciones múltiples, y el null se mantiene con fine-tuning completo. Lectura obligatoria para el proyecto: es el contrapunto riguroso a LLM-JEPA y un modelo de cómo evaluar.
