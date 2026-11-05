# Learning by Reconstruction Produces Uninformative Features For Perception

## 0. Ficha

| Campo | Valor |
|---|---|
| Autores | Randall Balestriero (independiente), Yann LeCun (NYU) |
| Venue y año | arXiv 2024 (v1, 17 feb 2024, cs.CV). El PDF no indica venue; se publicó en ICML 2024 (dato externo al PDF) |
| Enlace (arXiv / DOI) | https://arxiv.org/abs/2402.11337 |
| Código | No publicado (solo fragmentos de NumPy para la eigendecomposition y el alignment metric en el Apéndice 6.7–6.8) |
| Fecha de lectura | 2026-11-05 |

---

## 1. El núcleo

**X — ¿Qué intentan hacer y por qué es relevante?**
Explicar por qué aprender por reconstrucción en el espacio de entrada (autoencoders, MAE) produce reconstrucciones convincentes pero representaciones poco competitivas para percepción (clasificación), y por qué necesita entrenamientos muy largos y fine-tuning. MAE no satura el linear eval ni tras 1600 epochs en ImageNet, y pasa de ~70 % a 84 % top-1 con fine-tuning. [§1, §2 "Known limitations"]

**Y — ¿Por qué es difícil?**
- No había una medida formal de cuánto se alinean el objetivo de reconstrucción y el de percepción. Las líneas de trabajo previas (nuevas losses perceptuales, interpretar dimensiones del embedding, imponer estructura en el latente) no explican el desajuste de fondo. [§2]
- La reconstrucción da una señal cualitativa fácil de evaluar (las muestras se ven bien), lo que oculta que las features no sirven para percepción. [§1, §2]

**Z — ¿Cuál es su contribución?**
Muestran que la reconstrucción con MSE dedica la capacidad del modelo al **subespacio principal de los datos** (el que explica más varianza de píxeles), mientras que las features útiles para percepción viven en el **subespacio de baja varianza**. Lo formalizan en el caso lineal, donde reconstrucción + supervisado interpola entre PCA y OLS, y derivan una medida cerrada de *alignment*. Formulan tres problemas: **R1 misaligned** (las features que más reconstruyen son las menos informativas), **R2 ill-conditioned** (las features de percepción se aprenden las últimas) y **R3 ill-posed** (modelos con la misma loss de reconstrucción pueden tener rendimientos de percepción muy distintos). Finalmente demuestran que el ruido gaussiano aditivo en un denoising autoencoder no cambia nada, mientras que el masking sí puede mejorar el alineamiento. [§1, §3, §4]

**1a — ¿Cómo lo verifican experimentalmente?**
- Titular: en TinyImagenet, un ResNet9 entrenado sobre imágenes proyectadas en el top subspace que explica el 90 % de la varianza de píxeles alcanza ~45 % de test accuracy; con el bottom subspace (solo el 20 % de la varianza) alcanza ~55 %. [Abstract, Fig. 1]
- Alignment cerrado (Eq. 7) en 7 datasets (ArabicCharacters, ArabicDigits, FashionMNIST, SVHN, CIFAR10, CIFAR100, TinyImagenet). [Fig. 2]
- Curvas de aprendizaje por autovector en autoencoders lineales, MLP y conv/deconv: primero se aprende el top subspace. [Fig. 1 abajo, Fig. 3]
- Pares de ResNet34 autoencoders en Imagenette con y sin una loss supervisada auxiliar: misma loss de reconstrucción, pero ~60 % frente a ~85 % de linear accuracy. [Fig. 5, Fig. 6]
- Efecto de la forma y el ratio del masking sobre el alignment cerrado. [Fig. 7]

**1b — ¿Hay respaldo teórico?**
Sí, en el régimen lineal:
- Teorema 1: solución cerrada del encoder, decoder y clasificador lineales para la loss conjunta (Eq. 3). [§3.1, Apéndice 6.1]
- Corolario 1.1: con $\lambda \to 0$ se recupera OLS y con $\lambda \to \infty$, PCA. [Apéndice 6.2]
- Proposición 1 y Corolario 1.2: condición de alineamiento perfecto y medida continua de alignment. [§3.1, Apéndice 6.3]
- Teorema 2: con un encoder de alta capacidad, el caso no lineal (Eq. 8) es equivalente al lineal en la inicialización y siempre que el decoder sea lineal. [§3.1, Apéndice 6.6]
- Teorema 3 y Corolario 3.1: solución cerrada para denoising autoencoders; el ruido gaussiano aditivo isotrópico no cambia la solución supervisada para ningún $\sigma$. [§4.2, Apéndices 6.4–6.5]

**2 — Contribuciones que reclaman los autores**
- Identificar y formalizar R1, R2 y R3 como causas de que la reconstrucción requiera entrenamientos largos y fine-tuning. [§1]
- Una medida cerrada de alignment entre reconstrucción y percepción, con implementación eficiente. [§3.1, Apéndice 6.8]
- Demostrar que el masking puede ser beneficioso y que el ruido gaussiano aditivo no aporta nada, independientemente de la tarea supervisada. [Abstract, §4.2]
- Una forma de detectar *a priori*, sin conocer la tarea, si una estrategia de ruido nunca tendrá efecto en percepción. [Abstract, §4.2]

---

## 2. Trabajos relacionados (hermanos)

| Trabajo | Diferencia (supuestos / método) | ¿Comparado en experimentos? | Si no, ¿por qué? |
|---|---|---|---|
| MAE (He et al., 2022) | Reconstrucción de píxeles con masking por patches; es el caso que intentan explicar | No como baseline; se citan sus números (70 % → 84 % con fine-tuning, 50 % → 74 % con masking) [§2, §5] | El paper es de análisis, no propone un método competidor |
| Denoising / Variational AE (Vincent et al., 2010; Kingma & Welling, 2013) | Reconstrucción con ruido o con prior latente | Denoising sí, en forma lineal cerrada [§4.2, Fig. 7] | Los VAE no se analizan; no se justifica |
| Autoregresivos (PixelCNN, PixelSNAIL) | Reconstrucción secuencial de píxeles | No | No se justifica |
| SSL joint-embedding (SimCLR, Barlow Twins, VICReg, cookbook) | Invarianza + anti-collapse en el espacio latente, sin reconstruir la entrada | No | Solo se mencionan como el paradigma que domina el state of the art [§1] |
| Score matching / NCE | Tarea sustituta de clasificar muestras frente a ruido | No | Solo como contexto [§1] |
| Spectral bias de DNs (Rahaman et al., 2019; Chakrabarty & Maji, 2019; Schwarz et al., 2021) | Las redes aprenden antes las bajas frecuencias | No | Se presentan como hallazgo consistente con R2 [§3.2] |
| Losses perceptuales / métricas (SSIM, metric learning, Ballé et al.) | Cambiar la distancia de reconstrucción | No | Se citan como línea de trabajo que no resuelve el problema de fondo [§2] |

---

## 3. Antecedentes (ancestros)

**Conceptos previos necesarios**
- PCA como solución óptima del autoencoder lineal y OLS como solución del problema supervisado lineal. [§2, §3.1]
- Eigendecomposition de la matriz de covarianza y problema de autovalores generalizado. [Apéndice 6.1]
- Velocidad de convergencia en función del eigengap (power iteration) y el decaimiento exponencial del espectro de las imágenes naturales. [§3.2]
- Denoising autoencoders y momentos analíticos del ruido de data augmentation (Balestriero et al., 2022). [§4.2, Apéndice 6.4]

**Definición del problema**
Datos $X \in \mathbb{R}^{D\times N}$ (columnas = muestras), etiquetas $Y \in \mathbb{R}^{C\times N}$. Encoder $V \in \mathbb{R}^{K\times D}$, decoder $Z$ y clasificador $W$ lineales, con loss conjunta
$$\mathcal{L}(V,W,Z) = \|W^\top V^\top X - Y\|_F^2 + \lambda\,\|Z^\top V^\top X - X\|_F^2 .$$
El alignment para dimensión latente $k$ es
$$\text{alignment}(k) = \frac{\|Y^\top Y (P_{XX^\top})_{1:k}\|_F^2}{\|Y^\top Y P_{XX^\top}\|_F^2} \in [0,1],$$
es decir, el error supervisado mínimo alcanzable con la representación que minimiza la reconstrucción. [§3.1, Eq. 3, Eq. 7]

**Supuestos inusuales o restrictivos**
- Todo el análisis formal es lineal, con loss MSE en el espacio de píxeles y clasificador por regresión de mínimos cuadrados (no cross-entropy). [§3.1]
- $X$ de rango completo. [§2]
- El Teorema 2 supone un encoder tan potente que puede producir cualquier representación (versión no paramétrica). [Apéndice 6.6]
- El denoising se analiza en expectativa sobre el ruido, con $\mathbb{E}[X'X'^\top]$ de rango completo. [Apéndice 6.4]

**¿El escenario del problema es una contribución nueva?**
Parcialmente. La loss conjunta lineal y sus extremos (PCA, OLS) son clásicos; la medida de alignment y su uso para comparar distribuciones de ruido son la parte nueva. [§3.1, §4.2]

---

## 4. Método

**Qué hacen**
1. Resuelven en forma cerrada la loss conjunta lineal (Teorema 1) y muestran que interpola entre OLS y PCA (Corolario 1.1). [§3.1]
2. Definen la condición de alineamiento perfecto: el top-K de $X^\top X$ y el de $Y^\top Y$ deben intersecar en dimensión $K$ (Proposición 1). En la práctica nunca se cumple. [§3.1]
3. Proponen la medida continua de alignment (Eq. 7) y la calculan para un barrido de $K$ en 7 datasets. [§3.1, Fig. 2]
4. Extienden el argumento a encoders no lineales (Eq. 8, Teorema 2). [§3.1]
5. Proyectan los datos (train y test) sobre el top o el bottom subspace de PCA antes de entrenar un clasificador ResNet9, para comprobar dónde están las features de percepción. [§3.2, Fig. 1, Fig. 4]
6. Siguen la evolución del error de reconstrucción proyectado en cada autovector durante el entrenamiento de varios autoencoders. [§3.2, Fig. 1, Fig. 3]
7. Entrenan pares de ResNet34 autoencoders con y sin loss supervisada auxiliar sobre el mismo embedding, para mostrar R3. [§4.1, Fig. 5–6]
8. Derivan la solución cerrada del denoising autoencoder lineal (Teorema 3), prueban que el ruido gaussiano aditivo no cambia la solución supervisada (Corolario 3.1) y miden el efecto del masking en función del tamaño de patch y del ratio. [§4.2, Fig. 7]

**Por qué lo hacen así**
- **Régimen lineal:** da soluciones cerradas y la medida de alignment; el Teorema 2 y la validación con Adam (Fig. 8) se usan para defender que es informativo. [§3.1, Apéndice 6.9]
- **Proyección PCA de la entrada:** es un experimento controlado que separa "lo que más reconstruye" de "lo que menos reconstruye" sin depender del autoencoder. [§3.2]
- **R2 vía eigengap:** las imágenes naturales tienen un espectro con decaimiento exponencial, así que el top subspace se aprende exponencialmente más rápido. [§3.2]
- **Loss supervisada auxiliar:** sirve para obtener un "oráculo" con la misma reconstrucción pero mejores features. [§4.1]
- **Denoising en forma cerrada:** permite comparar distribuciones de ruido sin entrenar redes y, para el ruido gaussiano, sin conocer la tarea. [§4.2]

---

## 5. Configuración experimental

**Instancia del problema**
- Alignment cerrado: ArabicCharacters, ArabicDigits, FashionMNIST, SVHN, CIFAR10, CIFAR100, TinyImagenet. [Fig. 2]
- Proyección top/bottom: TinyImagenet, ArabicCharacters, ArabicDigits, SVHN, CIFAR10, CIFAR100, con y sin data augmentation; métrica top-1 test accuracy. [Fig. 1, Fig. 4]
- Dinámica de aprendizaje: TinyImagenet (Fig. 1) y CIFAR (Fig. 3). [§3.2]
- R3: Imagenette (Imagenet-10); linear y nonlinear probe. [Fig. 5–6]
- Masking: ArabicCharacters, ArabicDigits, FashionMNIST, SVHN, CIFAR100, con patches de 1×1, 2×2 y 4×4 y probabilidad de drop de 0 a 99 %. [Fig. 7]

**Detalles de implementación**
- Clasificador: ResNet9. [Fig. 1, Fig. 4]
- Autoencoders: conv/deconv y MLP de profundidad 3 y 5 (Fig. 3); ResNet34 con varias dimensiones de embedding, aprox. 500–4000 (Fig. 5). [§3.2, §4.1]
- Validación del Teorema 1 con Adam: N=256, D∈{16, 512}, K∈{3, 16}, λ∈{0, 0.1, 1, 10}. [Fig. 8]
- Optimizador, learning rates, epochs, batch size, augmentations concretas y cómputo de los experimentos con DNs: No se aborda en el paper.

---

## 6. Resultados y discusión

**Resultados principales frente a baselines**

| Tarea / dataset | Métrica | Método | Mejor baseline |
|---|---|---|---|
| TinyImagenet, bottom subspace (~20 % de la varianza) vs. top subspace (~90 %) | top-1 test ↑ | ~55 % (bottom) | ~45 % (top); 58.7 % con la imagen completa [Abstract, Fig. 1] |
| Imagenette, ResNet34 AE con loss supervisada auxiliar vs. solo reconstrucción | linear acc ↑ | ~85 % | ~60 % (misma loss de reconstrucción en train y test) [Fig. 6] |
| Imagenette, varios K | gap de linear acc | 42–54 puntos | — [Fig. 5] |
| Imagenette, varios K | gap de nonlinear acc | 26–34 puntos | — [Fig. 5] |

No hay baselines en sentido clásico: es un paper de análisis. Las comparaciones son entre condiciones (top frente a bottom, con o sin señal supervisada, tipo de ruido).

**Rigor estadístico**
No se reportan desviaciones estándar, intervalos de confianza ni número de semillas en ninguna figura. No se aborda en el paper.

**¿Comparación justa?**
En Fig. 5–6 los pares de modelos comparten todo salvo la loss supervisada, según el texto. Los hiperparámetros no se describen, así que no se puede verificar. [§4.1, Fig. 5]

**Ablaciones**
- Excepto en datasets sin fondo y bien alineados (ArabicCharacters, ArabicDigits), el bottom subspace da mejor accuracy que el top para cualquier porcentaje de filtrado, con o sin data augmentation. [Fig. 4]
- El alignment baja con fondos complejos, con más clases (CIFAR10 → CIFAR100) y con mayor resolución; en TinyImagenet crece solo linealmente con K. [Fig. 2]
- Masking: en datasets ya bien alineados (ArabicDigits) cualquier masking salvo el de forma 1×1 es perjudicial; en CIFAR100 ayuda, sobre todo con K pequeño. El beneficio depende de la forma del patch, del ratio y del dataset, y se reduce al crecer K. [Fig. 7]
- Ruido gaussiano aditivo: sin efecto para cualquier $\sigma$ (resultado teórico). [Corolario 3.1]

**Limitaciones que reconocen los autores**
- Elegir la distribución de ruido sin conocer la tarea es difícil; su criterio solo dice si un ruido tiene impacto, no si es positivo. [§4.2]
- El masking requiere ajuste por dataset. [§5]

---

## 7. Conclusión y trabajo futuro (descendientes)

**Trabajo futuro propuesto**
- Usar la medida cerrada para seleccionar *a priori* distribuciones de ruido para denoising. [§5]
- Explorar nuevas distribuciones de ruido como principal palanca del aprendizaje por reconstrucción. [§1]
- Estudiar los métodos de reconstrucción en otras modalidades, como series temporales y NLP. [§5]

**Preguntas abiertas**
- Cómo diseñar el ruido o el masking sin etiquetas para que sea beneficioso. [§4.2]
- No se aborda en el paper si el mismo desajuste afecta a la predicción en espacio latente (JEPA) ni cómo se compara con ella.

---

## 8. Lectura crítica

> Esta sección es la valoración del lector, no lo que dice el paper.

**¿Contribuciones claramente separadas del trabajo previo?**
En general sí: PCA/OLS como extremos, el spectral bias y la convergencia por eigengap se atribuyen a trabajo previo. La parte nueva (la medida de alignment, la proyección top/bottom como experimento controlado y el análisis cerrado del denoising) está bien delimitada. Sin embargo, la idea de que el ruido gaussiano aditivo equivale a una regularización tipo ridge que no cambia los autovectores es conocida; el propio Apéndice 6.4 la llama "trivial and commonly done", lo que choca con presentarla como contribución en el abstract.

**Afirmaciones sin respaldo o lenguaje grandilocuente**
- "We prove that ... masking [is] indeed beneficial" [Abstract]: no hay ningún teorema sobre masking. El Teorema 3 da una forma cerrada general, y la Fig. 7 muestra que el masking *empeora* el alignment en varios datasets (ArabicCharacters, ArabicDigits, FashionMNIST con patches 2×2 y 4×4).
- Teorema 2: el enunciado dice "equivalent at initialization for any decoder", pero la prueba (Apéndice 6.6) solo resuelve el caso no paramétrico con decoder lineal; la parte de la inicialización no se demuestra.
- Las cifras de MAE (70 → 84 % y 50 → 74 %) aparecen sin cita ni tabla. [§2, §5]
- Errata: el Corolario 1.1 dice "the solution from Theorem 2" cuando se refiere al Teorema 1. [§3.1]

**Limitaciones no mencionadas por los autores**
- Sin semillas ni varianza en ninguna figura, y sin hiperparámetros de los experimentos con DNs: R3 se apoya esencialmente en un par de entrenamientos (Fig. 6).
- El experimento de proyección (Fig. 1, 4) también entrena el clasificador sobre los datos proyectados, así que mide qué subespacio contiene información discriminativa, no qué aprende un autoencoder. El enlace con la dinámica del autoencoder viene de Fig. 1 abajo y Fig. 3, que son cualitativas.
- El análisis depende de la MSE en píxeles y de la estructura espectral de las imágenes naturales. Para texto, donde la reconstrucción es una cross-entropy sobre tokens discretos y no hay "varianza de píxeles", la conclusión no se traslada directamente. Es lo relevante para este proyecto: el paper apoya la intuición de JEPA (no gastar capacidad en detalles de alta varianza de la entrada), pero no la demuestra para lenguaje.
- No se compara con ningún método de predicción latente ni joint-embedding, aunque es la alternativa implícita.
- Ni MAE ni ningún ViT se entrenan; las conclusiones sobre MAE son por analogía.

**¿Es coherente la historia X → Y → Z → 1?**
Sí para R1 y R2: la teoría lineal, la Fig. 2 y las Fig. 1/4 encajan bien. R3 es más débil (un experimento sin réplicas) y la parte de denoising promete más de lo que demuestra: el resultado sólido es negativo (ruido gaussiano inútil), mientras que el positivo sobre masking es empírico y dependiente del dataset.

**TL;DR propio**
Con MSE en píxeles, la reconstrucción prioriza el subespacio de alta varianza, que es el menos útil para clasificar; las features discriminativas se aprenden tarde y la loss de reconstrucción no distingue un buen modelo de uno malo. La teoría lineal es limpia y el experimento de proyección top/bottom es convincente, pero la parte de denoising está sobrevendida y no hay semillas ni hiperparámetros. Merece la pena leer §3 (y la Fig. 1) como argumento canónico de por qué JEPA predice en latente en vez de reconstruir, teniendo en cuenta que no se traslada directamente a la cross-entropy sobre tokens.
