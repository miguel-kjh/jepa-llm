# 2026-11-05 — LLM-JEPA en NL-RX-Synth con Llama-3.2-1B-Instruct

Primera prueba del método de LLM-JEPA (Huang et al., 2025; [resumen](../summaries/Huang%20et%20al.%20-%202025%20-%20LLM-JEPA%20Large%20Language%20Models%20Meet%20Joint%20Embedding%20Predictive%20Architectures.md)) frente a un fine-tuning normal. Notebook: `notebooks/llm_jepa_nlrx.ipynb`. Artefactos: `outputs/llm_jepa_nlrx/` (no versionado).

## Conclusión

- LLM-JEPA **no mejora de forma significativa** al baseline: 87.40 % frente a 86.80 % de exact match en test (+0.6 puntos, McNemar p = 0.26).
- **El término JEPA colapsa**: todos los embeddings del último token apuntan en la misma dirección (coseno de 0.9999 entre ejemplos distintos), así que la distancia coseno llega a ~0 sin alinear nada útil entre Text y Code.
- **Nuestro baseline es mucho más fuerte que el del paper**: 86.80 % frente a 57.29 %, en línea con el 88.9 % de Sengupta con LoRA r = 32. Refuerza la sospecha de que el baseline de LLM-JEPA está poco entrenado (ver `wiki/notes.md`).

## Configuración

| Parámetro | Valor |
|---|---|
| Datos | `datasets/raw/NL-RX-Synth/` (`src.txt` → `targ.txt`), 10 000 pares |
| Split | Fijo 80/10/10 (8000/1000/1000), aleatorio por ejemplo. 16 pares de test aparecen idénticos en train |
| Modelo | `meta-llama/Llama-3.2-1B-Instruct`, bf16 |
| Fine-tuning | LoRA r = 32, α = 64, dropout 0.05, en q/k/v/o/gate/up/down |
| Optimización | AdamW, LR 2e-5, batch 8, 4 épocas (4000 pasos), warmup lineal 3 %, clip 1.0 |
| LLM-JEPA | k = 1 token `[PRED]` (`<|reserved_special_token_0|>`), λ = 1, coseno, sin stop-gradient, sin EMA |
| Vistas | Text = descripción como mensaje de usuario (chat template); Code = regex como mensaje de assistant |
| Enc / Pred | Último hidden state del último token (`<|eot_id|>` en Code, `[PRED]` en Text), mismo LLM |
| Forwards | NTP sobre prompt + regex; JEPA empaquetado `[Text + PRED \| Code]` con máscara 4D causal por bloques y position ids que vuelven a 0 |
| Evaluación | Greedy, exact match de la regex en test |
| Ejecuciones | Una por método, misma inicialización LoRA y mismo orden de batches |
| Hardware | RTX 4070 SUPER (12 GB), WSL2 |

Diferencias con el paper: LoRA en vez de fine-tuning completo, split propio, (k, λ) fijados de antemano en vez de elegir el mejor de un grid de 20, y una sola ejecución en vez de 5 semillas.

## Resultados

| Ejecución | Exact match (test) | Val NTP final | Val distancia JEPA final | Tiempo de entrenamiento |
|---|---|---|---|---|
| Baseline (λ = 0) | **86.80 %** (868/1000) | 0.0264 | 0.470 | 994 s |
| LLM-JEPA (λ = 1, k = 1) | **87.40 %** (874/1000) | 0.0274 | 0.0001 | 1519 s (+53 %) |

- **Test pareado:** 20 ejemplos discrepan; 13 los acierta solo LLM-JEPA y 7 solo el baseline. McNemar exacto: p = 0.263.
- **Curvas de NTP:** prácticamente iguales en train y validation durante todo el entrenamiento.
- **Distancia JEPA:** en el baseline se queda plana en ~0.47 (la NTP sola no alinea las vistas, igual que en la Fig. 3 del paper). En LLM-JEPA cae a 0.004 en 250 pasos y a 0.0001 al final.

## Diagnóstico de collapse

Sobre los 300 primeros ejemplos de validation, con el último hidden state de cada vista (Pred = último `[PRED]` de Text, Code = último token de Code). Reproducible con `uv run jepa-llm nlrx-diagnose --run all`, que guarda `outputs/llm_jepa_nlrx/<tag>/collapse.json` (issue #2):

| Modelo | cos(Pred, Code) del mismo par | cos medio entre Codes distintos | Effective rank Code / Text (centrado) |
|---|---|---|---|
| Baseline | 0.53 | 0.93 | 166.1 / 143.3 |
| LLM-JEPA | 0.9999 | **0.9999** | 214.5 / 218.7 |

Los cosenos coinciden con los del script original (que no se conservó). El effective rank (Roy & Vetterli, embeddings centrados) difiere: 165.5 / 142.4 y 204.4 / 212.3 en el script original. No se sabe qué definición ni qué ejemplos usó; en LLM-JEPA el rank se calcula sobre residuos minúsculos alrededor de la dirección común, donde el ruido numérico de bf16 pesa mucho, así que ese número es frágil. La conclusión no cambia.

Métricas centradas (se resta la media de cada vista antes de medir; añadidas en el issue #2):

| Modelo | ‖media‖ / ‖x‖ (Code) | cos mismo par, centrado | Top-1 Text→Code, centrado (azar 0.003) |
|---|---|---|---|
| Baseline | 0.963 | 0.021 | 0.010 |
| LLM-JEPA | 0.9999 | 0.078 | 0.017 |

- **El regularizador lleva todo a un punto en vez de reorganizar el espacio.** En LLM-JEPA el embedding es casi solo el vector medio y el residuo propio de cada ejemplo es de alrededor del 1 % de la norma. Sin centrar, el coseno del mismo par (0.99986) es incluso algo menor que el coseno entre ejemplos distintos (0.99988).
- **Apenas queda señal de pares en el residuo:** dado un Text, el Code correcto se recupera en el 1.7 % de los casos (unas 5 veces el azar).
- **Es un collapse en dirección, no dimensional:** el effective rank del residuo sube (166 → 214), así que la variación entre ejemplos no se concentra en menos dimensiones. El baseline ya parte de un cono estrecho (0.963 de la norma es dirección común, la anisotropía típica de los LLMs).
- **Mecanismo probable (no verificado con un experimento):** las dos posiciones comparadas (último `[PRED]` y `<|eot_id|>` final de Code) tienen hidden states que la NTP nunca restringe. El `[PRED]` no aparece en el forward NTP y el `<|eot_id|>` final no tiene token siguiente con label. Llevarlas a un vector constante reduce la loss JEPA a cero sin coste para la NTP, lo que encaja con un collapse en menos de 250 pasos y un exact match sin cambios. Por eso el pooling sobre tokens de contenido (paso 3) parece más prometedor que un término de varianza.

- Si el coseno entre Codes **distintos** es igual que el coseno entre Pred y Code **del mismo par**, el término no está aprendiendo correspondencias, sino una dirección común para todo.
- El effective rank centrado sube, pero mide la estructura de unos residuos minúsculos alrededor de esa dirección común. No contradice el collapse en dirección.
- Ya en el baseline los embeddings de `<|eot_id|>` están muy concentrados (0.93): el último token es un punto de partida propicio al collapse, y las dos vistas terminan en el mismo token.
- Nada lo impide: LLM-JEPA no tiene EMA, ni stop-gradient, ni término de varianza.

## Tests del notebook (todos pasan)

1. Integridad de datos: archivos paralelos, sin líneas vacías, splits disjuntos que cubren todo.
2. Máscara: Code no atiende a Text ni a `[PRED]`, causal dentro de cada bloque, posiciones de Code empiezan en 0.
3. Labels NTP: solo cuentan los tokens de la regex (+ `<|eot_id|>`).
4. Equivalencia en fp32: el forward empaquetado da los mismos hidden states que los forwards separados (atol 1e-4). Sin la máscara por bloques la diferencia es de 3.5, así que el test sí puede fallar.
5. Con λ = 0 la loss es igual a la cross-entropy calculada a mano (3.9942).
6. Smoke test de 20 pasos con 8 ejemplos: NTP 3.872 → 0.002, distancia 0.641 → 0.009.

## Problemas técnicos

- **Segfault de PyTorch en WSL2:** el `torch` 2.15 de PyPI (runtime CUDA 13.2) se cae en `cudaGetDeviceCount` al importar `torch.cuda`. El driver funciona (`cuInit` desde ctypes va bien). Solución: `torch==2.14.1+cu126` desde el índice `pytorch-cu126`, configurado en `pyproject.toml`.
- **Llama chat template:** inserta la fecha del día; se fija con `date_string="26 Jul 2024"` para que los prompts sean reproducibles.
- **Pad token:** Llama 3.2 no tiene; se usa `<|finetune_right_pad_id|>`.

## Limitaciones

- Una sola ejecución por método: la diferencia de 0.6 puntos está dentro del ruido esperable (el paper reporta ±5.32 para su baseline).
- Un solo (k, λ). El collapse podría depender de λ, pero con la distancia en 0.004 a los 250 pasos es poco probable que un λ menor lo evite por sí solo.
- Exact match no reconoce regex equivalentes escritas de otra forma.

## Siguientes pasos

1. ~~Añadir el diagnóstico de collapse y monitorizarlo durante el entrenamiento.~~ Hecho en el issue #2: `jepa_llm.diagnostics`, comando `nlrx-diagnose` y métricas `val_cos_*` / `val_erank_*` en `history.json` de las ejecuciones nuevas.
2. Probar anti-collapse: término de varianza/covarianza tipo VICReg o SIGReg, o un target EMA con stop-gradient.
3. Probar otro pooling: mean-pooling sobre los tokens de contenido de cada vista, o un token distinto de `<|eot_id|>`.
4. Si alguna variante evita el collapse, repetir con 2–3 configuraciones de (k, λ) y varias ejecuciones para medir si hay ganancia real.
