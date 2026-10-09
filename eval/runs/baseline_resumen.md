## Resultados `baseline` (16 preguntas dentro del corpus + 4 de control)

| Metrica | Que mide | `baseline` |
|---|---|---|
| faithfulness | Fidelidad al contexto (generacion) | 0.956 |
| answer_relevancy | Relevancia de la respuesta (generacion) | 0.867 |
| context_precision | Precision del contexto (recuperacion) | 0.908 |
| context_recall | Cobertura del contexto (chunking/recuperacion) | 1.000 |
| hit_articulo | Acierto de articulo en top_k (sin LLM) | 1.000 |
| rechazo_correcto | Rechazo correcto en preguntas de control | 1.000 |

Citas de articulos que no estaban en el contexto recuperado: 1 (en 1 respuestas).
Las 4 metricas Ragas se promedian sobre las preguntas *dentro del corpus*; las de control (fuera de alcance / sin respuesta) se juzgan por el comportamiento de rechazo.