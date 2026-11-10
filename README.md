# NuCLS · Entregable 2

**Andrés Duarte Chaves · Maestría en Ingeniería Biomédica · Fecha límite: 14 de noviembre de 2026**

Evolución independiente del [Entregable 1](https://github.com/aduartech81/NuCLS_DeepLearning), fijado como antecedente en el commit `9eb1c3037a648e0ff5bbf9748445fedf653b441a`. Este repositorio no despliega ni modifica el dashboard, el libro o los resultados anteriores.

## Primero: completar las semillas de los modelos base

El cierre anterior contiene 11 semillas por modelo. Este bloque completa exclusivamente el plan original de 22, con los mismos modelos, hiperparámetros, partición y motor. **No ejecuta búsqueda ni piloto, no selecciona otro baseline y no evalúa la reserva.** ResNet50 continúa como referencia heredada.

[Abrir notebook nuevo en Colab](https://colab.research.google.com/github/aduartech81/NuCLS_DeepLearning_Entregable2/blob/main/colab/Continuar_NuCLS_E2.ipynb) · [Ver archivo](colab/Continuar_NuCLS_E2.ipynb). Código de continuación fijado en `da26ec58e232d3079d6c69e21f7f0ef1af36ca53`.

1. Seleccionar GPU en el nuevo Colab y conectar Drive.
2. Preparar el código y datos de desarrollo con las celdas en orden.
3. Revisar el inventario: completas / reanudar / nuevas.
4. Ejecutar la continuación. Los resultados completos se omiten. Los checkpoints parciales propios se copian antes de reanudar, preservando el origen.

Origen, solo lectura: `/content/drive/MyDrive/NuCLS_V2_RESULTADOS_pool_corregido`.
Salida nueva: `/content/drive/MyDrive/NuCLS_E2_CONTINUACION_BASES_22`.

No ejecutar `modelos/experimento_v2.py`: se conserva íntegro para verificar procedencia, pero incluye el pipeline antiguo completo. El único punto de entrada de continuación es `scripts/continuar_semillas.py`.

## Garantías y límites de recuperación

- Inventario de 66 combinaciones (6101–6122 × tres modelos), no de archivos ZIP de resumen.
- Para omitir una corrida se validan resultado, historia, predicciones, identidad, hiperparámetros y presencia del checkpoint; se contrastan los hashes fijados cuando existen.
- Si falta evidencia del cierre 6101–6111 o una corrida declarada completa está dañada, se detiene; no se reentrena silenciosamente.
- Reanudación desde `ultimo.pt`: pesos, optimizador, historia, mejor modelo y estados aleatorios. Una época interrumpida antes de guardar se repite desde el último checkpoint; no hay recuperación a mitad de lote.
- No cambia épocas, paciencia, batch, precisión, aumentos ni optimizador. CUDA determinista y versiones de PyTorch, torchvision, NumPy y Python compatibles con el origen.
- Checkpoints y curvas siguen guardándose cada época. La recompresión de todas las evidencias se hace al final, evitando el ZIP redundante de cada época.
- Un bloqueo impide dos entrenadores sobre la salida nueva. Tras una desconexión brusca, verificar que el proceso anterior terminó antes de retirar el bloqueo con `--liberar-bloqueo`. Una desconexión del navegador no necesariamente detiene Colab.
- El inventario no inicia entrenamiento. El notebook se entrega sin resultados inventados: las pruebas locales verifican coordinación e integridad, no rendimiento GPU.

## Estado de las siguientes fases

El Top 10, benchmark completo, nuevo Jupyter Book y nuevo dashboard están **en preparación**, no se presentan como ejecutados. La guía exige diez modelos adicionales, los tres modelos base y comparadores clásico y trivial, con al menos cuatro familias en el Top 10. El número de semillas del nuevo benchmark debe planificarse por separado: 22 no garantiza su potencia.

La reserva anterior ya fue abierta. Su exposición queda documentada; no se denominará inédita ni se utilizará para ajustar los nuevos modelos. La política confirmatoria del benchmark debe quedar fijada antes de sus entrenamientos.

## Estructura independiente

- `modelos/`: motor original exacto para continuidad de las bases.
- `scripts/`: inventario, continuación y preparación de desarrollo.
- `colab/`: nuevos cuadernos del Entregable 2.
- `protocolo/`: procedencia, configuración y alcance del bloque inicial.
- `tests/`: pruebas de omisión, recuperación y protección del origen.
- `book/`: espacio propio del informe del Entregable 2.
- `dashboard/`: aplicación futura separada; no tiene acceso al despliegue anterior.

Pruebas sin GPU: `python -m unittest discover -s tests -v`.
