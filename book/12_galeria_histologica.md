# Galería ampliada: imágenes limpias, anotaciones y predicciones

**36 núcleos diferentes de 21 pacientes**:18 aciertos y18 discrepancias. Dos casos por combinación de tres modelos, tres clases y dos resultados, con semilla6113 fija. Se priorizó diversidad de pacientes; el desempate es determinista por paciente y archivo. No se escogieron semillas por desempeño.

## Fuente y cotejo

Se recuperó el archivo original QC con campos RGB, visualizaciones oficiales y CSV. Se verificaron **6.147 predicciones sobre2.049 núcleos de21 pacientes de validación V2**: identificador de campo, fila de anotación, paciente, clase de referencia, suma de probabilidades y argmax. Se guardan los CSV de predicciones y las anotaciones exactas de cada imagen, con huellas. [Resumen verificable](../evidencias/galeria_ampliada_20261115/Verificacion.json) y [cotejo completo](../evidencias/galeria_ampliada_20261115/Cotejo_predicciones.csv).

Se muestra el **campo limpio original**, junto al mismo campo con la anotación del núcleo objetivo resaltada. La visualización oficial QC de todas las anotaciones está enlazada por caso. El modelo clasifica un recorte del núcleo identificado por el sufijo de fila; la imagen de contexto no se presenta como su entrada exacta de128×128. No se reabrió Colab ni se verificaron allí los bytes del ZIP de entrenamiento. La ampliación no depende del manifiesto histórico de recortes ni cambia ninguna partición.

## Alcance

Las imágenes y referencias son existentes, no generadas. El resaltado vectorial usa las coordenadas originales, sin inventar contornos. Una caja rectangular no equivale a segmentación; ninguna superposición es Grad-CAM. «Discrepancia» significa diferencia frente a la etiqueta registrada, no causa de error adjudicada. La selección estratificada por resultado no estima tasas, prevalencias ni rendimiento clínico. La revisión de100 errores por modelo y el acuerdo multi-rater siguen siendo tareas distintas. No se visualizaron imágenes de reserva.

## Distribución de los ejemplos

| Paciente | Casos |
|---|---:|
| TCGA-AQ-A04J | 2 |
| TCGA-AQ-A54N | 2 |
| TCGA-BH-A0AV | 2 |
| TCGA-BH-A0B3 | 2 |
| TCGA-BH-A0B9 | 2 |
| TCGA-BH-A0BG | 2 |
| TCGA-BH-A0BL | 2 |
| TCGA-BH-A0BW | 2 |
| TCGA-BH-A0E0 | 2 |
| TCGA-BH-A0E6 | 2 |
| TCGA-BH-A0WA | 2 |
| TCGA-BH-A18G | 2 |
| TCGA-BH-A18V | 1 |
| TCGA-BH-A1EW | 2 |
| TCGA-BH-A1F6 | 2 |
| TCGA-OL-A5D6 | 1 |
| TCGA-OL-A5D7 | 2 |
| TCGA-OL-A5RW | 1 |
| TCGA-OL-A66I | 1 |
| TCGA-OL-A66P | 1 |
| TCGA-OL-A6VO | 1 |

## Casos cotejados

| Caso | Paciente | Modelo | Referencia | Predicción | Resultado | Imagen anotada |
|---|---|---|---|---|---|---|
| 01 | TCGA-BH-A18V | resnet50 | Tumor no mitótico | Tumor no mitótico | Acierto | [Ver](../evidencias/galeria_ampliada_20261115/caso_01/referencia.svg) |
| 02 | TCGA-OL-A5D6 | resnet50 | Tumor no mitótico | Tumor no mitótico | Acierto | [Ver](../evidencias/galeria_ampliada_20261115/caso_02/referencia.svg) |
| 03 | TCGA-BH-A0B3 | densenet121 | Tumor no mitótico | Tumor no mitótico | Acierto | [Ver](../evidencias/galeria_ampliada_20261115/caso_03/referencia.svg) |
| 04 | TCGA-BH-A0B9 | densenet121 | Tumor no mitótico | Tumor no mitótico | Acierto | [Ver](../evidencias/galeria_ampliada_20261115/caso_04/referencia.svg) |
| 05 | TCGA-BH-A0E6 | softmaxcnn_bn | Tumor no mitótico | Tumor no mitótico | Acierto | [Ver](../evidencias/galeria_ampliada_20261115/caso_05/referencia.svg) |
| 06 | TCGA-BH-A18G | softmaxcnn_bn | Tumor no mitótico | Tumor no mitótico | Acierto | [Ver](../evidencias/galeria_ampliada_20261115/caso_06/referencia.svg) |
| 07 | TCGA-AQ-A04J | resnet50 | Tumor no mitótico | Linfocito | Discrepancia | [Ver](../evidencias/galeria_ampliada_20261115/caso_07/referencia.svg) |
| 08 | TCGA-AQ-A54N | resnet50 | Tumor no mitótico | Estroma no TIL / no macrófago | Discrepancia | [Ver](../evidencias/galeria_ampliada_20261115/caso_08/referencia.svg) |
| 09 | TCGA-BH-A0BL | densenet121 | Tumor no mitótico | Estroma no TIL / no macrófago | Discrepancia | [Ver](../evidencias/galeria_ampliada_20261115/caso_09/referencia.svg) |
| 10 | TCGA-BH-A0BW | densenet121 | Tumor no mitótico | Estroma no TIL / no macrófago | Discrepancia | [Ver](../evidencias/galeria_ampliada_20261115/caso_10/referencia.svg) |
| 11 | TCGA-BH-A1EW | softmaxcnn_bn | Tumor no mitótico | Linfocito | Discrepancia | [Ver](../evidencias/galeria_ampliada_20261115/caso_11/referencia.svg) |
| 12 | TCGA-BH-A1F6 | softmaxcnn_bn | Tumor no mitótico | Estroma no TIL / no macrófago | Discrepancia | [Ver](../evidencias/galeria_ampliada_20261115/caso_12/referencia.svg) |
| 13 | TCGA-AQ-A04J | resnet50 | Linfocito | Linfocito | Acierto | [Ver](../evidencias/galeria_ampliada_20261115/caso_13/referencia.svg) |
| 14 | TCGA-OL-A6VO | resnet50 | Linfocito | Linfocito | Acierto | [Ver](../evidencias/galeria_ampliada_20261115/caso_14/referencia.svg) |
| 15 | TCGA-BH-A0BG | densenet121 | Linfocito | Linfocito | Acierto | [Ver](../evidencias/galeria_ampliada_20261115/caso_15/referencia.svg) |
| 16 | TCGA-BH-A0BL | densenet121 | Linfocito | Linfocito | Acierto | [Ver](../evidencias/galeria_ampliada_20261115/caso_16/referencia.svg) |
| 17 | TCGA-BH-A0WA | softmaxcnn_bn | Linfocito | Linfocito | Acierto | [Ver](../evidencias/galeria_ampliada_20261115/caso_17/referencia.svg) |
| 18 | TCGA-BH-A1EW | softmaxcnn_bn | Linfocito | Linfocito | Acierto | [Ver](../evidencias/galeria_ampliada_20261115/caso_18/referencia.svg) |
| 19 | TCGA-BH-A0AV | resnet50 | Linfocito | Tumor no mitótico | Discrepancia | [Ver](../evidencias/galeria_ampliada_20261115/caso_19/referencia.svg) |
| 20 | TCGA-BH-A0B3 | resnet50 | Linfocito | Estroma no TIL / no macrófago | Discrepancia | [Ver](../evidencias/galeria_ampliada_20261115/caso_20/referencia.svg) |
| 21 | TCGA-BH-A0E0 | densenet121 | Linfocito | Tumor no mitótico | Discrepancia | [Ver](../evidencias/galeria_ampliada_20261115/caso_21/referencia.svg) |
| 22 | TCGA-BH-A0E6 | densenet121 | Linfocito | Estroma no TIL / no macrófago | Discrepancia | [Ver](../evidencias/galeria_ampliada_20261115/caso_22/referencia.svg) |
| 23 | TCGA-OL-A66I | softmaxcnn_bn | Linfocito | Estroma no TIL / no macrófago | Discrepancia | [Ver](../evidencias/galeria_ampliada_20261115/caso_23/referencia.svg) |
| 24 | TCGA-OL-A66P | softmaxcnn_bn | Linfocito | Tumor no mitótico | Discrepancia | [Ver](../evidencias/galeria_ampliada_20261115/caso_24/referencia.svg) |
| 25 | TCGA-AQ-A54N | resnet50 | Estroma no TIL / no macrófago | Estroma no TIL / no macrófago | Acierto | [Ver](../evidencias/galeria_ampliada_20261115/caso_25/referencia.svg) |
| 26 | TCGA-BH-A0AV | resnet50 | Estroma no TIL / no macrófago | Estroma no TIL / no macrófago | Acierto | [Ver](../evidencias/galeria_ampliada_20261115/caso_26/referencia.svg) |
| 27 | TCGA-BH-A0BW | densenet121 | Estroma no TIL / no macrófago | Estroma no TIL / no macrófago | Acierto | [Ver](../evidencias/galeria_ampliada_20261115/caso_27/referencia.svg) |
| 28 | TCGA-BH-A0E0 | densenet121 | Estroma no TIL / no macrófago | Estroma no TIL / no macrófago | Acierto | [Ver](../evidencias/galeria_ampliada_20261115/caso_28/referencia.svg) |
| 29 | TCGA-BH-A1F6 | softmaxcnn_bn | Estroma no TIL / no macrófago | Estroma no TIL / no macrófago | Acierto | [Ver](../evidencias/galeria_ampliada_20261115/caso_29/referencia.svg) |
| 30 | TCGA-OL-A5D7 | softmaxcnn_bn | Estroma no TIL / no macrófago | Estroma no TIL / no macrófago | Acierto | [Ver](../evidencias/galeria_ampliada_20261115/caso_30/referencia.svg) |
| 31 | TCGA-BH-A0B9 | resnet50 | Estroma no TIL / no macrófago | Linfocito | Discrepancia | [Ver](../evidencias/galeria_ampliada_20261115/caso_31/referencia.svg) |
| 32 | TCGA-BH-A0BG | resnet50 | Estroma no TIL / no macrófago | Linfocito | Discrepancia | [Ver](../evidencias/galeria_ampliada_20261115/caso_32/referencia.svg) |
| 33 | TCGA-BH-A0WA | densenet121 | Estroma no TIL / no macrófago | Linfocito | Discrepancia | [Ver](../evidencias/galeria_ampliada_20261115/caso_33/referencia.svg) |
| 34 | TCGA-BH-A18G | densenet121 | Estroma no TIL / no macrófago | Linfocito | Discrepancia | [Ver](../evidencias/galeria_ampliada_20261115/caso_34/referencia.svg) |
| 35 | TCGA-OL-A5D7 | softmaxcnn_bn | Estroma no TIL / no macrófago | Tumor no mitótico | Discrepancia | [Ver](../evidencias/galeria_ampliada_20261115/caso_35/referencia.svg) |
| 36 | TCGA-OL-A5RW | softmaxcnn_bn | Estroma no TIL / no macrófago | Tumor no mitótico | Discrepancia | [Ver](../evidencias/galeria_ampliada_20261115/caso_36/referencia.svg) |

La galería interactiva del dashboard permite filtrar por paciente, clase, resultado y modelo. Las tablas de rendimiento completo permanecen separadas de esta selección ilustrativa.
