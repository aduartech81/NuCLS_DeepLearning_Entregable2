# Cierre operativo de E2

Corte: 12 de noviembre de 2026. Este documento no declara ejecutadas las evaluaciones preparadas. El bloque activo de Colab permanece intacto.

## Recepción en una sola operación

El programa `scripts/recibir_evidencias.py` recibe varios ZIP exportados, comprueba integridad, identifica corridas y recalcula macro-F1, accuracy, MCC, Brier multiclase, ECE con diez intervalos fijos, AUC/AP por clase y métricas descriptivas por paciente. Valida probabilidades y coincidencia del macro-F1 con el resultado registrado. Lee los ZIP directamente, no extrae archivos ni modifica entradas. Rechaza carpetas de informe preexistentes. Mantiene búsqueda y piloto separados.

```
python scripts/recibir_evidencias.py --zip BASES_66_EVIDENCIAS.zip ADICIONALES_RESULTADOS.zip TRES_NUEVOS_RESULTADOS.zip --salida informes_recepcion_01
```

Pasar solo archivos existentes; no sustituir predicciones faltantes por promedios. Requiere NumPy y scikit-learn. La salida contiene `recepcion.json` y `metricas_validacion.csv`. Su análisis es descriptivo y condicionado al ajuste en validación, no evaluación confirmatoria.

## Qué entregar al terminar Colab

1. ZIP de evidencias base de 66 corridas, con los archivos antiguos incluidos; el índice solo no verifica los resultados antiguos.
2. ZIP de la búsqueda actual, protocolo y selección de hiperparámetros. Un archivo de resultados de búsqueda no acredita piloto terminado.
3. Notebook ejecutado con salidas. Los pesos permanecen en Drive; conservar sus carpetas y hashes.
4. Después, evidencias del complemento de tres nuevos modelos y de los pilotos necesarios. Esta etapa todavía requiere GPU y no ocurre automáticamente al recibir el ZIP actual.

## Dependencias que no resuelve la búsqueda actual

- Potencia y benchmark final: diferencias pareadas del piloto para dimensionar semillas; la decisión depende del presupuesto y no se sustituye por los 22 entrenamientos base.
- Robustez y atribuciones: imágenes de desarrollo, pesos y nuevas inferencias; no se pueden reconstruir desde macro-F1.
- EDA de casi duplicados: imágenes y manifiesto; etiquetas inconsistentes y techo multi-rater requieren anotaciones enlazadas. La ausencia de anotaciones se declara, no se imputa un techo.
- Replicación: ejecutar el protocolo/dataset del artículo guía o declarar adaptación metodológica; entrenar en NuCLS no es una replicación numérica de PCam.
- Historial bibliográfico: cadenas y filtros originales, si se conservan. No repetir búsquedas para reconstruir retrospectivamente ese historial.
- Reproducción independiente: ejecución por otro entorno/persona con evidencia; no se acredita mediante pruebas locales propias.

## Criterios de aceptación de la entrega

No sustituir una métrica faltante por cero; reportar NR y la causa. No mezclar configuraciones con semillas como repeticiones equivalentes. No seleccionar variantes, hiperparámetros o muestras con la reserva histórica. Conservar el presupuesto real, los fallos y el alcance exploratorio. Publicar resultados de validación y sus límites aun si el plazo impide cerrar el benchmark confirmatorio.
