# Reproducibilidad

> Actualización online del 16 de noviembre de 2026: consultar el [cierre de reserva y estadística](14_cierre_reserva.md). Los análisis de validación que siguen conservan su alcance original.


El paquete conserva resultados, historias, predicciones y huellas SHA256. Los archivos originales y las métricas recalculadas permiten contrastar los valores del informe. Los checkpoints no se redistribuyen en este paquete.

## Organización

- `evidencias/`: tablas, predicciones y trazabilidad.
- `modelos/`, `scripts/`: implementación y análisis.
- `colab/`: cuadernos de ejecución.
- `cuaderno_unico/`: procedimiento reproducible de cierre y experimentos.
- `book/` y `dashboard/`: fuentes y presentación de resultados.

Los cuadernos sin salidas son procedimientos reproducibles y no se contabilizan como evidencia de ejecución. La evidencia experimental de esta edición procede de los archivos de resultados e historias incluidos.

## Alcance estadístico

La selección de hiperparámetros y época utiliza validación. Las 22 semillas de las bases y las cinco de los candidatos tienen denominadores distintos y se presentan por separado. El baseline permanece fijado antes de la comparación. La reserva heredada fue consultada en E1, circunstancia que debe acompañar cualquier evaluación posterior.

La inspección ilustrada utiliza anotaciones originales; no constituye adjudicación histológica independiente, techo multi-rater ni revisión de cien errores por modelo. La réplica externa y las ablaciones no forman parte de los resultados tabulados en esta edición.

## Cuaderno ejecutado

Se incluye `colab/ejecutados/NuCLS_E2_Cola_Unica_Ejecutado.ipynb`, con diez celdas de código con salidas y sin errores registrados. Su salida final acredita `COLA_EXPLORATORIA_COMPLETADA`; no acredita evaluación de reserva.
