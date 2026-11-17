# NuCLS · Entregable 2

**Andrés Duarte Chaves · Deep Learning · 15 de noviembre de 2026**

Comparación reproducible de arquitecturas para clasificación nuclear en tres clases: tumor no mitótico, linfocito y estroma. El estudio conserva la partición por paciente y sitio, el baseline ResNet50 y el margen de relevancia práctica δ=0,02 de E1.

Esta edición integra **diez modelos adicionales con cinco semillas por modelo**, 50 corridas verificadas y 102.450 predicciones de validación. Incluye el análisis de 22 semillas de los tres modelos base, la selección bibliográfica, PRISMA, EDA y una galería de 36 núcleos de 21 pacientes cotejados con anotaciones originales.

## Recorrido de lectura

1. Problema, preguntas y EDA.
2. Literatura, selección de arquitecturas y protocolo.
3. Resultados de las bases y comparación de diez candidatos.
4. Imágenes, anotaciones, aciertos y discrepancias.
5. Discusión, alcance científico y reproducibilidad.

La inferencia se interpreta según su unidad de análisis. La variabilidad entre semillas no equivale a incertidumbre entre pacientes. Los resultados de desarrollo se distinguen del cierre de la reserva histórica, ya consultada en E1.


**Actualización online · 16 de noviembre de 2026:** [cierre de reserva y estadística](14_cierre_reserva.md), 13 modelos y 65 evaluaciones. Complementa la versión entregada sin sustituir su archivo.


**Ampliación online · 16 de noviembre de 2026:** [ablaciones, ensambles y Grad-CAM](16_experimentos_post.md), con evidencia verificada y límites de interpretación. El ZIP entregado permanece como versión de entrega.


**17 de noviembre de 2026:** [tabla de trazabilidad de correcciones de E2](17_correcciones_e2.md), con evidencias, impacto y límites para la transición a E3.
