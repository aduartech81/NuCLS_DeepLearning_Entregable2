# NuCLS · Entregable 2

Andrés Duarte Chaves · Deep Learning · 15 de noviembre de 2026.

## Acceso

Abrir `ABRIR_ENTREGA.html` después de extraer el ZIP completo. El portal enlaza el Jupyter Book, el dashboard y las evidencias numéricas. La entrega es independiente de E1.

## Contenido científico

Diez arquitecturas adicionales con cinco semillas cada una: 50 corridas y 102.450 predicciones de validación verificadas. Se incluye el análisis de 22 semillas de las tres bases, PRISMA, selección bibliográfica, EDA y 36 ejemplos visuales de 21 pacientes con anotaciones originales.

ResNet50 conserva su papel de baseline. Los intervalos entre semillas están condicionados a selección en validación; no son incertidumbre entre pacientes. La reserva fue consultada en E1 y no se presenta como prueba inédita. Los resultados tabulados de esta edición corresponden a validación.

## Reproducibilidad

`evidencias/` contiene resultados originales y cálculos. `book/` y `dashboard/` contienen la presentación. `modelos/`, `scripts/` y los cuadernos documentan la implementación. Las salidas experimentales se distinguen del código preparado; los notebooks sin salidas no se presentan como ejecutados. Los pesos permanecen en el almacenamiento experimental.

## Cuaderno ejecutado

Se incluye `colab/ejecutados/NuCLS_E2_Cola_Unica_Ejecutado.ipynb`, con diez celdas de código con salidas y sin errores registrados. Su salida final acredita `COLA_EXPLORATORIA_COMPLETADA`; no acredita evaluación de reserva.


## Actualización online · 16 de noviembre de 2026

Se incorpora el cierre de reserva (13 modelos, cinco semillas comunes, 65 evaluaciones), con estadística y evidencia descargable. La publicación complementa la versión entregada el 15 de noviembre; su ZIP permanece inalterado. Véase `book/14_cierre_reserva.md`.
