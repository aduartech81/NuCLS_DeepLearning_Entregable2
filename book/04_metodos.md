# Métodos y plan de evaluación

## Diseño vigente

Datos, partición, normalización de train, resolución128, batch32, AdamW y float32 comunes. Cuatro configuraciones por arquitectura: lr {0,0001;0,001} × weight decay {0;0,0001}. Máximo25 épocas, mínimo5 y paciencia5. Selección de época e hiperparámetros únicamente con validación. La misma cantidad de intentos no implica igual costo ni una grilla óptima para todas las familias.

El piloto de cinco semillas estima variabilidad y permite un benchmark mínimo exploratorio; no acredita por sí solo potencia suficiente para el benchmark final. El tamaño requerido se calculará con dispersión de diferencias pareadas frente al baseline, δ=0,02, potencia80% y planificación conservadora para diez contrastes. No reutilizar cinco semillas como prueba suficiente de ausencia de diferencia.

## Evaluaciones pendientes y entradas necesarias

| Bloque | Procedimiento previsto | Insumo / restricción |
|---|---|---|
| Calibración | ECE con bordes fijados, Brier multiclase, curva de fiabilidad | Probabilidades, etiquetas y paciente; ajuste de temperatura solo en desarrollo separado |
| Incertidumbre | Tres modelos seleccionados por validación; cobertura y tamaño de conjuntos predictivos | Subconjunto de calibración independiente del ajuste y evaluación agrupada; no prometer cobertura por paciente a partir de intercambio de recortes |
| Robustez | Tres niveles de brillo, desenfoque y ruido; semilla y parámetros fijados antes de evaluar | Pares limpio/perturbado y agrupación por paciente; parámetros concretos pendientes de congelar |
| Interpretabilidad | Muestras estratificadas fijas, atribuciones compatibles con cada arquitectura | Checkpoints; fidelidad por eliminación/inserción y estabilidad entre semillas; mapas no son evidencia causal |
| Costo | Parámetros, operaciones, memoria máxima y latencia tras calentamiento | Misma GPU, tamaño de lote y sincronización CUDA; perfil no sustituye latencia de inferencia |
| Sobreajuste | Curvas train/validación, brecha y prueba de lote pequeño | Historias completas; no interpretar igualdad de medias como ausencia de sobreajuste |

Los intervalos entre semillas describen variabilidad de optimización condicionada a la partición. El bootstrap por paciente responde otra pregunta. Informar ambas sin elegir retrospectivamente la que favorezca un modelo. Los datos de validación usados para selección no sirven como evaluación confirmatoria independiente.

La evaluación final conserva la reserva heredada. Tras congelar arquitecturas, hiperparámetros, épocas y semillas, se evaluará cada checkpoint del lote una vez, sin reajustes posteriores. Se declara su exposición previa en E1: sigue siendo una partición de prueba separada del entrenamiento, pero no un conjunto nunca visto por el equipo. Las predicciones históricas de checkpoints idénticos se reutilizarán si sus hashes coinciden. No se modificará la partición ni se seleccionarán modelos o semillas con la reserva.

## Configuraciones efectivas de AdamW

La grilla ejecutó cuatro intentos nominales. Con lr=10⁻⁴ y weight decay=10⁻⁴, el factor de decaimiento 1−lr·wd = 1−10⁻⁸ se redondea a 1 en float32. La prueba local de AdamW con gradiente cero reproduce ausencia de cambio tanto con foreach activado como desactivado. Con lr=10⁻³, el factor representable es 0,9999998807907104 y sí hay decaimiento. Esto explica los empates observados entre configuraciones 0 y 1 y reduce a tres las configuraciones numéricamente distintas de esta receta.

La duplicación afecta por igual a la grilla común y no justifica repetir ahora la búsqueda ni ampliar el espacio tras ver resultados. Se conserva el costo de los cuatro intentos; no se cuentan los empates como réplicas independientes. La verificación local CPU y su versión están en `evidencias/revision_docente_20261114/adamw_verificado.json`; no se atribuye a una prueba nueva en la GPU activa.

## Plan operativo del cierre

Los contrastes, márgenes, población de remuestreo y límites del cuaderno preparado se especifican en [Alcance y reproducibilidad](07_reproducibilidad.md). Preparación y ejecución se informan como estados distintos.
