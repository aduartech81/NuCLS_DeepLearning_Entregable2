# Análisis exploratorio del conjunto de desarrollo

<!-- PRESENTACION_CAPITULO -->
<div class="chapter-deck"><p class="book-eyebrow">02 / ANÁLISIS EXPLORATORIO</p><p>Calidad, composición y procedencia de la muestra</p><div class="chapter-tags"><span>16.588 recortes</span><span>104 pacientes de desarrollo</span><span>Reserva separada</span></div></div>

<div class="chapter-takeaway"><b>Integridad no equivale a representatividad.</b> Las diferencias de balance y apariencia entre grupos condicionan la interpretación del experimento. El EDA de píxeles se restringe al desarrollo.</div>
<!-- FIN_PRESENTACION_CAPITULO -->

El análisis exploratorio examina tres condiciones que determinan la interpretación del clasificador: la integridad de sus entradas, la representación de las clases y la heterogeneidad entre pacientes y procedencias. El conjunto de desarrollo reúne 16.588 recortes: 14.539 de entrenamiento, procedentes de 83 pacientes, y 2.049 de validación, de 21 pacientes. La separación por paciente y código TSS se mantiene entre particiones. Las mediciones se restringen a estos grupos y preservan la reserva para la evaluación final.

![Composición por clase, retención e imágenes aportadas por paciente.](figuras/eda_desarrollo_v2.svg)

## Integridad y completitud de los datos

Se verificaron los campos paciente, TSS, partición, clase, ruta de archivo y las dos huellas digitales en los 16.588 registros de desarrollo: cada campo presenta **0 valores vacíos (0 %)**. Este resultado describe el manifiesto de recortes seleccionados, no la completitud de todos los metadatos originales de NuCLS. Tampoco demuestra que una etiqueta no vacía sea correcta.

La unidad de entrada es una ventana RGB de 128×128 píxeles vinculada a una anotación, una clase y un paciente. En los 16.588 recortes de desarrollo se completó la lectura y verificación de formato, dimensiones y huellas digitales. La ausencia de fallos en ese conjunto procesado no acredita por sí sola que todos los archivos originales de NuCLS estén completos ni que sus anotaciones sean correctas.

Los faltantes deben distinguirse de las exclusiones geométricas: una ventana que exige relleno no es un valor ausente imputable. El conjunto analítico conserva únicamente recortes elegibles y no incorpora píxeles sintéticos para completar bordes. Las etiquetas no se imputan. Tampoco se imputa la identidad del paciente, indispensable para separar las particiones. Las entradas del clasificador son los píxeles, no metadatos clínicos; por ello no se aplica imputación tabular al modelo actual. No se ha establecido un mecanismo MCAR, MAR o MNAR para los metadatos originales.

La verificación de integridad y la evaluación de calidad visual responden a preguntas distintas. Una imagen puede abrir correctamente y presentar artefactos o una anotación discutible. Los descriptores de color y textura caracterizan heterogeneidad; no constituyen un diagnóstico automático de desenfoque ni validan las etiquetas. La concordancia entre anotadores requiere un análisis específico de anotaciones comparables.

## Balance de clases y criterio de evaluación

La entropía de las tres clases es **1,5068 bits** en entrenamiento y **1,4302 bits** en validación; normalizada por log₂(3), corresponde a **0,9507** y **0,9024**, respectivamente. El máximo normalizado es uno. La menor entropía de validación confirma una composición más concentrada, coherente con su mayor proporción tumoral. La entropía resume balance; no mide dificultad visual ni independencia entre recortes.


La razón entre la clase más frecuente y la menos frecuente es 2,25 en entrenamiento y 3,22 en validación. La mayor concentración de la validación refuerza la necesidad de evaluar cada categoría por separado. El protocolo mantiene entropía cruzada sin ponderación para los tres candidatos, con macro-F1 como criterio de selección y métricas por clase para identificar pérdidas de sensibilidad. La pérdida común permite comparar las arquitecturas bajo el mismo criterio de optimización; el posible perjuicio para la clase minoritaria debe establecerse mediante sus resultados específicos.

La proporción tumoral aumenta 6,09 puntos porcentuales en validación y la estromal disminuye 4,55. Por tanto, la exactitud global depende de una composición distinta en cada grupo. Macro-F1 conserva el mismo peso para las tres categorías, pero no elimina cambios de dominio ni diferencias entre pacientes. Su interpretación debe acompañarse de sensibilidad, precisión y F1 por clase.

## Efecto de la selección espacial

La retención se calcula sobre las anotaciones de cada clase antes de las exclusiones geométricas. En entrenamiento se conserva una proporción menor de linfocitos que de núcleos tumorales: una diferencia descriptiva de 4,34 puntos. La regla espacial modifica así la composición de la muestra. Los resultados posteriores corresponderán a núcleos con una ventana interior elegible, sin generalización automática a núcleos periféricos.

Los intervalos proceden de 5.000 remuestreos de pacientes, con semilla 4101, conservando juntas sus anotaciones. En cada repetición se calcula la razón entre recortes retenidos y anotaciones originales. Son intervalos percentiles, condicionados a los TSS observados; no describen la incertidumbre entre centros ni prueban una diferencia entre clases.

## Agrupamiento y contribución de los pacientes

La mediana es de 121 recortes por paciente en entrenamiento, con rango de 7 a 547, y de 79 en validación, con rango de 2 a 231. Esta heterogeneidad hace que las métricas agregadas otorguen mayor influencia a los pacientes con más recortes. Por ello, la evaluación debe distinguir dos fuentes de incertidumbre: la variación entre pacientes y la variación de optimización entre semillas. Repetir el entrenamiento caracteriza la segunda, pero no incrementa el número de pacientes observados.

## Intensidad y variabilidad de los recortes

Se analizaron los píxeles de los 16.588 recortes de desarrollo, con verificación de dimensiones RGB 128×128 y hashes frente al manifiesto. Las estadísticas de normalización se estimaron exclusivamente en entrenamiento, sobre 238.206.976 píxeles por canal.

| Canal | Media en escala 0–1 | Desviación estándar |
|---|---:|---:|
| Rojo | 0,63920 | 0,21397 |
| Verde | 0,45917 | 0,22150 |
| Azul | 0,62775 | 0,17429 |

La media del canal verde es inferior a las de rojo y azul. La distribución no es equivalente entre canales: una transformación que centre y escale cada uno debe utilizar sus parámetros de entrenamiento. Esta operación no normaliza la tinción ni elimina variación de adquisición. Los valores corresponden a recortes nucleares con contexto; no son mediciones exclusivas del núcleo ni medidas cuantitativas de hematoxilina o eosina.

![Histogramas RGB por clase y partición, normalizados dentro de cada distribución.](figuras/histogramas_pixeles_v2.svg)

Cada curva representa la proporción de píxeles por intervalo dentro de una clase y partición. La normalización permite comparar la forma de las distribuciones sin confundirla con el número de recortes. Los histogramas contienen tanto el núcleo objetivo como su entorno; sus diferencias no pueden atribuirse exclusivamente a morfología celular o a tinción.

## Desplazamiento entre particiones y señal de procedencia

Se construyeron diez descriptores de color y textura por recorte. Para evitar que los pacientes con más imágenes dominaran el diagnóstico, se utilizó la mediana de cada descriptor por paciente. Una regresión logística con pesos balanceados se evaluó en tres pliegues, manteniendo cada paciente en un único grupo y estimando el escalado dentro de entrenamiento. Los hiperparámetros se fijaron antes de la ejecución, sin búsqueda.

| Diagnóstico | Pacientes | Categorías | Exactitud balanceada | AUC |
|---|---:|---:|---:|---:|
| Entrenamiento frente a validación | 104 | 2 | 0,6770 | 0,7097 |
| TSS entre pacientes de entrenamiento | 77 | 8 | 0,3983 | No calculada |

![Matrices de confusión de los diagnósticos, con predicciones fuera del pliegue.](figuras/diagnosticos_pixeles_v2.svg)

Los descriptores permitieron identificar la partición de 61 de 83 pacientes de entrenamiento y de 13 de 21 pacientes de validación. La AUC de 0,7097 muestra una separación parcial entre ambos grupos y aporta evidencia descriptiva de heterogeneidad visual. Su origen no puede atribuirse a un único factor: la composición celular, el tejido, la adquisición y la procedencia pueden contribuir conjuntamente. La AUC se interpreta como exploratoria; los intervalos posteriores corresponden a exactitud balanceada, no a AUC. La partición por TSS controla el solapamiento de procedencia, mientras que el diagnóstico caracteriza las diferencias que persisten entre grupos.

El diagnóstico de procedencia incluyó únicamente los ocho TSS de entrenamiento con al menos tres pacientes. Su exactitud balanceada de 0,3983 supera descriptivamente la referencia uniforme de 0,125 para ocho categorías. Esto indica que los descriptores contienen señal asociada al TSS; no demuestra que una CNN entrenada para clasificar núcleos utilice esa señal. Tampoco estima rendimiento en TSS nuevos. Se mantiene la separación por paciente y TSS, y se evita interpretar la normalización RGB como solución demostrada a este desplazamiento.

## Incertidumbre y heterogeneidad multivariada

![Intervalos de los diagnósticos y distancias robustas por paciente.](figuras/eda_incertidumbre_v2.svg)

Se obtuvieron intervalos percentiles del 95 % para la exactitud balanceada mediante 5.000 remuestreos de pacientes, estratificados por la etiqueta real y con semilla 4101. El intervalo del diagnóstico de partición fue **0,5579–0,7846** y el de procedencia **0,2937–0,5038**. Se mantuvieron fijas las predicciones fuera del pliegue: los intervalos describen variación de composición de pacientes condicionada al ajuste realizado, sin incluir reentrenamiento ni incertidumbre entre centros. No se infiere significación estadística a partir de estos intervalos exploratorios.

El análisis multivariado utiliza las medianas por paciente de diez descriptores RGB, saturación, fondo, Laplaciano y entropía. Se aplica log(1+x) al Laplaciano y se ajustan mediana, IQR y covarianza robusta Minimum Covariance Determinant exclusivamente con los 83 pacientes de entrenamiento. El percentil 95 empírico de sus distancias prioriza **cinco pacientes de entrenamiento y uno de validación**. No se eliminan recortes ni se modifican las particiones. Una distancia elevada expresa una combinación infrecuente de descriptores; no demuestra artefacto, error de etiqueta ni enfermedad distinta.

Estos resultados corresponden a V2. La priorización de cuatro pacientes entre 79, descrita en el análisis histórico de campos, utiliza otra población y otros indicadores. Ambos diagnósticos se conservan identificados para evitar una equivalencia incorrecta.

## Similitud perceptual entre entrenamiento y validación

Se compararon 29.790.411 pares de recortes mediante dHash de 64 bits. No se identificaron candidatos con distancia de Hamming menor o igual a cuatro. La comparación fue recalculada de forma independiente a partir de los hashes exportados. El resultado complementa la comprobación de duplicados exactos, pero no descarta todas las rotaciones, cambios de escala, recortes parcialmente solapados ni otras similitudes que este descriptor puede omitir. No se eliminaron imágenes por este análisis.

## Procedencia y alcance de la verificación

El procesamiento de píxeles se ejecutó en el equipo del autor. La segunda verificación contrastó los 16.588 registros con el manifiesto, recalculó medias y desviaciones RGB mediante los descriptores, comprobó los conteos de los histogramas y reconstruyó las matrices de confusión y sus exactitudes balanceadas. La AUC adversarial se conserva tal como la registró el script: el archivo exporta etiquetas predichas, pero no sus probabilidades, por lo que esa AUC no se recalculó de forma independiente. La verificación secundaria se realizó sobre los archivos de resultados y los descriptores exportados, sin repetir la lectura de las imágenes originales.

No se utilizaron imágenes de reserva. Las mediciones de calidad de campos del piloto se conservan en su apartado original y no se atribuyen a los recortes de esta nueva partición.

## Implicaciones para el diseño experimental

El EDA fundamenta tres decisiones del experimento. Primero, la selección de ventanas interiores define la población de recortes sobre la que son válidos los resultados. Segundo, el desequilibrio y el cambio de composición entre particiones justifican priorizar macro-F1 y acompañarlo de sensibilidad, precisión y F1 por clase. Tercero, la contribución desigual de los pacientes y la señal de procedencia obligan a conservar la agrupación en la evaluación y a interpretar con cautela la generalización.

La estandarización RGB utiliza exclusivamente estadísticas de entrenamiento. Su función es establecer una escala común de entrada; no se le atribuye la corrección del desplazamiento de dominio. La integridad de archivos y la ausencia de candidatos de alta similitud bajo el criterio aplicado respaldan controles concretos del conjunto, pero la validez de las etiquetas requiere evidencia específica de anotación.

## Decisiones analíticas derivadas de los datos

| Evidencia | Decisión aplicada | Alcance |
|---|---|---|
| Identificadores y hashes completos | Mantener vinculación a paciente y TSS; no imputar identidad ni etiqueta | Integridad del manifiesto seleccionado |
| Diferencias de balance y retención | Macro-F1 y métricas por categoría; pérdida común sin ponderación | No demuestra que el desbalance esté corregido |
| Ventanas interiores elegibles | Entrada RGB 128×128 sin relleno | Población condicionada a la selección geométrica |
| Heterogeneidad RGB | Estandarización con media y DE de entrenamiento | No equivale a normalización de tinción |
| Señal de partición y TSS | Conservar separación por procedencia y evaluación agrupada | No identifica por sí sola la causa del desplazamiento |
| Distancias multivariadas elevadas | Priorizar inspección, sin excluir pacientes | Indicador exploratorio, sin adjudicación visual |

Las tablas numéricas completas se conservan como evidencia descargable; las figuras del capítulo presentan distribuciones y relaciones, sin repetir íntegramente los mismos valores. El análisis no incorpora imágenes de reserva ni resultados del entrenamiento en curso.


## Cierre documental y límite de las etiquetas

La revisión del capítulo y del resumen de cierre confirma que los análisis descritos corresponden al desarrollo V2 y que su incertidumbre se interpreta por paciente. Esta conciliación documental no repite la lectura de píxeles ni aporta nuevas mediciones. Permanece pendiente la concordancia entre evaluadores sobre objetos alineados: se necesitan identificador del mismo núcleo, identificador del evaluador y etiqueta individual previa a la adjudicación. La etiqueta de consenso o la comparación entre predicción y etiqueta no sustituyen esas observaciones. Hasta disponer y verificar ese emparejamiento no se informa un coeficiente de concordancia ni se asume independencia entre anotaciones. Los resultados publicados de concordancia de NuCLS son antecedentes del dataset, no una estimación propia de los recortes seleccionados.
