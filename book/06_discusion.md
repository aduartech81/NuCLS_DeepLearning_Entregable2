# Discusión, paper guía y gap

La diferencia de validación ResNet50–DenseNet121 con 22 semillas no resulta concluyente en las pruebas pareadas corregidas. Esto no demuestra equivalencia. La superioridad respecto a la CNN-BN aparece en desarrollo, condicionado al proceso de selección y la partición. El análisis de las 22 semillas base usa validación. El cierre previsto evaluará la reserva heredada tras congelar el lote y declarará su exposición previa en E1.

## Propuesta de paper guía

Springenberg et al. (2023), *From Modern CNNs to Vision Transformers*, DOI 10.1016/j.media.2023.102809, ofrece una comparación de arquitecturas y condiciones de entrenamiento en histopatología. Se propone como guía del protocolo comparativo. Entrenar variantes a128 en NuCLS no replica por sí mismo sus resultados en PCam: cambian datos, tarea y receta. Esta edición no aporta una réplica numérica externa del artículo; su uso es metodológico y comparativo.

## Gap e hipótesis de trabajo

La evidencia reunida no establece cuál familia mantiene mejor rendimiento y calibración bajo el presupuesto común de este estudio y desplazamiento de sitio en clasificación nuclear. La hipótesis exploratoria es que incorporar contexto o estructura espacial puede mejorar discriminación, con un costo y sensibilidad al dominio que deben medirse. Para E3, mecanismos de contexto multiescala y mezcla espacial son candidatos; no se atribuye novedad mundial ni una ganancia comprobada.

El techo por desacuerdo multi-rater no está estimado. Sin una correspondencia de etiquetas y una medida comparable no se calcula una distancia normalizada al techo. Las fuentes de incertidumbre restantes incluyen pocos pacientes de reserva histórica, ajuste limitado, adaptación de recetas, selección sobre validación y presupuesto de semillas aún abierto.

## Equidad de presupuesto y adecuación de la receta

Compartir resolución, intentos y criterio de parada no garantiza una receta igualmente adecuada para cada familia. DeiT, Swin y Mixer se entrenan desde cero con 14.539 recortes, sin warmup ni scheduler. El deterioro de Swin con lr=0,001 es evidencia de sensibilidad bajo esta receta, no inferioridad general de los Transformers. ResNeXt50 permanece fuera del Top 10 pese a su resultado exploratorio: incorporarlo por ese resultado introduciría selección posterior.

## Lectura conjunta de desarrollo y prueba

La discusión final confrontará, por cada arquitectura, la media y dispersión de validación con la reserva. Un descenso es compatible con desplazamiento de dominio, sobreajuste de selección o heterogeneidad clínica; el diagnóstico adversarial heredado (AUC 0,7097) aporta contexto pero no identifica por sí solo una causa. No se atribuirá una diferencia a la familia arquitectónica sin considerar receta, número de semillas y seis pacientes de reserva.

La elección de lr en el borde superior de la grilla limita la búsqueda. La receta común controla presupuesto, pero no neutraliza diferencias de inductive bias o necesidad de warmup, scheduler y preentrenamiento. No se ampliará retrospectivamente la grilla por los resultados de reserva.

## Variabilidad observada en diez candidatos

El piloto integra cinco semillas en cada uno de los diez modelos. Inception V3 alcanza una media de 0,8235 (DE 0,0132), EfficientNet B0 de 0,8138 (DE 0,0128) y ResNet18 de 0,8041 (DE 0,0102). MobileNetV3 Small y Mixer redondean ambos a 0,7758, con DE de 0,0099 y 0,0356: la misma media puede ocultar distinta estabilidad. El orden es descriptivo y no modifica el baseline heredado ni la selección bibliográfica.

La selección de época e hiperparámetros utiliza validación; las diferencias deben interpretarse con ese condicionamiento y la dependencia entre recortes de un paciente. La evaluación de reserva no se sustituye por estos resultados.
