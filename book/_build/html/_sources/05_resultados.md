# Validación de los modelos base — 22 semillas

> Actualización online del 16 de noviembre de 2026: consultar el [cierre de reserva y estadística](14_cierre_reserva.md). Los análisis de validación que siguen conservan su alcance original.


Análisis exploratorio posterior al cierre. Semillas pareadas 6101–6122; misma partición. La validación seleccionó época y antes hiperparámetros: no es prueba independiente. Los intervalos entre semillas no representan incertidumbre entre pacientes. Reserva no evaluada en esta continuación.

|Modelo|Media|DE|IC95 entre semillas|
|---|---:|---:|---|
|softmaxcnn_bn|0.7600|0.0219|[0.7503; 0.7697]|
|densenet121|0.7846|0.0217|[0.7750; 0.7942]|
|resnet50|0.7922|0.0227|[0.7822; 0.8023]|

## Comparaciones pareadas
Corrección Holm dentro de las tres comparaciones, por cada método. t es el análisis principal; Wilcoxon y permutación son sensibilidad, no tres oportunidades de declarar éxito. IC95 de diferencias sin ajuste simultáneo. Shapiro examina diferencias; no determina retrospectivamente qué prueba se publica.

|Comparación|Δ F1|t Holm|Wilcoxon Holm|Permutación Holm|dz|
|---|---:|---:|---:|---:|---:|
|resnet50 - densenet121|0.0077|0.17452|0.12888|0.17496|0.300|
|resnet50 - softmaxcnn_bn|0.0322|0.00067846|0.0010858|0.00138|0.947|
|densenet121 - softmaxcnn_bn|0.0246|0.0010475|0.0016184|0.00138|0.872|

Friedman: χ²=18.0909, p=0.00011793; Kendall W=0.4112. Post hoc Nemenyi en archivo separado.

## Potencia y relevancia
δ=0,02 es el umbral de relevancia heredado. MDE calculada con t no central, 80% de potencia y Bonferroni conservador (tres contrastes exploratorios o diez modelos contra baseline). No es potencia observada calculada a partir del efecto obtenido. Las dispersiones base son escenarios de planificación, no sustituyen cinco semillas piloto de los nuevos modelos.

- resnet50 - densenet121: DE pareada 0.0255; MDE con R=22/familia10=0.0220; R estimado para δ=0,02/familia10=26.
- resnet50 - softmaxcnn_bn: DE pareada 0.0340; MDE con R=22/familia10=0.0292; R estimado para δ=0,02/familia10=43.
- densenet121 - softmaxcnn_bn: DE pareada 0.0282; MDE con R=22/familia10=0.0242; R estimado para δ=0,02/familia10=31.

## Alcance de evidencia
31 resultados nuevos verificados contra SHA256; 35 previos solo referenciados por el índice. Los checkpoints permanecen en Drive y no se han verificado sus bytes aquí. No se calculan McNemar, DeLong ni bootstrap completo por paciente con medias por semilla: requieren las predicciones antiguas y nuevas alineadas. No se cuentan 22 semillas como 22 cohortes independientes.

## Integración de candidatos adicionales

Los diez candidatos cuentan con cinco semillas completas cada uno. Véase [la comparación de 50 corridas](13_piloto_diez.md). La potencia del piloto no permite interpretar ausencia de significación como equivalencia.

## Verificación adicional de predicciones disponibles

Se procesaron 31 corridas con predicciones individuales incluidas en el ZIP recibido. Su macro-F1 recalculado coincide con el resultado guardado. Las otras 35 de las 66 corridas siguen referenciadas por índice: no se extrapola esta comprobación a sus archivos. Se incluyen métricas por paciente, matrices de confusión, AUC/AP por clase y tablas de fiabilidad en `evidencias/recepcion_bases_disponibles/recepcion.json`.

| Modelo | Corridas verificadas | Brier medio (suma multiclase) | ECE medio (10 intervalos) |
|---|---:|---:|---:|
|densenet121|10|0.2301|0.0318|
|resnet50|11|0.2234|0.0252|
|softmaxcnn_bn|10|0.2564|0.0456|

Son resúmenes descriptivos de las corridas disponibles, con cobertura desigual; no constituyen un ranking completo ni calibración en test independiente. AUC no se calcula si una clase carece de positivos o negativos.

## Búsqueda adicional recibida desde Drive — 14 de noviembre

Se recibieron la selección de hiperparámetros y los diez resultados individuales seleccionados de la búsqueda exploratoria. Se comprobó la coincidencia de modelo, semilla7001 e hiperparámetros, y se recalculó macro-F1 desde cada matriz de confusión (2.049 recortes). Los cuatro valores por modelo proceden del archivo de selección; esta verificación no sustituye la auditoría de todas las predicciones de las 40 corridas.

**Una semilla de búsqueda por configuración, no un benchmark multisemilla.** El máximo sobre cuatro ajustes usa la misma validación para seleccionar época y configuración. No se compara estadísticamente este máximo con la media de 22 semillas de ResNet50 ni se declara superioridad. No hay barras de error porque estas observaciones no aportan estimación entre semillas.

| Modelo | Papel en selección final | lr | weight decay | Macro-F1 seleccionado | Épocas |
|---|---|---:|---:|---:|---:|
|efficientnet_b0|retenido|0.001|0.0001|0.8245|21|
|resnext50_32x4d|exploratorio_fuera_Top10|0.001|0|0.8162|14|
|resnet18|retenido|0.001|0.0001|0.8045|14|
|mobilenetv3_small_100|retenido|0.001|0|0.7846|14|
|mobilevit_xs|exploratorio_fuera_Top10|0.0001|0|0.7765|14|
|mobilevit_xxs|retenido|0.0001|0|0.7762|19|
|resmlp_12_224|exploratorio_fuera_Top10|0.0001|0|0.7739|25|
|mixer_s16_224|retenido|0.0001|0|0.7688|21|
|swin_tiny_patch4_window7_224|retenido|0.0001|0|0.7333|8|
|deit_tiny_patch16_224|retenido|0.0001|0|0.7305|14|

EfficientNet-B0 obtiene el mayor valor seleccionado en esta búsqueda (0,8245), seguido por ResNeXt50 (0,8162), que permanece exploratorio fuera del Top10 bibliográfico. El rendimiento no se usa para revertir retrospectivamente esa selección. Swin presenta valores 0,4124 y 0,2340 en las configuraciones de lr0,001; muestra sensibilidad al ajuste bajo esta receta, no inferioridad general de su familia.

### Comprobación de los empates

Las configuraciones0 y1 (lr0,0001; weight decay0 frente a0,0001) tienen exactamente el mismo macro-F1 en los diez candidatos. En ResNet18 y ConvNeXt se compararon las historias y coinciden las métricas por época, con tiempos distintos. Los archivos registran weight decay diferente. La verificación local de AdamW explica esta coincidencia mediante el redondeo float32 del factor de decaimiento, descrito a continuación. No se afirma que sean trayectorias independientes ni se cambia la receta ya ejecutada.

Las evidencias y sus enlaces están en `evidencias/drive_20261114/`. Se preservan los archivos originales en Drive y se distingue este análisis de búsqueda de la estadística base y del piloto multisemilla.

## Configuraciones efectivas de AdamW

La grilla ejecutó cuatro intentos nominales. Con lr=10⁻⁴ y weight decay=10⁻⁴, el factor de decaimiento 1−lr·wd = 1−10⁻⁸ se redondea a 1 en float32. La prueba local de AdamW con gradiente cero reproduce ausencia de cambio tanto con foreach activado como desactivado. Con lr=10⁻³, el factor representable es 0,9999998807907104 y sí hay decaimiento. Esto explica los empates observados entre configuraciones 0 y 1 y reduce a tres las configuraciones numéricamente distintas de esta receta.

La duplicación afecta por igual a la grilla común y no justifica repetir ahora la búsqueda ni ampliar el espacio tras ver resultados. Se conserva el costo de los cuatro intentos; no se cuentan los empates como réplicas independientes. La verificación local CPU y su versión están en `evidencias/revision_docente_20261114/adamw_verificado.json`; no se atribuye a una prueba nueva en la GPU activa.

## Alcance de la evaluación

Las tablas de esta edición corresponden a desarrollo y validación. No contienen resultados de reserva de E2. La reserva histórica se distingue del conjunto empleado para seleccionar época e hiperparámetros; su exposición previa en E1 limita cualquier interpretación como prueba inédita. Véase [Alcance y reproducibilidad](07_reproducibilidad.md).
