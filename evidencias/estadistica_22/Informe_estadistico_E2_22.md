# Validación de los modelos base — 22 semillas

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

## Próximo bloque
Diez candidatos en cuatro familias. Primero perfilado y piloto; fijar después el tamaño final con dispersión pareada y presupuesto observado. Si el plazo impide potencia 80%, declarar limitación y MDE, sin presentar ausencia de significación como equivalencia.
