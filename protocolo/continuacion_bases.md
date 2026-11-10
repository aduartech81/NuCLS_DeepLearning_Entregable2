# Continuación del plan original · bloque 01

Fecha de preparación: 8 de noviembre de 2026. Este documento registra prospectivamente la continuación pendiente; no retrofecha el experimento anterior ni lo presenta como un prerregistro nuevo.

- Objetivo: completar las semillas 6101–6122 de CNN-BN, DenseNet121 y ResNet50, reutilizando las ya completas.
- Baseline heredado: ResNet50, elegido antes de la apertura de reserva del Entregable 1.
- Métrica de desarrollo: macro-F1 de validación; máximo por corrida, empate conserva primera época.
- Presupuesto inalterado: batch 32, máximo 25 épocas, paciencia 5, mínimo 5; AdamW, clip de gradiente 5, sin preentrenamiento.
- Hiperparámetros: lr 0,001 en las tres; weight decay 0,0001 en CNN-BN y 0 en DenseNet121/ResNet50.
- Se preservan Python RNG, NumPy RNG, Torch RNG, CUDA RNG y generador del DataLoader al reanudar.
- Datos: manifiesto SHA256 `623adf36520751addc324e44cd1b89d95e47842e53cda494914c0248a8be86d4`; train 14.539/83 pacientes, validación 2.049/21.
- Se mantiene el orden semilla → CNN-BN → DenseNet121 → ResNet50, omitiendo las completas.
- El origen y sus checkpoints no se modifican. Los resultados nuevos se escriben en otra carpeta; el índice enlaza resultados antiguos y nuevos sin duplicar sus pesos.
- No se ejecuta inferencia de reserva. La planificación del benchmark ampliado será un documento posterior antes de entrenarlo.
- Tiempo de referencia: las primeras 33 corridas sumaron 11,12 horas de épocas; las siguientes pueden diferir. No incluye todos los gastos de disco, preparación o disponibilidad del servicio.

Prerregistro del Top 10: pendiente. Requiere RQ, selección bibliográfica, familia de contrastes, potencia, margen de sobreajuste, presupuesto equitativo, política de prueba previamente expuesta y plan de robustez/interpretabilidad/ablación. Ningún entrenamiento del Top 10 se inicia con este script.
