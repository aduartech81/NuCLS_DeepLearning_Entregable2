# Bloque adicional E2 — preparación 11 noviembre 2026

Estado: implementación exploratoria y piloto, no Top 10 bibliográfico definitivo ni benchmark confirmatorio completo. El Entregable 1 no cambia. El plazo del 14 de noviembre no justifica fabricar ejecuciones ni declarar potencia no alcanzada.

## Diseño fijado antes de los nuevos entrenamientos

- RQ exploratoria: ¿cómo cambian macro-F1 de validación, dispersión, costo y errores al introducir cuatro familias bajo datos y presupuesto comunes?
- Diez candidatos: ResNet18, ResNeXt50-32x4d, MobileNetV3-Small, EfficientNet-B0; DeiT-Tiny y Swin-Tiny; MobileViT-XXS y XS; Mixer-S/16 y ResMLP-12. Cuatro familias principales: CNN, atención, híbridos, MLP de visión. Las subfamilias residual/eficiente no se cuentan como familias distintas.
- Implementación timm 1.0.22; adaptación de salida a tres clases y entrada a 128×128. Anchos y profundidades de las variantes nombradas; sin búsqueda de arquitectura arbitraria. Las variantes con sufijo 224 reciben img_size=128; esto cambia cantidad de tokens/posiciones y matrices de mezcla espacial, sin reescalar las imágenes. Es una adaptación al dominio, NO replicación literal del paper. MobileViT usa su adaptación interna de parches; Swin su padding de ventanas. Sin pesos preentrenados ni destilación de DeiT. No se atribuyen al entrenamiento propio las cifras de ImageNet de los papers.
- Mismo manifiesto, train 14.539 recortes/83 pacientes, validation 2.049/21. Normalización de canal fijada con train; aumentos rotación múltiplos de 90° y reflexión horizontal. No se extraen imágenes de reserva.
- Selección de época: mayor macro-F1, empate primera. Máximo 25 épocas, mínimo 5, paciencia 5, batch 32, AdamW, clipping 5, float32, determinismo estricto.
- Grilla pequeña de cuatro configuraciones por candidato: lr {0,0001; 0,001} × weight decay {0; 0,0001}; semilla de búsqueda 7001. Configuración ganadora por F1, empate primera. Esta grilla acotada no satisface por sí sola todo el espacio de búsqueda por familia de §11.2; reportar limitación y ampliar si el presupuesto permite. No se oculta que el protocolo de entrenamiento difiere del original de cada paper.
- Piloto de cinco semillas comunes 6101–6105 después de búsqueda, comparables por semilla con los tres modelos base. No son el tamaño final. La guía §9.2 requiere piloto para dimensionar el final. Con cinco pares Wilcoxon bilateral no puede alcanzar p<0,05.
- Referencias: mayoría de train y regresión logística, cuatro valores C sobre representación fija de color/medias espaciales. StandardScaler solo train. Estas referencias deterministas no se replican artificialmente por semilla. Criterio de convergencia obligatorio; valorar ampliar descriptores como HOG para una referencia clásica más fuerte.
- Primero perfil de 12 pasos por modelo en un lote real: compatibilidad, memoria y extrapolación de tiempo. No sustituye prueba de sobreajuste de lote ni FLOPs ni evaluación científica. No se seleccionan candidatos por F1 del perfil.
- No se continúa al benchmark confirmatorio de forma automática después del piloto. Calcular R con δ=0,02, α=0,05 y familia de diez comparaciones contra ResNet50, Holm; usar Bonferroni para potencia conservadora, 80%. Conservar semillas pareadas, definir nuevas semillas si R>22 y extender baseline sin tocar E1.
- Reserva original ya consultada en E1: nunca llamarla holdout intacto para E2. El código no la evalúa. Las conclusiones de este bloque son exploratorias de desarrollo. Antes de prueba final resolver diseño externo/nuevo holdout por paciente o declarar explícitamente reutilización y limitación; no fabricar independencia.
- No mezclar ajustes y piloto en la media final. No excluir semillas con mala métrica. No usar reserva para escoger arquitectura, R, HP o umbrales.
- Guardado por época con modelo, optimizador, mejor modelo y estados RNG. Si se corta dentro de una época, se repite esa época. Carpetas separadas para perfil, búsqueda y piloto; archivos de configuración y entorno congelados; bloqueo de ejecución para impedir dos procesos escribiendo a la vez. No se elimina automáticamente un bloqueo dejado por cierre forzado.

## Selección bibliográfica pendiente antes del Top 10 definitivo

Estos diez son candidatos operativos, escogidos para diversidad y viabilidad, no una clasificación ya validada de los más frecuentes. Falta completar puntuación reproducible de citación normalizada, recencia, desempeño comparable en histopatología, disponibilidad y representatividad, y evaluar exclusiones. Dos variantes MobileViT comparten paper; justificar representatividad o sustituir antes del benchmark final. No se inventan citaciones ni puntuaciones ausentes. Replicación del paper, FLOPs, análisis por paciente/TSS, incertidumbre, calibración, robustez, errores, Grad-CAM y ablaciones siguen pendientes; no están resueltos por este cuaderno.

## Fuentes primarias de arquitectura

- He et al., Deep Residual Learning: https://arxiv.org/abs/1512.03385
- Xie et al., Aggregated Residual Transformations: https://arxiv.org/abs/1611.05431
- Howard et al., Searching for MobileNetV3: https://arxiv.org/abs/1905.02244
- Tan y Le, EfficientNet: https://arxiv.org/abs/1905.11946
- Touvron et al., DeiT: https://arxiv.org/abs/2012.12877
- Liu et al., Swin Transformer: https://arxiv.org/abs/2103.14030
- Mehta y Rastegari, MobileViT: https://arxiv.org/abs/2110.02178
- Tolstikhin et al., MLP-Mixer: https://arxiv.org/abs/2105.01601
- Touvron et al., ResMLP: https://arxiv.org/abs/2105.03404
- Biblioteca y API: https://huggingface.co/docs/timm/quickstart ; https://pypi.org/project/timm/1.0.22/

## Secuencia hasta el 14

11: exportar evidencias completas, perfil GPU, referencias simples y comenzar ajustes si cabe el presupuesto observado. 12: ajustes/piloto, extracción bibliográfica y EDA. 13: cerrar piloto, dimensionar R, decidir qué puede completarse y ejecutar análisis restantes por prioridad. 14: integrar resultados realmente disponibles al libro/dashboard E2 y empaquetar. No prometer que 90 corridas más análisis caben sin medir GPU; cinco semillas no satisfacen automáticamente potencia.
