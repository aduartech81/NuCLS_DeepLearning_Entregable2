# Problema y preguntas de investigación

La entrada es un recorte RGB de 128×128 píxeles y la salida una de tres etiquetas: tumoral no mitótica, linfocitaria o estromal. La pérdida de entrenamiento es la entropía cruzada y la métrica primaria heredada es macro-F1. Se conserva ResNet50 como referencia y δ=0,02 como margen de relevancia práctica. La unidad clínica de independencia es el paciente; las semillas describen variabilidad del entrenamiento sobre una partición fija.

## Preguntas y decisiones

1. ¿Alguno de los diez candidatos mejora ResNet50 por un margen relevante? Estimar Δ macro-F1, intervalo y efecto pareado; familia de diez contrastes, Holm α=0,05. Distinguir evidencia de Δ>0 de evidencia de Δ>δ: el segundo criterio exige que el límite inferior apropiado supere 0,02. Las comparaciones actuales de validación son exploratorias porque se usan para selección.
2. ¿La ganancia se mantiene entre pacientes y sitios? Evaluar diferencias mediante remuestreo por paciente, conservando juntas sus imágenes; describir TSS y sensibilidad al retirar grupos. No tratar semillas o recortes como nuevos pacientes. Hipótesis de igualdad de rendimiento entre grupos como análisis exploratorio con intervalos y tamaños de grupo; no seleccionar el contraste más favorable.
3. ¿Cuánto rendimiento se pierde ante perturbaciones de imagen? Medir caída respecto a imagen limpia para tres severidades predefinidas, conservando pares de imágenes. Estimar por paciente la diferencia de degradación frente al baseline. El análisis de costo y degradación es exploratorio y no altera retroactivamente el Top 10.

Estas preguntas se documentan después del comienzo de la búsqueda exploratoria. No se presentan como preregistro anterior a todo entrenamiento. Un bloque confirmatorio requiere diseño congelado y resolver el acceso previo a la reserva.

## Formulación y contraste explícito de las preguntas

Sea $f_\theta: \mathbb{R}^{128\times128\times3}\rightarrow\Delta^2$ el clasificador y $L$ la entropía cruzada. Se minimiza el riesgo empírico de entrenamiento; el riesgo esperado $R(f)=\mathbb{E}_{(X,Y)\sim P}[L(f(X),Y)]$ no se identifica mediante el máximo de validación usado para seleccionar hiperparámetros.

| Pregunta | Hipótesis y unidad | Regla de interpretación |
|---|---|---|
| RQ1: mejora frente a ResNet50 | Para cada candidato, H0: Δ ≤ 0 frente a H1: Δ > 0; relevancia: H0: Δ ≤ 0,02 frente a H1: Δ > 0,02. Δ es la diferencia de macro-F1. Semillas emparejadas en desarrollo; remuestreo clínico por paciente por separado. | Diez comparaciones por familia, α=0,05, ajuste Holm; distinguir mejora estadística de mejora relevante. Cualquier contraste direccional añadido ahora es exploratorio. Mantener resultados bilaterales existentes sin cambiarlos para obtener significancia. |
| RQ2: heterogeneidad por sitio | H0: ausencia de diferencias de rendimiento entre TSS; H1: al menos un grupo difiere. Paciente como unidad, TSS como estrato. | Intervalos agrupados, tamaños de grupo y corrección de contrastes exploratorios. No confundir diferencias de composición con mecanismo causal. |
| RQ3: degradación ante perturbaciones | H0: diferencia de degradación candidato–baseline = 0; H1: distinta de 0, conservando pares limpio/perturbado por paciente. | Informar magnitud e IC por severidad y corregir la familia de contrastes. Sin margen de robustez previamente justificado no afirmar relevancia clínica. |

Esta formalización es formalización del protocolo fechada el 14 de noviembre, posterior al comienzo experimental. No sustituye un preregistro previo. δ=0,02 es el umbral operativo heredado; su justificación clínica externa no está establecida. El margen de sobreajuste δg no se deduce automáticamente de δ y permanece por justificar antes de nuevos contrastes definitivos.
