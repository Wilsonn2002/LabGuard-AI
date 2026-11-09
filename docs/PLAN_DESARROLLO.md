# Plan de desarrollo

La prioridad es mejorar evidencia y reproducibilidad antes de añadir más automatización. Cada mejora debe incluir datos autorizados, criterios de aceptación, pruebas y revisión de privacidad.

## Prioridad 0: línea base reproducible

1. Fijar versiones compatibles de Python, PyTorch, Ultralytics, OpenCV y Streamlit.
2. Definir una licencia para el repositorio.
3. Automatizar la verificación SHA-256 antes de cargar modelos.
4. Crear un conjunto de validación local, anonimizado y separado por persona/sesión/laboratorio.
5. Registrar precision, recall, mAP y matriz de errores por clase.
6. Añadir una prueba manual documentada del flujo Streamlit con ambos modelos.

Criterio de salida: entorno reproducible, modelo identificable por hash y métricas base aprobadas.

## Gafas de seguridad

- Etiquetar gafas de protección reales de varios diseños, transparencias y ángulos.
- Incluir negativos difíciles: gafas graduadas, gafas de sol, reflejos y protectores faciales.
- Revisar objetos pequeños a resoluciones 512 y 640.
- Evaluar por distancia al rostro e iluminación.
- Considerar asociación con región ocular solo después de integrar landmarks fiables.

Criterio de salida: mejora medida frente al checkpoint actual y tasa de confusión con gafas comunes documentada.

## Guantes

- Mantener una caja por guante visible y documentar oclusiones.
- Aumentar variedad de colores, materiales, tamaños y fondos.
- Incluir manos desnudas, mangas y herramientas como negativos difíciles.
- Evaluar detección de objetos pequeños y pares incompletos.
- Investigar pose de manos para asociar cada guante con una persona sin asumir que estar en el recorte basta.

Criterio de salida: métricas por guante y pruebas específicas de manos desnudas sin falsos positivos críticos.

## Batas de laboratorio

- No reutilizar `Vest`: una prenda reflectiva industrial no representa una bata.
- Recolectar batas abiertas/cerradas, colores claros/oscuros y vistas parciales.
- Incluir camisas, chaquetas, delantales y uniformes como negativos.
- Definir si se detecta la prenda visible o el estado de uso correcto; son problemas distintos.
- Integrar la nueva clase `lab_coat` y actualizar `class_mapping`.

Criterio de salida: bata evaluable en UI sin mapear clases semánticamente incorrectas.

## Gorros de laboratorio

- No reutilizar `helmet`: casco industrial y gorro de laboratorio son clases distintas.
- Recolectar cofias, gorros desechables y gorros textiles autorizados.
- Incluir cabello, capuchas, sombreros y cascos como negativos.
- Evaluar oclusiones por ángulo superior y baja resolución.
- Integrar `lab_cap` solo después de validar su contrato y hash.

Criterio de salida: clase dedicada con errores frente a cabello y otras prendas documentados.

## Rostros y advertencias

Las primitivas de advertencia ya existen, pero no deben conectarse sin evidencia facial fiable.

1. Evaluar un detector de rostro o landmarks que funcione con tapabocas y gafas.
2. Localizar boca/rostro con confianza y asociar la región a una persona.
3. Definir evidencia explícita de ausencia de tapabocas; la falta de una detección positiva no es suficiente.
4. Mostrar advertencias solo al superar umbrales de rostro y ausencia.
5. Mantener el estado `No determinado` para oclusión o baja calidad.
6. Probar que los iconos no cubran EPP ni identifiquen erróneamente a otra persona.
7. Realizar evaluación de impacto de privacidad antes de almacenar landmarks o recortes faciales.

Criterio de salida: pruebas automatizadas y validación humana que demuestren ubicación facial, asociación y evidencia de ausencia.

## Asociación por persona

- Medir errores de recortes superpuestos y deduplicación.
- Añadir reglas espaciales por parte corporal o pose.
- Conservar trazabilidad entre persona, detección y fuente.
- Evitar convertir IDs temporales en identidades persistentes.
- Evaluar rendimiento con muchas personas antes de aumentar `maximum_people`.

Criterio de salida: conjunto de casos multi-persona con asignación revisada y métricas de asociación.

## Video y cámara

`VideoAnalyzer` todavía no está implementado. El desarrollo debe realizarse por etapas:

1. Procesar un iterable de fotogramas sin interfaz.
2. Añadir límites de FPS, resolución y memoria.
3. Incorporar tracking anónimo para estabilizar IDs, sin reconocimiento facial.
4. Agregar pruebas con video sintético o autorizado.
5. Integrar archivos de video y después `st.camera_input` o captura controlada.
6. Mostrar latencia y estados temporales sin afirmar incumplimiento por un fotograma.
7. Definir retención cero por defecto y controles explícitos antes de guardar video.

Criterio de salida: procesamiento acotado, cancelable, probado y con política de privacidad aprobada.

## Integración del modelo especializado

El orden del dataset es fijo:

```text
0 person
1 lab_coat
2 protective_gloves
3 safety_goggles
4 lab_cap
5 face_mask
```

Para cada candidato:

1. Validar el dataset con `training/validate_dataset.py`.
2. Entrenar en Colab sin sobrescribir pesos existentes.
3. Seleccionar hiperparámetros solo con `val`.
4. Evaluar una vez en `test` con `training/evaluate.py`.
5. Revisar errores por laboratorio, persona e iluminación.
6. Guardar fuente, parámetros, métricas y SHA-256.
7. Actualizar `configs/model.yaml` para los nombres nuevos.
8. Ejecutar pruebas y comparación manual antes de promover el checkpoint.

## Entregables futuros sugeridos

- Registro de modelos con versión, hash, clases y métricas.
- Dataset card sin datos personales y con licencia/procedencia.
- Matriz de trazabilidad entre requisito, prueba y evidencia.
- Reporte JSON de análisis sin imagen por defecto.
- CI para pruebas, revisión de secretos y validación documental.
- Guía de respuesta ante exposición de fotografías o credenciales.
