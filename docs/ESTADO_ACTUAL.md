# Estado actual

Estado auditado desde el commit `1929ba4`.

## Funcionalidades verificadas

| Funcionalidad | Estado | Evidencia técnica |
|---|---|---|
| Arranque de Streamlit | Implementada | `app.py`, prueba de humo |
| Carga JPG/JPEG/PNG | Implementada | `decode_uploaded_image` |
| Inferencia EPP local | Implementada | `YoloPpeDetector` |
| Guantes | Evaluable con modelo actual | `Gloves` se mapea a `protective_gloves` |
| Gafas de seguridad | Evaluable con modelo actual | `goggles` se mapea a `safety_goggles` |
| Tapabocas | Evaluable con modelo actual | `mask` se mapea a `face_mask` |
| Bata de laboratorio | No evaluable | `Vest` se ignora y no equivale a bata |
| Gorro de laboratorio | No evaluable | `helmet` se ignora y no equivale a gorro |
| Análisis global | Implementado | Una inferencia EPP sobre la imagen |
| Análisis por persona | Implementado con asociación heurística | YOLO COCO, recortes y deduplicación |
| Cajas, etiquetas y conteos | Implementados | OpenCV y utilidades UI |
| Ajustes de confianza y tamaño | Implementados | Controles Streamlit |
| Advertencias faciales | No implementadas | Toggle informativo sin evidencia conectada |
| Cámara y video | No implementados | `VideoAnalyzer` lanza `NotImplementedError` |
| Reportes de análisis | No implementados | Solo existe `ReportGenerator` como protocolo |
| Validación de dataset | Implementada | `training/validate_dataset.py` |
| Entrenamiento | Script preparado, no ejecutado en esta entrega | Requiere dataset y `--execute` |
| Evaluación de checkpoint nuevo | Script preparado | Requiere dataset y contrato de seis clases |

## Pruebas

La suite contiene 39 pruebas en 10 archivos. En esta entrega se ejecuta con:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

Resultado verificado durante la preparación de la entrega: `Ran 39 tests` y `OK`.

Cobertura funcional aproximada:

| Área | Cantidad |
|---|---:|
| Modelos de dominio | 2 |
| Reglas de cumplimiento | 2 |
| Configuración | 3 |
| Detector EPP | 5 |
| Detector de personas | 1 |
| Análisis por persona | 3 |
| Utilidades OpenCV/UI | 11 |
| Construcción y caché UI | 4 |
| Arranque Streamlit | 1 |
| Dataset y métricas | 7 |
| **Total** | **39** |

Las pruebas usan modelos falsos y datasets temporales. No ejecutan inferencia real con los pesos, no miden precisión y no entrenan.

## Alcance real de los modelos

El checkpoint EPP contiene seis clases industriales, pero LabGuard solo acepta tres. `Vest`, `helmet` y `safety_shoe` se descartan para no presentarlos incorrectamente como bata, gorro o EPP requerido.

El análisis por persona usa la clase COCO `person`. Los IDs duran una fotografía y no representan reconocimiento facial.

El pipeline de entrenamiento define seis clases nuevas: `person`, `lab_coat`, `protective_gloves`, `safety_goggles`, `lab_cap` y `face_mask`. Ese contrato no coincide con los nombres del checkpoint de producción actual.

## Limitaciones técnicas

- Solo se procesa una imagen estática por carga.
- La aplicación está restringida a CPU desde la configuración.
- El modo individual ejecuta una inferencia adicional por persona y puede ser lento.
- La asignación EPP-persona depende del recorte y de una deduplicación geométrica.
- Detectar un objeto no demuestra que esté correctamente puesto.
- No detectar un objeto no demuestra que esté ausente.
- No hay estimación de pose, manos, rostro, boca ni calidad de uso.
- No hay seguimiento temporal, cámara ni archivo de video.
- No hay almacenamiento, autenticación, exportación ni registro de auditoría.
- `configs/app.yaml` y `configs/compliance.yaml` no afectan la ejecución.
- El hash configurado del modelo EPP no se verifica automáticamente.
- Las dependencias no están fijadas; una actualización puede cambiar comportamiento.
- El dataset de entrenamiento no está incluido.
- Un modelo nuevo requiere actualización de `class_mapping`, no solo reemplazar `best.pt`.

## Falsos positivos y falsos negativos

No existe todavía un conjunto de errores versionado ni una métrica local validada en laboratorios universitarios. Por tanto, no se debe afirmar una tasa de falsos positivos. Los siguientes son riesgos que deben probarse y registrarse:

| Clase | Riesgo de falso positivo | Riesgo de falso negativo |
|---|---|---|
| Gafas | Gafas comunes, reflejos, monturas u objetos cerca de los ojos | Gafas transparentes, pequeñas, laterales u ocluidas |
| Guantes | Mangas, manos, herramientas u objetos de color parecido | Guantes oscuros, manos pequeñas, oclusión o baja resolución |
| Tapabocas | Barba, cuello de ropa, sombras u objetos frente a la cara | Mascarilla lateral, parcialmente cubierta o rostro distante |
| Persona | Figuras, carteles o reflejos con forma humana | Personas parciales, pequeñas, ocluidas o con baja iluminación |

En modo global, una detección correcta puede producir una conclusión engañosa: un solo guante visible representa la categoría para toda la imagen. En modo individual, recortes superpuestos pueden asignar un objeto a la persona equivocada.

Reducir el umbral de confianza recupera más candidatos y normalmente aumenta falsos positivos. Aumentarlo puede ocultar objetos reales. El slider no sustituye una calibración sobre un dataset local.

## Riesgos de privacidad

- Las fotografías pueden contener rostros, nombres, pantallas, matrículas, documentos o ubicaciones identificables.
- El procesamiento es local, pero Streamlit conserva datos en memoria durante la sesión.
- No hay borrado seguro, control de acceso ni política automática de retención.
- Las imágenes no deben subirse a GitHub, issues, pull requests ni conversaciones con asistentes.
- Las pruebas con personas requieren consentimiento y autorización institucional.

## Criterio de interpretación

Los textos `Posiblemente no visibles` y `No determinado` son deliberados. Significan que el modelo no confirmó un objeto, no que una persona incumpla. Toda decisión de seguridad debe tener revisión humana.
