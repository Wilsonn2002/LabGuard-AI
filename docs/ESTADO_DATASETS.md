# Estado tecnico de datasets LabGuard AI

Fecha de corte: 2026-11-10 UTC. Rama inspeccionada: `main`.

Este informe documenta las exportaciones YOLO existentes. Una exportacion
integra no equivale a una validacion semantica completa: ambos datasets siguen
en revision humana y no se declaran listos para entrenamiento.

## Contrato de clases

| ID | Clase | Criterio actual |
|---:|---|---|
| 0 | `person` | Cada persona visible, con caja sobre la porcion corporal observable. |
| 1 | `lab_coat` | Solo bata tradicional blanca de laboratorio; no incluye cleanroom suits, overoles, uniformes medicos o batas de otros colores. |
| 2 | `protective_gloves` | Un objeto por guante de proteccion visible; no incluye manos desnudas. |
| 3 | `safety_goggles` | Gafas de proteccion; no incluye gafas de uso diario. |
| 4 | `lab_cap` | Gorro de proteccion visible; no incluye cabello ni sombreros comunes. |
| 5 | `face_mask` | Tapabocas o respirador visible sobre la zona facial. |

Los IDs forman parte del formato YOLO y no deben reordenarse.

## Dataset 1

- Workspace: `.labguard-work/annotation_review/`.
- Imagenes y archivos de etiqueta exportados: 2.223.
- Exportacion mas reciente: `20261110T033741Z-e878480c`.
- Fecha de exportacion: `2026-11-10T03:37:43.616127+00:00`.
- Objetos YOLO exportados: 6.189.
- Etiquetas vacias validas: 32.
- Distribucion de archivos: train 2.103, val 70, test 50.
- Distribucion de objetos: train 5.852, val 199, test 138.

| ID | Clase | Objetos exportados |
|---:|---|---:|
| 0 | `person` | 1.467 |
| 1 | `lab_coat` | 1.290 |
| 2 | `protective_gloves` | 2.427 |
| 3 | `safety_goggles` | 454 |
| 4 | `lab_cap` | 3 |
| 5 | `face_mask` | 548 |

La integridad de los 2.223 archivos coincide con los SHA-256 del manifiesto,
los conteos recalculados coinciden con `class_counts` y no se encontraron
lineas YOLO mal formadas. La exportacion tambien coincide byte a byte con las
etiquetas actuales del workspace.

El historial registra 43 guardados sobre 36 imagenes: 56 cajas creadas, 15
actualizadas y 1 eliminada. Quedan 55 cajas de origen manual. Las otras 2.187
imagenes tienen revision 0; por tanto, no deben describirse como revisadas
individualmente solo por estar exportadas.

Permanecen 441 anotaciones con origen provisional Roboflow: 15 procedentes de
`Safety_Glasses` y 426 de `goggles`. Actualmente 440 conservan la clase
`safety_goggles` y una fue reclasificada a `person`, pero mantiene su origen
provisional y requiere comprobacion. La presencia de solo 3 `lab_cap` tambien
indica cobertura limitada, no ausencia real de gorros.

El staging registra cinco grupos de video presentes en mas de un split. Los
splits no fueron cambiados durante la revision ni la exportacion.

## Dataset 2

- Workspace: `.labguard-work/annotation_review_2/`.
- Imagenes y archivos de etiqueta exportados: 370.
- Exportacion mas reciente: `20261110T033453Z-9d5f043c`.
- Fecha de exportacion: `2026-11-10T03:34:54.157741+00:00`.
- Objetos YOLO exportados: 1.670.
- Etiquetas vacias validas: 36.
- Distribucion de archivos: train 180, val 95, test 95.
- Distribucion de objetos: train 691, val 359, test 620.

| ID | Clase | Objetos exportados |
|---:|---|---:|
| 0 | `person` | 0 |
| 1 | `lab_coat` | 502 |
| 2 | `protective_gloves` | 226 |
| 3 | `safety_goggles` | 62 |
| 4 | `lab_cap` | 497 |
| 5 | `face_mask` | 383 |

La integridad de los 370 archivos coincide con los SHA-256 del manifiesto,
los conteos recalculados coinciden con `class_counts`, incluyendo el cero de
`person`, y no se encontraron lineas YOLO mal formadas. La exportacion coincide
byte a byte con las etiquetas actuales del workspace.

El historial registra 94 guardados sobre 92 imagenes: 96 cajas eliminadas y 2
actualizadas, sin cajas nuevas. Las otras 278 imagenes tienen revision 0. Las
62 cajas restantes provenientes de `Glasses` conservan marca provisional y
estan en imagenes sin revision guardada; no deben presentarse como gafas de
proteccion confirmadas.

Dataset 2 no contiene anotaciones `person`; no se inventaron cajas. Sus 502
cajas `lab_coat` provienen de la conversion de `Blue Lab Coat`, `Full Lab Coat`
y `Lab Coat`. Bajo el criterio actual, todas requieren revision para conservar
solo batas tradicionales blancas. La revision visual de guantes, gorros y
mascaras tambien permanece incompleta.

El staging conserva 1.036 cajas de clases negativas o fuera del esquema en su
auditoria, pero no las incorpora a YOLO. Registra un grupo de variantes
Roboflow y una posible secuencia nominal presentes en mas de un split. No hubo
redistribucion automatica.

## Que esta verificado

- El formato, los IDs, la cantidad de archivos y los conteos por clase de las
  dos exportaciones.
- Los SHA-256 individuales de todos los archivos de etiqueta declarados en
  cada manifiesto.
- La correspondencia entre cada exportacion y el estado actual de su
  workspace.
- La trazabilidad de los guardados mediante el hash del historial incluido en
  cada manifiesto.
- Las correcciones humanas descritas por los historiales; no se extrapolan a
  imagenes sin revision.

## Que sigue provisional o pendiente

- 441 anotaciones con origen provisional en Dataset 1.
- 62 anotaciones `Glasses` provisionales en Dataset 2.
- Todas las conversiones de batas del Dataset 2 bajo la politica de bata
  tradicional blanca.
- Dataset 2 completo respecto a la clase `person`, que tiene cero cajas.
- Revision visual exhaustiva de gorros, guantes, mascaras y posibles objetos
  omitidos en ambos datasets.
- Resolucion por grupos de videos, variantes o secuencias relacionadas entre
  splits antes de cualquier entrenamiento definitivo.

## Respaldo externo

- Ruta: `~/LabGuard-Backups/LabGuard_Anotaciones_Final.tar.gz`.
- Tamano observado: 79 MiB.
- SHA-256: `8b4fa4f6634d580ddbbeb0dddd1d82163d67a1d5db5b0fd51b34f8799d5dfce1`.

El paquete ligero de entrega no duplica este respaldo, las imagenes, los
archivos de etiqueta ni modelos `.pt`. Incluye este informe, un resumen JSON,
los dos manifiestos completos y sus archivos `dataset.yaml` como evidencia.
