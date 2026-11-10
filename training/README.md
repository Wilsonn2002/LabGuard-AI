# Entrenamiento del detector LabGuard

Este directorio prepara el ajuste fino de YOLOv8n para detectar personas y EPP en imagenes de laboratorios reales. El entrenamiento se ejecuta en Google Colab; WSL se usa para organizar, validar y probar el dataset.

## Clases

Los IDs son parte del contrato del dataset y no deben reordenarse despues de anotar:

| ID | Clase | Criterio de anotacion |
|---:|---|---|
| 0 | `person` | Caja de cada persona visible, incluyendo todo el cuerpo que aparezca en el encuadre. |
| 1 | `lab_coat` | Unicamente bata tradicional blanca de laboratorio; la caja sigue la prenda visible, no todo el torso. |
| 2 | `protective_gloves` | Una caja por guante visible. No etiquetar manos desnudas. |
| 3 | `safety_goggles` | Gafas de proteccion; no etiquetar gafas de uso diario. |
| 4 | `lab_cap` | Gorro de proteccion visible; no etiquetar cabello ni sombreros comunes. |
| 5 | `face_mask` | Tapabocas o respirador que cubra la zona facial correspondiente. |

Una caja se anota solo cuando el objeto es reconocible. Los objetos ausentes, totalmente ocluidos o ambiguos no reciben cajas. Todas las personas visibles deben etiquetarse aunque no usen EPP. Una imagen sin objetos validos necesita un archivo `.txt` vacio para dejar constancia de que fue revisada.

`lab_coat` no incluye trajes integrales de proteccion o cleanroom suits, overoles, uniformes medicos o quirurgicos ni batas de colores distintos del blanco. Esta politica se aplica durante la revision humana: no autoriza borrar cajas existentes de forma automatica.

## Organizacion

`training/dataset.yaml` espera esta estructura, ignorada por Git:

```text
data/labguard/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Cada imagen y etiqueta comparte ruta relativa y nombre base. Por ejemplo, `images/train/lab-a/001.jpg` corresponde a `labels/train/lab-a/001.txt`.

Cada linea de etiqueta usa el formato YOLO normalizado:

```text
class_id x_center y_center width height
```

Los cuatro valores geometricos deben estar entre 0 y 1, y la caja completa debe quedar dentro de la imagen. Se recomiendan imagenes JPEG o PNG de al menos 10 pixeles por dimension.

## Recoleccion real

1. Obtener autorizacion institucional y consentimiento informado antes de capturar personas identificables.
2. Registrar variedad de laboratorios, camaras, distancias, iluminacion, posturas, tonos de piel, tallas y modelos de EPP realmente usados.
3. Incluir escenas con EPP completo, incompleto, uso incorrecto, oclusiones parciales, varias personas y fondos sin personas.
4. Evitar secuencias casi identicas. Si se extraen cuadros de video, conservar pocos cuadros separados temporalmente.
5. Mantener fuera del repositorio las imagenes, identificadores, consentimientos y metadatos sensibles. Definir acceso, retencion y eliminacion.
6. No aplicar desenfoque sobre gafas, mascarillas o gorros: alteraria precisamente las caracteristicas que debe aprender el modelo. Priorizar sesiones controladas y consentidas.

La division debe realizarse por sesion, persona o laboratorio, no imagen por imagen. Ninguna persona ni rafaga de video debe aparecer en mas de un split. Como punto de partida, asignar aproximadamente 70 % de grupos a `train`, 15 % a `val` y 15 % a `test`, conservando diversidad y presencia de todas las clases en los tres conjuntos. `test` se mantiene cerrado hasta seleccionar el modelo final.

Antes de entrenar, revisar una muestra de etiquetas entre dos personas. Corregir cajas sueltas, clases confundidas, objetos omitidos y criterios inconsistentes. El conteo por clase del validador ayuda a detectar desbalances, pero no sustituye la inspeccion visual.

## Importacion provisional desde Roboflow

`import_roboflow.py` lee un ZIP sin modificarlo y escribe un staging separado bajo `.labguard-work/`, ignorado por Git. No mezcla archivos con `data/labguard/` ni cambia los splits originales.

```bash
python training/import_roboflow.py \
  --zip "/mnt/d/8vo Semestre/Fundamentos_CPD/dataset_LabGuard.zip"
```

El importador conserva los miembros originales, convierte anotaciones de deteccion y poligonos a cajas, registra procedencia, duplicados, variantes Roboflow y videos de origen. La union de `Safety_Glasses` y `goggles` en `safety_goggles` queda marcada como `PROVISIONAL` para revision humana. El dataset no contiene anotaciones `lab_cap` y el staging no se considera listo para entrenamiento.

Reportes principales:

```text
.labguard-work/roboflow_import/reports/import_report.md
.labguard-work/roboflow_import/reports/import_report.json
.labguard-work/roboflow_import/review/index.html
```

Para revisar las muestras:

```bash
python -m http.server 8000 --directory .labguard-work/roboflow_import/review
```

Luego abrir `http://localhost:8000`. Para repetir la importacion, usar otra ruta con `--output`; el script no sobrescribe un staging existente.

### Segundo dataset Roboflow

`Dataset2_LabGuard.zip` usa un perfil independiente y se importa sin modificar el primer staging:

```bash
python training/import_roboflow.py \
  --zip "/mnt/d/8vo Semestre/Fundamentos_CPD/Dataset2_LabGuard.zip" \
  --output .labguard-work/roboflow_import_2 \
  --profile labguard_dataset_2 \
  --reference-staging .labguard-work/roboflow_import
```

El perfil convierte `Blue Lab Coat`, `Full Lab Coat` y `Lab Coat` a `lab_coat`; `Gloves` a `protective_gloves`; `Hair Cap` a `lab_cap`; y `Mask` a `face_mask`. Toda conversion de las tres clases originales de batas del segundo dataset es provisional bajo la politica actual y debe revisarse visualmente: solo se conserva como `lab_coat` una bata tradicional blanca. `Glasses` pasa provisionalmente a `safety_goggles` y requiere confirmacion visual. Las clases `No glasses`, `No gloves`, `No hair cap`, `No lab coat`, `No mask` y `Shoe Cover` no se convierten a YOLO: se conservan en `provenance/excluded_annotations.jsonl` y aparecen como `EXCLUIDA` en la revision estatica.

El dataset no contiene cajas `person`; el importador no las inventa. Conserva los splits originales, registra variantes y secuencias entre splits, compara imagenes con el primer staging y no elimina automaticamente batas fusionadas o superpuestas. Los pequenos excesos de coordenadas producidos por redondeo de Roboflow se recortan solo dentro de la tolerancia del perfil y quedan auditados en `provenance/boundary_adjustments.jsonl` con su geometria original y convertida.

Para abrir una copia editable completamente separada:

```bash
python training/review_annotations.py \
  --source .labguard-work/roboflow_import_2 \
  --work .labguard-work/annotation_review_2 \
  --open-browser
```

El primer workspace `.labguard-work/annotation_review/` no se reinicializa ni se modifica. Antes de usar Dataset2 para entrenamiento se deben revisar las 158 cajas provisionales de `Glasses`, confirmar la calidad de los gorros y batas, y resolver las variantes o secuencias relacionadas que atraviesan splits.

El editor abre por defecto una vista comparativa con la fotografia limpia a la izquierda y el canvas editable a la derecha. Los controles **Comparativa** e **Individual anotada** cambian solamente la presentacion; **Mostrar nombres de clases** oculta o muestra los rotulos sin ocultar las cajas ni modificar anotaciones. La clase de una caja seleccionada siempre permanece disponible en el inspector.

### Editor local de anotaciones

El editor crea `.labguard-work/annotation_review/` como copia independiente del staging. Permite crear, mover, redimensionar, reclasificar y eliminar cajas sin modificar el ZIP, el staging auditado ni `data/labguard/`:

```bash
python training/review_annotations.py --open-browser
```

La interfaz muestra la clase Roboflow original de cada caja y distingue las gafas provenientes de `Safety_Glasses` y `goggles`. Cada guardado conserva un historial before/after en `history/corrections.jsonl`. El boton de exportacion genera snapshots versionados bajo `exports/` con etiquetas YOLO, manifiesto y conteos por clase; no redistribuye imagenes ni cambia los splits.

Para preparar la copia sin iniciar el servidor:

```bash
python training/review_annotations.py --prepare-only
```

La copia existente se reutiliza en ejecuciones posteriores. Para reiniciar desde el staging se debe elegir otra ruta con `--work`; el editor no sobrescribe copias existentes de forma automatica.

#### Auditoria de consistencia

El boton **Auditar anotaciones** compara los estados actuales sin cambiar etiquetas. Agrupa variantes aumentadas por frame canonico y secuencias por video de origen; senala diferencias de clases o conteos entre variantes, y usa ambos frames vecinos cercanos como evidencia temporal conservadora. Los filtros permiten limitar por prioridad, clase, tipo de relacion, split y texto.

Los hallazgos son sugerencias para inspeccion humana, no detecciones: una clase ausente no demuestra que el EPP este ausente. **Revisar y decidir en el editor** abre la imagen correspondiente, pero no agrega, mueve ni copia cajas. Esto es especialmente importante entre variantes con recortes, transformaciones o diferencias visuales. Solo un guardado manual modifica la copia de trabajo; los splits permanecen intactos.

## Validacion en WSL

Desde la raiz del repositorio:

```bash
python training/validate_dataset.py
```

El comando falla si cambia el orden de clases, falta algun split, una imagen no abre, falta su etiqueta, hay etiquetas huerfanas, IDs invalidos, campos no numericos, duplicados o cajas fuera de rango. Para una revision estructural rapida sin decodificar imagenes:

```bash
python training/validate_dataset.py --skip-image-check
```

El siguiente comando valida el dataset y muestra la configuracion, pero no entrena porque no incluye `--execute`:

```bash
python training/train.py --model yolov8n.pt
```

No usar la GPU MX230 para este ajuste fino. Mantener los pesos base, datasets y resultados fuera de Git.

## Entrenamiento en Colab

1. Crear un runtime con GPU desde `Runtime > Change runtime type > T4 GPU`.
2. Guardar en Google Drive un ZIP cuyo contenido raiz sea `images/` y `labels/`, y una copia de `yolov8n.pt`.
3. Clonar el repositorio y montar Drive. Sustituir `<URL_DEL_REPOSITORIO>` y las rutas de Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!git clone <URL_DEL_REPOSITORIO> /content/LabGuard-AI
%cd /content/LabGuard-AI
!pip install -q ultralytics==8.4.208 PyYAML opencv-python-headless
!mkdir -p data/labguard
!unzip -q "/content/drive/MyDrive/LabGuard-AI/labguard-yolo.zip" -d data/labguard
!cp "/content/drive/MyDrive/LabGuard-AI/yolov8n.pt" ./yolov8n.pt
```

4. Validar antes de reservar tiempo de GPU:

```bash
!python training/validate_dataset.py
!python training/train.py --model yolov8n.pt
```

5. Entrenar y escribir checkpoints directamente en Drive:

```bash
!python training/train.py \
  --model yolov8n.pt \
  --epochs 100 \
  --patience 20 \
  --image-size 640 \
  --batch 16 \
  --device 0 \
  --project "/content/drive/MyDrive/LabGuard-AI/runs" \
  --name labguard_yolov8n_v1 \
  --execute
```

Si Colab agota memoria, reducir primero `--batch` a 8 o 4. Para continuar una ejecucion interrumpida, indicar su `last.pt` y agregar `--resume --execute`.

## Evaluacion

Evaluar una sola vez sobre `test` despues de elegir hiperparametros con `val`:

```bash
!python training/evaluate.py \
  --model "/content/drive/MyDrive/LabGuard-AI/runs/labguard_yolov8n_v1/weights/best.pt" \
  --split test \
  --device 0 \
  --plots \
  --output "/content/drive/MyDrive/LabGuard-AI/runs/labguard_yolov8n_v1/test_metrics.json"
```

El reporte incluye precision, recall, mAP50 y mAP50-95 globales y por clase. Ademas deben inspeccionarse falsos positivos y negativos por laboratorio, iluminacion y grupo de personas antes de reemplazar `models/best.pt` o integrar el modelo en Streamlit.
