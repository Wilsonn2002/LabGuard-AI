# Modelos, rutas e integridad

La aplicación no descarga modelos automáticamente. Los archivos `.pt` están ignorados por Git y deben obtenerse de las fuentes y revisiones documentadas después de clonar. `yolov8n.pt` proviene de Ultralytics; `best.pt` es un checkpoint de un publicador tercero en Hugging Face.

## Advertencia de seguridad

Los checkpoints de PyTorch pueden contener objetos serializados y deben tratarse como código potencialmente ejecutable. Descarga solo desde fuentes verificadas, comprueba SHA-256 antes de abrir el archivo y no cargues pesos recibidos por correo, mensajería o enlaces desconocidos.

Los hashes siguientes corresponden a los archivos auditados en el commit `1929ba4`. Si el proveedor reemplaza un archivo, detén la instalación y revisa el cambio en un pull request; no actualices el hash sin validar clases, métricas, licencia y comportamiento.

## Resumen

| Archivo | Función | Ruta | Tamaño auditado | SHA-256 conocido |
|---|---|---|---:|---|
| `yolov8n.pt` | Detector COCO de personas | `./yolov8n.pt` | 6,549,796 bytes | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| `best.pt` | Detector EPP industrial | `./models/best.pt` | 5,625,014 bytes | `07172ef3ae9e256c40a1fb0ce3eefe5547d90170645aa73dded0fffc382cdb31` |

## `yolov8n.pt`: personas

- Arquitectura: Ultralytics YOLOv8n, tarea de detección.
- Fuente usada para reproducción: release `v8.3.0` de `ultralytics/assets`.
- URL: <https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt>
- Metadato interno del checkpoint auditado: Ultralytics `8.0.0.dev0`, fecha `2022-12-30`.
- Clases: 80 clases COCO. LabGuard solicita únicamente el ID `0`, `person`, mediante `classes=[0]`.
- Ruta requerida: `yolov8n.pt` en la raíz.

Descarga y verificación:

```bash
(
  set -eu
  tmp_file="$(mktemp yolov8n.pt.download.XXXXXX)"
  cleanup() { [ -z "${tmp_file:-}" ] || rm -f -- "$tmp_file"; }
  trap cleanup EXIT
  trap 'exit 130' HUP INT TERM

  curl --fail --location --output "$tmp_file" \
    https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt
  printf '%s  %s\n' \
    'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36' \
    "$tmp_file" | sha256sum --check
  mv "$tmp_file" yolov8n.pt
  tmp_file=""
)
```

Los identificadores `Persona 1`, `Persona 2`, etc. se generan temporalmente de izquierda a derecha para cada fotografía. No son identidades biométricas ni tracking persistente.

## `models/best.pt`: EPP actual

- Repositorio: <https://huggingface.co/Tanishjain9/yolov8n-ppe-detection-6classes>
- Revisión fijada: `a2c1a97646ade96e48adda74781dd455d0456c40`.
- Archivo: `best.pt`.
- Arquitectura: YOLOv8n, tarea de detección.
- Metadato interno auditado: Ultralytics `8.3.0`, fecha `2025-11-25`.
- Entrada indicada por la tarjeta: 640 x 640.
- Ruta requerida: `models/best.pt`.

Clases del checkpoint:

| ID | Clase fuente | Uso actual en LabGuard |
|---:|---|---|
| 0 | `Gloves` | Se mapea a `protective_gloves` |
| 1 | `Vest` | Se ignora; no equivale a bata de laboratorio |
| 2 | `goggles` | Se mapea a `safety_goggles` |
| 3 | `helmet` | Se ignora; no equivale a gorro de laboratorio |
| 4 | `mask` | Se mapea a `face_mask` |
| 5 | `safety_shoe` | Se ignora |

Descarga reproducible desde la revisión fijada:

```bash
mkdir -p models
(
  set -eu
  tmp_file="$(mktemp models/best.pt.download.XXXXXX)"
  cleanup() { [ -z "${tmp_file:-}" ] || rm -f -- "$tmp_file"; }
  trap cleanup EXIT
  trap 'exit 130' HUP INT TERM

  curl --fail --location --output "$tmp_file" \
    'https://huggingface.co/Tanishjain9/yolov8n-ppe-detection-6classes/resolve/a2c1a97646ade96e48adda74781dd455d0456c40/best.pt?download=true'
  printf '%s  %s\n' \
    '07172ef3ae9e256c40a1fb0ce3eefe5547d90170645aa73dded0fffc382cdb31' \
    "$tmp_file" | sha256sum --check
  mv "$tmp_file" models/best.pt
  tmp_file=""
)
```

El hash también está registrado en `configs/model.yaml`, pero el código actual no lo verifica durante el arranque. La comprobación manual sigue siendo obligatoria.

## Contrato del futuro modelo LabGuard

`training/dataset.yaml` define un contrato distinto para fine-tuning:

| ID | Clase de entrenamiento |
|---:|---|
| 0 | `person` |
| 1 | `lab_coat` |
| 2 | `protective_gloves` |
| 3 | `safety_goggles` |
| 4 | `lab_cap` |
| 5 | `face_mask` |

Un checkpoint producido por este pipeline no es intercambiable directamente con el `best.pt` actual. Antes de integrarlo se debe:

1. Evaluarlo con `training/evaluate.py`.
2. Copiarlo a una ruta de versión nueva, sin sobrescribir el modelo validado.
3. Actualizar `configs/model.yaml` para mapear los nuevos nombres o permitir el mapeo por defecto.
4. Actualizar revisión, SHA-256 y documentación.
5. Ejecutar las 39 pruebas y una validación manual con imágenes autorizadas.

## Versiones de software

- `pyproject.toml` no fija versiones de Ultralytics ni PyTorch.
- El entorno auditado utilizó Ultralytics `8.4.208`.
- La aplicación ejecuta en CPU porque `load_model_configuration` rechaza otros dispositivos.
- El script de entrenamiento acepta un dispositivo de Ultralytics y está pensado para Google Colab; no cambia el modo CPU de la aplicación.

## Licencias

- La tarjeta de Hugging Face de `Tanishjain9/yolov8n-ppe-detection-6classes` declara MIT.
- El propio checkpoint `models/best.pt` auditado contiene el metadato `license: AGPL-3.0 (https://ultralytics.com/license)`. Esta declaración entra en conflicto con la etiqueta MIT del publicador.
- Ultralytics publica su software y modelos bajo AGPL-3.0 o una licencia Enterprise. Consulta <https://www.ultralytics.com/license> y <https://github.com/ultralytics/ultralytics/blob/main/LICENSE>.
- Por prudencia, trata ambos checkpoints YOLO como sujetos a AGPL-3.0 o a una licencia Enterprise válida hasta obtener una aclaración jurídica documentada. No asumas que la etiqueta MIT elimina las obligaciones incorporadas por Ultralytics.
- El repositorio LabGuard AI no contiene un archivo `LICENSE` en el commit auditado.
- La tarjeta del modelo menciona un dataset personalizado, pero esta entrega no incluye ese dataset ni establece derechos para reutilizar sus imágenes.

Antes de una distribución pública o uso comercial, confirma con los responsables institucionales que las licencias del código, pesos y datos son compatibles. Esta sección documenta fuentes conocidas y no constituye asesoría legal.

## No subir a Git

`*.pt`, `*.pth`, `*.onnx` y `*.engine` están ignorados. No fuerces su inclusión con `git add -f`. Comparte instrucciones, URL, revisión, tamaño y hash, no el binario. Tampoco subas imágenes privadas usadas para comprobar el modelo.
