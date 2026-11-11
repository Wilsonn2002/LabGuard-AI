# LabGuard AI

LabGuard AI es un prototipo académico de visión artificial para revisar elementos de protección personal (EPP) visibles en fotografías de laboratorios. La aplicación procesa imágenes localmente, muestra detecciones orientativas y mantiene explícita la necesidad de supervisión humana.

Esta documentación corresponde al estado auditado del commit `1929ba4`.

## Objetivos

- Detectar EPP visible en fotografías mediante modelos YOLOv8 locales.
- Presentar resultados comprensibles sin afirmar incumplimientos cuando la evidencia es insuficiente.
- Comparar un análisis global con un análisis opcional por persona.
- Preparar un flujo reproducible para validar datos, ajustar YOLOv8n y evaluar nuevos checkpoints.
- Servir como base académica para mejorar detección, privacidad y trazabilidad.

LabGuard AI no reemplaza una inspección profesional, no determina por sí solo el uso correcto del EPP y no debe utilizarse para sanciones automáticas.

## Estado funcional

Actualmente está implementado:

- Carga de una fotografía JPG, JPEG o PNG desde Streamlit.
- Inferencia local y en CPU con `models/best.pt`.
- Detección efectiva en la interfaz de guantes, gafas de seguridad y tapabocas.
- Dibujo de cajas, etiquetas y confianza mediante OpenCV, con conteos mostrados en Streamlit.
- Análisis global de todas las detecciones EPP soportadas que devuelve el modelo para la imagen.
- Análisis opcional por persona usando `yolov8n.pt`, recortes individuales y deduplicación heurística.
- Controles de confianza y tamaño de inferencia.
- Validación de datasets YOLO, script de fine-tuning y evaluación por clase.
- 39 pruebas automatizadas.

No están implementados la cámara, el análisis de video, el seguimiento temporal, la generación de reportes de usuario, la descarga automática de modelos, la detección facial ni advertencias visuales basadas en evidencia facial. La bata y el gorro todavía aparecen como no evaluables en la interfaz actual.

Consulta [Estado actual](docs/ESTADO_ACTUAL.md) para el alcance verificado y sus limitaciones.

## Tecnologías

| Tecnología | Uso actual |
|---|---|
| Python 3.12 o superior | Aplicación, scripts y pruebas |
| Streamlit | Interfaz web local |
| Ultralytics YOLOv8 | Inferencia, entrenamiento y evaluación |
| PyTorch CPU | Ejecución de checkpoints `.pt` en la aplicación |
| OpenCV | Decodificación y anotación de imágenes BGR |
| NumPy | Representación y recortes de imágenes |
| PyYAML | Lectura de configuración y datasets |
| `unittest` | Suite automatizada |

Las dependencias declaradas no están fijadas a versiones exactas. El entorno auditado utilizó Ultralytics `8.4.208`; antes de actualizar dependencias se deben ejecutar todas las pruebas.

## Arquitectura resumida

```text
Streamlit
   ├── análisis global ───────> YOLO EPP ──> reglas globales
   └── análisis por persona ──> YOLO COCO person
                                └── recorte por persona ──> YOLO EPP

OpenCV <── decodificación, cajas, etiquetas y composición visual
```

El análisis global agrupa detecciones de toda la imagen. El análisis individual detecta personas, ejecuta el modelo EPP en cada recorte y traduce las cajas a coordenadas globales. Esta asociación es heurística y no equivale a seguimiento ni estimación de pose.

Consulta [Arquitectura](docs/ARQUITECTURA.md) para el flujo completo.

## Estructura

```text
LabGuard-AI/
├── app.py                         # Entrada de Streamlit
├── configs/
│   ├── model.yaml                 # Configuración activa de modelos e inferencia
│   ├── app.yaml                   # Configuración reservada, todavía no cargada
│   └── compliance.yaml            # Configuración reservada, todavía no cargada
├── docs/                          # Entrega técnica y continuidad
├── models/
│   └── best.pt                    # Peso EPP local, ignorado por Git
├── src/labguard_ai/
│   ├── compliance/                # Reglas globales básicas
│   ├── core/                      # Tipos de dominio
│   ├── detection/                 # Adaptadores YOLO EPP y person
│   ├── processing/                # Flujos global, por persona y futuro video
│   ├── reporting/                 # Contrato futuro, sin generador concreto
│   └── ui/                        # Interfaz y utilidades OpenCV
├── tests/                         # 39 pruebas
├── training/                      # Dataset, validación, entrenamiento y evaluación
├── pyproject.toml
└── yolov8n.pt                     # Peso COCO local, ignorado por Git
```

Los pesos, `data/`, `reports/` y el archivo local `bus.jpg` están excluidos de Git. Otras fotografías no se ignoran de forma general y podrían quedar staged por accidente: revisa siempre `git status` y nunca añadas imágenes privadas. Un clon nuevo necesita obtener los dos modelos por separado.

## Guía rápida

La instalación completa para Windows y WSL2 está en [Instalación](docs/INSTALACION.md).

```bash
git clone https://github.com/Wilsonn2002/LabGuard-AI.git
cd LabGuard-AI
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
```

Después de obtener y verificar `yolov8n.pt` y `models/best.pt` según [Modelos](docs/MODELOS.md), inicia la aplicación:

```bash
python -m streamlit run app.py
```

Abre la URL local indicada por Streamlit, carga una fotografía y ajusta la confianza si es necesario. El análisis individual es más lento porque ejecuta una inferencia EPP por persona.

Ejecuta la suite desde la raíz:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```
##  Interfaz Gráfica (UI)

La interfaz de usuario ha sido desarrollada en Streamlit, ofreciendo un panel de configuración lateral y soporte para procesamiento de imágenes y video en tiempo real.

### Cómo ejecutar la aplicación

1. Asegúrate de estar en el entorno virtual del proyecto y tener las dependencias instaladas (`pip install -r requirements.txt`).
2. Verifica que los modelos base (`yolov8n.pt` o el modelo provisional) estén en el directorio correcto según la configuración en `configs/model.yaml`.
3. Ejecuta el siguiente comando desde la raíz del proyecto:

```bash
streamlit run src/labguard_ai/ui/app.py
## Documentación

- [Instalación en Windows y WSL2](docs/INSTALACION.md)
- [Arquitectura](docs/ARQUITECTURA.md)
- [Modelos y checksums](docs/MODELOS.md)
- [Estado actual y limitaciones](docs/ESTADO_ACTUAL.md)
- [Plan de desarrollo](docs/PLAN_DESARROLLO.md)
- [Guía de colaboración y OpenCode](docs/GUIA_COLABORACION.md)
- [Preparación y entrenamiento](training/README.md)

## Privacidad y seguridad

- No subas a GitHub fotografías de personas, datasets privados, consentimientos ni resultados que permitan identificar participantes.
- Obtén autorización institucional y consentimiento antes de procesar fotografías reales.
- Mantén los datos sensibles en almacenamiento con acceso restringido y define su periodo de conservación.
- No guardes contraseñas, tokens, claves, archivos `.env` ni credenciales de servicios en el repositorio, issues, pull requests o conversaciones con asistentes.
- Verifica origen y SHA-256 antes de cargar pesos `.pt`; los checkpoints de PyTorch deben tratarse como código potencialmente ejecutable.
- Si una credencial se expone, revócala inmediatamente y limpia el historial siguiendo el procedimiento institucional.

## Licencias

Este repositorio no contiene un archivo `LICENSE` en el commit documentado. Antes de redistribuir o desplegar el proyecto se debe definir una licencia propia compatible con todas las obligaciones de terceros y revisar las licencias de Ultralytics, PyTorch, OpenCV, Streamlit y de cada modelo. Consulta [Modelos](docs/MODELOS.md).

---

## Proyecto académico: LabGuard AI

LabGuard AI es un prototipo de visión artificial desarrollado para apoyar
la revisión de cinco elementos de protección personal en laboratorios:

- Bata de laboratorio.
- Guantes de protección.
- Gafas de seguridad.
- Gorro de laboratorio.
- Tapabocas.

### Integrantes

- Wilson Silva Nova
- Douglas Duvan Bonilla
- Johan Beltran Acevedo

### Tecnologías

- Python y Streamlit para la aplicación web.
- YOLOv8 para detección de objetos.
- Hugging Face como fuente del modelo EPP.
- Roboflow para la preparación de datasets.
- CUDA Toolkit para las pruebas de procesamiento en GPU.

### Datos utilizados

Se prepararon dos conjuntos de imágenes:

| Dataset | Entrenamiento | Validación | Pruebas | Total |
|---|---:|---:|---:|---:|
| Dataset 1 | 2103 | 70 | 50 | 2223 |
| Dataset 2 | 180 | 95 | 95 | 370 |
| **Total** | **2283** | **165** | **145** | **2593** |

Se realizaron revisiones de anotaciones y auditorías de consistencia
para identificar posibles etiquetas incorrectas.

### Modelo de inteligencia artificial

Modelo: YOLOv8n PPE Detection, publicado por Tanishjain9 en Hugging Face.

Repositorio:
https://huggingface.co/Tanishjain9/yolov8n-ppe-detection-6classes

- Parámetros verificados: 2.691.378.
- Arquitectura: YOLOv8 Nano.
- Clases originales: 6.
- Clases integradas para inferencia EPP: guantes, gafas y tapabocas.
- Modelo auxiliar: YOLOv8n para detección de personas.

Las clases de bata y gorro forman parte de los datasets preparados
y del alcance previsto del proyecto. Su reconocimiento requiere
un modelo específicamente entrenado y validado para estas categorías.

### Pruebas y CUDA

- 64 pruebas automatizadas aprobadas.
- GPU utilizada para pruebas CUDA: NVIDIA GeForce MX230.
- CUDA Toolkit 12.9.
- Kernel CUDA ejecutado correctamente con cinco hilos.
- La inferencia actual de la aplicación se realiza en CPU.

### Video de presentación

Enlace: pendiente de publicación.
## Arquitectura de detección de personas y EPP

LabGuard AI incorpora una arquitectura de visión artificial basada en dos modelos YOLOv8 especializados:

- **YOLOv8n COCO:** encargado de identificar las personas presentes en las fotografías de laboratorio.
- **LabGuard AI:** modelo entrenado para detectar bata, guantes protectores, gafas de seguridad, gorro de laboratorio y tapabocas.

### Comparación experimental

Se realizó una prueba utilizando una fotografía con dos personas. Ambos modelos detectaron correctamente las dos personas, obteniendo las siguientes puntuaciones de confianza:

| Modelo | Persona 1 | Persona 2 |
|---|---:|---:|
| LabGuard AI | 77,3 % | 58,8 % |
| YOLOv8n COCO | 93,4 % | 92,4 % |

Estos resultados motivaron la separación del detector de personas y el detector de elementos de protección personal, con el objetivo de mejorar el análisis individual.

### Entrenamiento acelerado con CUDA

El modelo propio fue entrenado en Google Colab utilizando una GPU NVIDIA Tesla T4, PyTorch y CUDA. El dataset consolidado contiene 2.593 imágenes, distribuidas en entrenamiento, validación y prueba.

### Mejoras de la aplicación

La interfaz desarrollada en Streamlit permite seleccionar el modelo de reconocimiento, ajustar la confianza y resolución de inferencia, cargar fotografías y visualizar los resultados individuales por persona.

El sistema es experimental y requiere supervisión humana. Las detecciones no constituyen por sí solas una certificación de cumplimiento de seguridad.

