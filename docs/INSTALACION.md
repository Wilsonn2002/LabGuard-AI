# Instalación en Windows con WSL2 Ubuntu

Esta guía prepara la aplicación actual para inferencia en CPU. No instala CUDA ni utiliza la GPU NVIDIA MX230. Los comandos se ejecutan desde una terminal WSL, salvo donde se indique PowerShell.

## 1. Instalar WSL2 y Ubuntu

Abre PowerShell como administrador:

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
wsl --status
```

Reinicia Windows si lo solicita y abre Ubuntu para crear el usuario Linux. Verifica la distribución:

```bash
lsb_release -a
```

Ubuntu 24.04 incluye Python 3.12. Para mejor rendimiento, clona el proyecto dentro del sistema Linux, por ejemplo `~/projects`, y no bajo `/mnt/c`.

## 2. Paquetes del sistema

```bash
sudo apt update
sudo apt install -y git curl python3.12 python3.12-venv python3-pip libgl1 libglib2.0-0
python3.12 --version
git --version
```

Se espera Python `3.12.x` o superior. `libgl1` y `libglib2.0-0` evitan errores comunes al importar OpenCV.

## 3. Clonar el repositorio

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/Wilsonn2002/LabGuard-AI.git
cd LabGuard-AI
git status
```

Confirma que la historia contiene la base técnica auditada:

```bash
git merge-base --is-ancestor 1929ba4 HEAD && echo 'Base 1929ba4 presente'
```

No cambies el checkout principal a `1929ba4`: ese commit es la base de código anterior a estos documentos y el cambio los retiraría del directorio. Si necesitas inspeccionar exactamente esa versión, crea un worktree separado:

```bash
git worktree add ../LabGuard-AI-1929ba4 1929ba4
```

Para desarrollar, actualiza `main` y crea una rama según [Guía de colaboración](GUIA_COLABORACION.md).

## 4. Entorno virtual

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install --upgrade pip setuptools wheel
```

Cada terminal nueva requiere:

```bash
cd ~/projects/LabGuard-AI
source .venv/bin/activate
```

Para salir del entorno:

```bash
deactivate
```

## 5. PyTorch CPU

Instala primero los wheels oficiales para CPU:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Comprueba el backend:

```bash
python -c "import torch; print('torch:', torch.__version__); print('CUDA disponible:', torch.cuda.is_available())"
```

En esta instalación se espera `CUDA disponible: False`. La interfaz valida `device: cpu` y rechaza otros dispositivos.

## 6. Dependencias del proyecto

```bash
python -m pip install -e .
```

Esto instala las dependencias declaradas en `pyproject.toml`: NumPy, OpenCV, PyYAML, Streamlit y Ultralytics. PyTorch ya instalado satisface la dependencia de Ultralytics.

Verifica las importaciones:

```bash
python -c "import cv2, numpy, streamlit, torch, ultralytics, yaml; print('Dependencias OK')"
python -c "import ultralytics; print('Ultralytics:', ultralytics.__version__)"
```

Las versiones no están bloqueadas en `pyproject.toml`. El entorno auditado utilizó Ultralytics `8.4.208`. Después de cualquier actualización ejecuta la suite completa.

## 7. Modelos

Un clon no incluye pesos porque `*.pt` está ignorado por Git. Deben existir exactamente estas rutas:

```text
LabGuard-AI/yolov8n.pt
LabGuard-AI/models/best.pt
```

Sigue [Modelos](MODELOS.md) para descargarlos desde sus fuentes, verificar SHA-256 y revisar licencias. No aceptes pesos enviados por canales no confiables.

## 8. Ejecutar pruebas

Desde la raíz y con el entorno activado:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

El estado de referencia es `Ran 39 tests` y `OK`. Estas pruebas usan dobles de modelo y datos temporales; no validan la precisión real de los pesos.

## 9. Iniciar Streamlit

```bash
python -m streamlit run app.py
```

Streamlit mostrará una URL como `http://localhost:8501`. Ábrela desde Windows. Para detener el servidor usa `Ctrl+C`.

No uses `python app.py` como comando habitual: el proceso Streamlit debe iniciarse con `streamlit run` para disponer del contexto de ejecución web.

## 10. Diagnóstico rápido

### Falta un modelo

Mensaje esperado: `No se encontro el modelo YOLO` o `No se encontro el modelo de personas`.

```bash
ls -lh yolov8n.pt models/best.pt
sha256sum yolov8n.pt models/best.pt
```

Compara el resultado con [Modelos](MODELOS.md).

### OpenCV no carga

```bash
sudo apt install -y libgl1 libglib2.0-0
python -c "import cv2; print(cv2.__version__)"
```

### Puerto ocupado

```bash
python -m streamlit run app.py --server.port 8502
```

### Entorno equivocado

```bash
which python
python -m pip --version
```

Ambos deben apuntar a `.venv`.

## 11. Privacidad y credenciales

- No coloques fotografías reales ni datasets en el repositorio. `data/` está ignorado, pero otras imágenes pueden ser rastreadas por Git; esto tampoco sustituye control de acceso, consentimiento ni cifrado.
- No ejecutes Streamlit en una dirección pública con fotografías sensibles sin autenticación y evaluación de seguridad.
- No escribas tokens en comandos que queden en el historial. Para GitHub prefiere el gestor de credenciales o `gh auth login`.
- Los archivos `.env`, `.env.*` y `*.key` están ignorados, excepto `.env.example`; nunca pongas secretos reales en el ejemplo.
- No pegues contraseñas, tokens o fotografías privadas en issues, pull requests, chats u órdenes para asistentes.
