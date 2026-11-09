# Guía de colaboración

Esta guía está pensada para dos o más desarrolladores trabajando mediante GitHub y OpenCode.

## Principios

- `main` debe permanecer ejecutable y con pruebas verdes.
- Cada cambio se desarrolla en una rama propia.
- Un pull request debe resolver un objetivo concreto y pequeño.
- Código, documentación y pruebas deben cambiar juntos cuando corresponda.
- No se suben pesos, datasets, fotografías privadas, reportes, tokens ni entornos virtuales.
- La ausencia de detección nunca debe documentarse como incumplimiento confirmado.

## Preparar una tarea

Actualiza `main` sin crear merges locales innecesarios:

```bash
git switch main
git pull --ff-only origin main
git status
```

Crea una rama descriptiva:

```bash
git switch -c feat/mejorar-deteccion-gafas
```

Prefijos recomendados:

| Prefijo | Uso |
|---|---|
| `feat/` | Funcionalidad nueva |
| `fix/` | Corrección de defecto |
| `docs/` | Documentación |
| `test/` | Cobertura automatizada |
| `refactor/` | Cambio interno sin alterar alcance |
| `chore/` | Mantenimiento |

Evita que dos personas trabajen sobre la misma rama. Coordinen la propiedad de archivos sensibles como `configs/model.yaml` y `src/labguard_ai/ui/app.py`.

## Ciclo de desarrollo

Antes de editar:

```bash
git status --short --branch
git log --oneline -5
```

Durante el trabajo:

```bash
git diff
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

Antes del commit:

```bash
git status --short
git diff --check
git diff
```

Revisa que no aparezcan `.pt`, fotografías, `.env`, claves o datos personales.

## Commits

Usa mensajes breves en modo imperativo y, preferiblemente, Conventional Commits:

```text
feat: add lab coat class mapping
fix: avoid duplicate PPE assignment
docs: document CPU installation
test: cover overlapping person crops
```

Separa cambios conceptualmente distintos. No uses `git add .` sin revisar; agrega solo los archivos previstos:

```bash
git add README.md docs/ARQUITECTURA.md
git commit -m "docs: document current architecture"
```

No reescribas commits compartidos ni hagas force push sobre ramas ajenas. Nunca uses `git reset --hard` para resolver cambios que no comprendes.

## Pull requests

Publica la rama:

```bash
git push -u origin feat/mejorar-deteccion-gafas
```

El pull request debe incluir:

- Problema y alcance.
- Cambios realizados.
- Cómo se verificó.
- Resultado de las 39 pruebas.
- Capturas solo con imágenes públicas o autorizadas y sin datos identificables.
- Riesgos, limitaciones y trabajo pendiente.
- Cambios de modelo con fuente, revisión, clases, métricas, licencia y SHA-256.

Lista mínima de revisión:

```text
[ ] El código corresponde al objetivo del PR.
[ ] No se presentan funciones futuras como terminadas.
[ ] Las pruebas pasan.
[ ] No hay pesos, datasets, fotografías ni credenciales.
[ ] Se revisaron privacidad y licencias.
[ ] La documentación refleja el comportamiento real.
```

La persona autora no debe aprobar su propio PR cuando haya otro colaborador disponible. Usa GitHub Issues para acordar criterios antes de cambios grandes.

## Continuar con OpenCode

OpenCode puede inspeccionar y modificar el mismo workspace local. Inícialo desde la raíz del repositorio según la instalación de cada desarrollador, normalmente:

```bash
cd ~/projects/LabGuard-AI
opencode
```

Al comenzar una conversación, proporciona contexto verificable:

```text
Trabaja desde la rama feat/mejorar-deteccion-gafas.
Revisa README.md, docs/ESTADO_ACTUAL.md y el código antes de editar.
No descargues modelos ni datos.
Ejecuta las 39 pruebas y no hagas commit ni push sin confirmación.
```

Pide siempre que revise `git status` y el commit actual. Antes de aceptar cambios, inspecciona tú mismo `git diff` y el resultado de las pruebas.

### Las conversaciones no se comparten por GitHub

GitHub comparte únicamente archivos y commits que se agregan y publican. El historial de una conversación de OpenCode, sus decisiones intermedias y el contexto de una sesión no se incluyen automáticamente en `git push`, un issue o un pull request.

Para transferir trabajo a otro compañero:

1. Deja código y documentación coherentes en la rama.
2. Registra decisiones relevantes en el issue, PR o documento correspondiente.
3. Incluye comandos de verificación y limitaciones.
4. Publica la rama o commit solo después de revisión humana.
5. El siguiente desarrollador debe iniciar una conversación nueva y pedir a OpenCode que inspeccione el repositorio; no debe asumir que conoce el chat anterior.

No copies conversaciones completas al repositorio. Resume únicamente decisiones técnicas necesarias y evita información personal, rutas privadas, tokens o contenido de fotografías. La retención de chats depende de la configuración y proveedor de OpenCode; sigue la política institucional además de Git.

## Conflictos entre colaboradores

Antes de actualizar tu rama:

```bash
git fetch origin
git rebase origin/main
```

Usa rebase solo en tu rama no compartida o después de coordinarlo. Si la rama ya es utilizada por otra persona, prefiere acordar el método antes de reescribir historial.

Después de resolver conflictos:

```bash
git diff --check
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
```

No elijas automáticamente una versión de `configs/model.yaml`: revisa nombres de clase, rutas y hashes.

## Pesos, datos y fotografías

- No uses `git add -f` para incluir modelos ignorados.
- No uses Git LFS sin una decisión explícita del equipo y revisión de licencias; el flujo actual comparte URL, revisión y SHA-256.
- No confíes en `.gitignore` para todas las fotografías: fuera de `data/`, una imagen común puede quedar staged. Revisa cada ruta antes del commit.
- Guarda datasets fuera del repositorio y con permisos mínimos.
- Separa `train`, `val` y `test` por persona, sesión o laboratorio.
- Elimina metadatos innecesarios y acuerda retención antes de capturar.
- No publiques ejemplos con rostros, nombres, pantallas o ubicaciones identificables.

## Credenciales

- Usa `gh auth login` o el gestor de credenciales de Git, no tokens dentro de URLs o scripts.
- Mantén secretos fuera de `.env.example`.
- No pegues secretos en OpenCode, GitHub Issues, PRs, terminales grabadas o capturas.
- Revisa staged files con `git diff --cached` antes de cada commit.
- Si un secreto llega a Git, revócalo primero; borrar el archivo en un commit posterior no elimina el historial.

## Entrega entre compañeros

Cada traspaso debe indicar:

- Rama y último commit.
- Objetivo y estado.
- Archivos modificados.
- Pruebas ejecutadas.
- Modelos/datos externos necesarios con hashes, nunca adjuntos privados.
- Limitaciones conocidas.
- Próximo paso concreto.

Usa [Estado actual](ESTADO_ACTUAL.md) como línea base y actualízalo cuando una funcionalidad pase de planificada a verificada.
