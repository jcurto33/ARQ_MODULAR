# 📦 <Nombre del Proyecto>

> **Descripción breve** — Una o dos frases que resuman el propósito del proyecto y los módulos que contiene.

| Campo | Detalle |
|---|---|
| **Autor(es)** | Nombre Apellido |
| **Fecha de inicio** | YYYY-MM |
| **Estado** | 🟡 En desarrollo / 🟢 Producción / 🔴 Archivado |

---

## Índice

1. [Arquitectura del repositorio](#arquitectura-del-repositorio)
2. [Requisitos previos](#requisitos-previos)
3. [Instalación y configuración](#instalación-y-configuración)
4. [Uso](#uso)
5. [Estructura de datos (`data/`)](#estructura-de-datos-data)
6. [Módulos (`src/`)](#módulos-src)
7. [API (`api/`)](#api-api)
8. [Demo (`demo/`)](#demo-demo)
9. [Docker](#docker)
10. [Scripts](#scripts)
11. [Tests](#tests)
12. [Notebooks](#notebooks)
13. [Flujo de trabajo con Git](#flujo-de-trabajo-con-git)
14. [Notas adicionales](#notas-adicionales)

---

## Arquitectura del repositorio

> Arquitectura modular estándar del equipo de IA. Cada carpeta tiene una responsabilidad clara. Consultar el documento *Arquitectura modular de repositorios* para la referencia completa.

```
.
├── requirements.txt            # Dependencias de producción
├── requirements-dev.txt        # Dependencias de desarrollo (jupyter, pytest…)
├── README.md                   # Este archivo
├── .env                        # Variables de entorno / secretos (NO versionar)
├── .env.example                # Plantilla de .env
├── .gitignore                  # Exclusiones de Git
├── .venv/                      # Entorno virtual local (NO versionar)
├── docker-compose.yaml         # Orquestación de servicios
├── pytest.ini                  # Configuración de tests
│
├── config/                     # Configuración estática (YAML)
│   ├── global_config.yaml      #   Rutas, parámetros, flags
│   └── logging_config.yaml     #   Niveles y destinos de log
│
├── data/                       # Persistencia y estado (volumen Docker, NO versionar)
│   ├── models/                 #   Artefactos (.pkl, .pt) y métricas
│   │   ├── artifacts/
│   │   └── metrics/
│   ├── files/                  #   Archivos planos
│   │   ├── raw/                #     Datos crudos e inmutables
│   │   ├── processed/          #     Datos limpios
│   │   ├── predictions/        #     Resultados batch
│   │   └── splits/             #     Train / test exportados
│   ├── bbdd/                   #   Bases de datos
│   │   ├── sql/                #     SQLite / relacional
│   │   └── vector/             #     ChromaDB / LanceDB
│   └── mlflow/                 #   Tracking de experimentos
│
├── shared/                     # Librerías internas compartidas
│   ├── __init__.py
│   ├── config_loader.py        #     Carga de global_config.yaml
│   ├── database/
│   │   ├── connection.py       #     Factory de conexión SQL
│   │   └── vector_store.py     #     Conexión a BBDD vectorial
│   └── utils/
│       ├── llm_client.py       #     Abstracción de modelos de lenguaje
│       └── security.py         #     Tokens, hashing, claves API
│
├── src/                        # Lógica central (producto)
│   ├── __init__.py
│   └── modules/
│       ├── [modulo_ml]/        #     Módulo ML clásico
│       │   ├── __init__.py
│       │   ├── entrypoint.py   #       Interfaz pública: train / predict
│       │   ├── training/
│       │   ├── inference/
│       │   └── data_processing/
│       └── [modulo_llm]/       #     Módulo Agente LLM
│           ├── __init__.py
│           ├── entrypoint.py   #       Interfaz pública: chat / run_server
│           ├── agents/
│           ├── agentflows/
│           ├── prompts/
│           ├── mcp_server/
│           └── data_processing/
│
├── api/                        # Capa REST (FastAPI)
│   ├── main.py                 #   Bootstrap, middlewares, CORS
│   └── routers/                #   Un router por módulo de src/
│
├── demo/                       # Capa visual (Streamlit)
│   ├── app.py                  #   Gestor de navegación
│   └── pages/                  #   Vistas por caso de uso
│
├── docker/                     # Dockerfiles por servicio
│   ├── api/
│   │   └── Dockerfile
│   └── demo/
│       └── Dockerfile
│
├── scripts/                    # Ejecución manual / CLI
│   ├── <modulo>_ingest.py
│   ├── <modulo>_train.py
│   └── <modulo>_predict.py
│
├── notebooks/                  # Sandbox de experimentación
│   ├── EDA.ipynb
│   ├── experiments.ipynb
│   ├── evaluation.ipynb
│   └── XAI.ipynb
│
└── tests/                      # Aseguramiento de calidad
    ├── conftest.py             #   Fixtures compartidas
    ├── unit/                   #   Tests unitarios
    └── integration/            #   Tests de integración
```

> **Nota:** Eliminar las carpetas y archivos que no apliquen al proyecto.
> Por ejemplo, si no hay módulo LLM se elimina `agents/`, `prompts/`, `mcp_server/`, etc.
> Si no hay módulo ML se elimina `training/`, `inference/`, etc.

---

## Requisitos previos

| Herramienta | Versión mínima | Notas |
|---|---|---|
| Python | 3.10+ | Se recomienda gestionar entornos con **conda** o **UV** |
| Docker & Docker Compose | 24+ / v2 | Necesario para despliegue |
| Git | 2.x | — |

<!-- Añadir o editar aquí cualquier otro prerequisito específico del proyecto (ej. GPU, credenciales de API, acceso a S3…) -->

---

## Instalación y configuración

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd <nombre-del-repo>
```

### 2. Crear el entorno virtual e instalar dependencias

```bash
# Con pip / venv
python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
pip install -r requirements-dev.txt   # solo para desarrollo

# Con conda
conda create -n <nombre-entorno> python=3.10
conda activate <nombre-entorno>
pip install -r requirements.txt
```

<!-- Modificar esta sección en caso de usar otro gestor de librerías como UV -->

### 3. Configurar variables de entorno

Crear un archivo `.env` en la raíz a partir del ejemplo:

```bash
cp .env.example .env
# Editar .env con las claves y secretos necesarios
```

### 4. Preparar la carpeta `data/`

<!-- Describir aquí los pasos necesarios para preparar los datos iniciales del proyecto.
     Ejemplos:
     - Copiar ficheros fuente a data/files/raw/
     - Ejecutar scripts de ingesta
     - Descargar datasets
-->

```bash
# Ejemplo: copiar datos fuente
mkdir -p data/files/raw
cp <archivo_fuente> data/files/raw/
```

---

## Uso

### Ejecución rápida

<!-- Listar los scripts en el orden en que deben ejecutarse para poner el proyecto en marcha -->

```bash
# 1. Ingestar datos
python scripts/<modulo>_ingest.py

# 2. Entrenar modelo (si aplica)
python scripts/<modulo>_train.py

# 3. Ejecutar predicción / inferencia (si aplica)
python scripts/<modulo>_predict.py
```

### Lanzar la demo de Streamlit en local

Streamlit importa directamente desde `src/`, por lo que **no necesitas levantar la API**.

```bash
streamlit run demo/app.py
```

Se abrirá automáticamente en `http://localhost:8501`. Usa el menú lateral para navegar entre las páginas.

### Levantar la API en local

```bash
uvicorn api.main:app --reload --port 8000
```

Documentación interactiva en `http://localhost:8000/docs` (Swagger UI).

### Levantar servicios con Docker

```bash
docker-compose up --build
```

| Servicio | Puerto por defecto | Descripción |
|---|---|---|
| API (FastAPI) | `8000` | Backend REST |
| Demo (Streamlit) | `8501` | Interfaz de usuario |

---

## Estructura de datos (`data/`)

> Esta carpeta se monta como **volumen Docker** para persistencia. Está en `.gitignore`.

| Subcarpeta | Contenido |
|---|---|
| `models/artifacts/` | Modelos serializados (`.pkl`, `.pt`) |
| `models/metrics/` | Reportes de evaluación (`.json`, `.csv`) |
| `files/raw/` | Datos crudos e inmutables |
| `files/processed/` | Datos limpios listos para modelado / ingesta RAG |
| `files/predictions/` | Resultados de procesos batch |
| `files/splits/` | Datasets train / test exportados |
| `bbdd/sql/` | SQLite u otras BBDD relacionales |
| `bbdd/vector/` | Índices vectoriales (ChromaDB / LanceDB / Qdrant) |
| `mlflow/` | Backend store y artifact store de MLflow |

---

## Módulos (`src/`)

Cada módulo dentro de `src/modules/` es autónomo y expone su funcionalidad a través de un archivo **`entrypoint.py`** con las funciones que se usarán en producción.

<!-- Rellenar la tabla con los módulos del proyecto -->

| Módulo | Tipo | Entrypoint | Descripción |
|---|---|---|---|
| `<nombre_modulo>` | ML Clásico / Agente LLM | `train()`, `predict()` / `chat()` | Breve descripción |

---

## API (`api/`)

Capa REST construida con **FastAPI**.

- **`main.py`** — Bootstrap de la app, configuración de middlewares (CORS, logging) y registro de routers.
- **`routers/`** — Un router por módulo de `src/`. Validan inputs con Pydantic y delegan en los entrypoints.

Documentación interactiva disponible en `http://localhost:8000/docs` (Swagger UI).

<!-- Rellenar la tabla con los endpoints del proyecto -->

| Método | Endpoint | Descripción |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/<modulo>/<accion>` | Descripción del endpoint |

---

## Demo (`demo/`)

Interfaz visual con **Streamlit** para demos internas y validación. Importa directamente los entrypoints de `src/`, por lo que **no requiere la API** para funcionar.

- **`app.py`** — Gestor de navegación y configuración de `sys.path`.
- **`pages/`** — Una vista por caso de uso.

<!-- Listar las páginas creadas -->

```bash
streamlit run demo/app.py
```

---

## Docker

La dockerización es **modular**: cada servicio (API, Demo) tiene su propio `Dockerfile` en `docker/`, optimizando dependencias, seguridad y tamaño de imagen.

```bash
# Levantar todo
docker-compose up --build

# Solo API
docker-compose up api

# Solo Demo
docker-compose up demo
```

---

## Scripts

Wrappers CLI que configuran el entorno y llaman a los entrypoints de `src/`.

<!-- Rellenar la tabla con los scripts del proyecto -->

| Script | Descripción |
|---|---|
| `scripts/<modulo>_ingest.py` | Ingesta de datos del módulo |
| `scripts/<modulo>_train.py` | Entrenamiento del modelo |
| `scripts/<modulo>_predict.py` | Predicción / inferencia batch |

---

## Tests

Ejecutar la suite de tests:

```bash
# Todos los tests
pytest

# Solo unitarios
pytest tests/unit/

# Solo integración
pytest tests/integration/

# Con cobertura
pytest --cov=src
```

- **`conftest.py`** — Fixtures compartidas (BBDD en memoria, mocks).
- **`unit/`** — Tests rápidos de lógica pura, sin dependencias externas.
- **`integration/`** — Tests de componentes conectados (API ↔ BBDD ↔ Modelo).

---

## Notebooks

Carpeta de **experimentación** (`notebooks/`). Aquí se desarrollan los prototipos que, una vez validados, se encapsulan en `src/` como código de producción.

<!-- Rellenar la tabla con los notebooks del proyecto -->

| Notebook | Propósito |
|---|---|
| `EDA.ipynb` | Análisis exploratorio de datos |
| `experiments.ipynb` | Comparativa de modelos / agentes |
| `evaluation.ipynb` | Evaluación de rendimiento: métricas, matrices, curvas |
| `XAI.ipynb` | Explicabilidad: feature importances, SHAP values |

---

## Flujo de trabajo con Git

<!-- Describir la estrategia de branching del equipo. Ejemplo: -->

| Rama | Propósito |
|---|---|
| `main` | Código estable / producción |
| `develop` | Integración de features |
| `feature/<nombre>` | Desarrollo de funcionalidades |
| `hotfix/<nombre>` | Correcciones urgentes |

```bash
# Crear una feature branch
git checkout -b feature/<nombre-descriptivo>

# Hacer commits atómicos
git add .
git commit -m "feat: descripción del cambio"

# Merge a develop
git checkout develop
git merge feature/<nombre-descriptivo>
```

---

## Notas adicionales

<!-- Espacio libre para documentar decisiones técnicas, limitaciones conocidas, pendientes, etc. -->

- Completar tras la configuración inicial del proyecto.
