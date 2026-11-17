# Best Execution con aprendizaje por refuerzo

Proyecto académico de ejecución algorítmica desarrollado para el Máster BME. El repositorio construye, de forma incremental, un entorno de ejecución de órdenes CARE, una referencia TWAP y un agente DDQN con exploración adaptativa, parada temprana y checkpoints reproducibles.

> [!IMPORTANT]
> Es un simulador con fines docentes y experimentales. No modela impacto de mercado ni costes de transacción y no debe utilizarse como sistema de ejecución en producción.

## Navegación rápida

| Ruta | Contenido |
| --- | --- |
| [`src/`](src/) | Implementación reutilizable del entorno, recompensas, agentes y entrenamiento. |
| [`workshop/submission/`](workshop/submission/) | Notebooks entregables de los ejercicios 1, 2 y 3 + 4C. |
| [`tests/`](tests/) | Pruebas del entorno, TWAP, exploración, parada y checkpoints. |
| [`data/`](data/) | Datos de mercado serializados utilizados por los notebooks. |
| [`output/pdf/`](output/pdf/) | Resúmenes técnicos y presentación exportada a PDF. |
| [`images/`](images/) | Gráficos e imágenes empleados en el workshop. |

## Cómo encajan las piezas

```text
Datos de mercado
      │
      ▼
BestExecutionEnv ─── EnvironmentConfig
      │
      ├── RewardStrategy ── VWAP / TWAP
      │
      ▼
DDQNAgent ────────── ExplorationPolicy
      │               ├── agenda lineal
      │               └── epsilon por desempeño
      ▼
Trainer ───────────── TwapEvaluator
      │
      ├── parada temprana
      ├── checkpoints y reanudación
      └── historiales de entrenamiento y evaluación
```

- El entorno encapsula el estado del episodio, las acciones y el registro de ejecuciones.
- Las recompensas implementan una interfaz común, por lo que pueden cambiarse sin reescribir el entorno.
- El agente DDQN recibe una política de exploración inyectable.
- El entrenador coordina aprendizaje, evaluación, parada y persistencia.
- Los notebooks cargan los datos, configuran estos componentes y conservan la evidencia experimental.

## Estructura del repositorio

```text
best-execution/
├── src/
│   ├── environment/
│   │   ├── best_execution.py   # Entorno y dinámica de los episodios
│   │   └── config.py           # Configuración tipada e inmutable
│   ├── rewards/
│   │   ├── base.py             # Contrato RewardStrategy y RewardContext
│   │   ├── vwap.py             # Recompensa respecto al VWAP
│   │   └── twap.py             # Recompensa de seguimiento temporal
│   ├── agents/
│   │   ├── dqn.py              # Agente Double DQN
│   │   ├── exploration.py      # Exploración lineal y por desempeño
│   │   └── twap.py             # Política TWAP determinista de referencia
│   ├── training/
│   │   ├── evaluation.py       # Casos fijos y métricas de evaluación
│   │   └── trainer.py          # Entrenamiento, parada y checkpoints
│   └── report/
│       └── report.py           # Utilidades de visualización
├── workshop/
│   ├── execution_lab.ipynb
│   ├── execution_lab-step_by_step.ipynb
│   └── submission/
│       ├── exercise_1.ipynb
│       ├── exercise_2.ipynb
│       └── exercise_3_4.ipynb
├── tests/
│   ├── test_environment.py
│   ├── test_twap.py
│   └── test_exploration.py
├── data/                       # REP, SAN y datos originales del workshop
├── images/                     # Evidencia gráfica
├── output/
│   ├── pdf/                    # Informes y presentación exportada
│   └── pptx/                   # Fuentes o exportaciones de la presentación
├── .github/workflows/tests.yml # Integración continua con Python 3.11
├── setup.py                    # Paquete y dependencias canónicas
└── LICENSE
```

## Evolución por ejercicios

| Bloque | Objetivo | Código principal | Notebook |
| --- | --- | --- | --- |
| Ejercicio 1 | Construir un entorno modular, una recompensa VWAP y una demostración DDQN reproducible. | `src/environment/`, `src/rewards/vwap.py`, `src/agents/dqn.py` | [`exercise_1.ipynb`](workshop/submission/exercise_1.ipynb) |
| Ejercicio 2 | Aprender una pauta TWAP, calibrar el agente y evaluar la finalización de las órdenes. | `src/rewards/twap.py`, `src/agents/twap.py` | [`exercise_2.ipynb`](workshop/submission/exercise_2.ipynb) |
| Ejercicio 3 + 4C | Adaptar epsilon al desempeño, detener el entrenamiento y guardar o reanudar checkpoints. | `src/agents/exploration.py`, `src/training/` | [`exercise_3_4.ipynb`](workshop/submission/exercise_3_4.ipynb) |

## Instalación recomendada

La fuente canónica de dependencias es [`setup.py`](setup.py). El proyecto admite Python `>=3.9.2,<3.12`; se recomienda Python 3.11, que es también la versión utilizada en integración continua.

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

La instalación editable permite importar el paquete desde los notebooks:

```python
from src.agents.dqn import DDQNAgent
from src.environment.best_execution import BestExecutionEnv
from src.environment.config import EnvironmentConfig
from src.rewards import TwapRewardStrategy, VwapRewardStrategy
```

### Sobre el Dev Container

Las carpetas [`.devcontainer/`](.devcontainer/) y [`dockerfiles/`](dockerfiles/) conservan el entorno original del workshop, basado en la imagen `jesussanz/best_execution:v1`. Su archivo `requirements.txt` contiene versiones históricas y no debe combinarse con las dependencias actuales de `setup.py`. Para reproducir los tests y los notebooks entregables se recomienda la instalación local anterior.

## Ejecución

### Tests

```bash
python -m pytest tests -q
```

La misma suite se ejecuta automáticamente en [GitHub Actions](.github/workflows/tests.yml).

### Notebooks

Inicia Jupyter desde la raíz del repositorio para mantener válidas las rutas a `data/` y `src/`:

```bash
jupyter lab
```

Orden recomendado de lectura:

1. [`workshop/submission/exercise_1.ipynb`](workshop/submission/exercise_1.ipynb)
2. [`workshop/submission/exercise_2.ipynb`](workshop/submission/exercise_2.ipynb)
3. [`workshop/submission/exercise_3_4.ipynb`](workshop/submission/exercise_3_4.ipynb)

## Datos

Los notebooks utilizan principalmente `data/rep_data.pkl`, que contiene particiones como `train`. Los ficheros de `data/new_version/` corresponden al material original o alternativo del workshop.

Los archivos son binarios de confianza del proyecto y se cargan con `pickle`; no debe utilizarse este mecanismo con datos descargados de fuentes no confiables.

## Documentación y resultados

- [Resumen consolidado de decisiones técnicas](output/pdf/resumen_decisiones_tecnicas_completo.pdf)
- [Resumen del ejercicio 1](output/pdf/resumen_decisiones_ejercicio_1.pdf)
- [Resumen del ejercicio 2](output/pdf/resumen_decisiones_ejercicio_2.pdf)
- [Resumen de los ejercicios 3 + 4C](output/pdf/resumen_decisiones_ejercicio_3_4.pdf)
- [Presentación de cinco minutos](output/pdf/best_execution_presentacion_5_minutos.pdf)

## Integrantes

- Javier Chulvi
- Hugo Vinuesa
- Higinio Paterna

## Licencia

Este repositorio se distribuye bajo los términos de [`LICENSE`](LICENSE).
