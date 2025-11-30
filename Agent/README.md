# Agent - Entrenamiento de CarAgent con Reinforcement Learning

Proyecto de Reinforcement Learning para entrenar un agente de carro en Unity usando una implementación personalizada de PPO.

## 📋 Tabla de Contenidos

- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Configuración en Unity](#configuración-en-unity)
- [Entrenamiento](#entrenamiento)
- [Arquitectura](#arquitectura)
- [Solución de Problemas](#solución-de-problemas)
- [Estructura del Proyecto](#estructura-del-proyecto)

## 🔧 Requisitos

- **Python 3.10** (especificado en `.python-version`)
- **Unity Editor** con el proyecto que contiene el `CarAgent`
- **uv** (gestor de paquetes Python) - Instalar con: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## 📦 Instalación

1. **Clonar o navegar al directorio del proyecto**:
   ```bash
   cd Agent
   ```

2. **Sincronizar dependencias**:
   ```bash
   uv sync
   ```

   Esto creará el entorno virtual (`.venv`) e instalará todas las dependencias necesarias:
   - `mlagents`
   - `torch`
   - `tensorboard`
   - Y otras dependencias (ver `pyproject.toml`)

## 🎮 Configuración en Unity

✅ **El proyecto Unity ya está configurado y listo para usar.**

El prefab `Car` ya incluye todos los componentes necesarios:
- ✅ `CarAgent` configurado con referencias correctas
- ✅ `Behavior Parameters` configurado:
  - **Behavior Name**: `CarAgent`
  - **Vector Observation Space Size**: `6` (5 sensores + 1 velocidad)
  - **Action Space Size**: `2` (turn, throttle)
       - **Action Type**: `Continuous`
- ✅ Escenas configuradas con `TrackManager` y prefabs listos

**No se requiere configuración manual adicional.** Puedes proceder directamente al entrenamiento.

## 🚀 Entrenamiento

### Entrenamiento con PPO Personalizado

Este proyecto usa **tu implementación de PPO** (`PPO/ppo.py`) para entrenar el agente.

#### Ventajas:
- ✅ Control total sobre el algoritmo
- ✅ Fácil de debuggear y modificar
- ✅ Experimentación libre
- ✅ Usa tu código Python

#### Uso:

**Opción A: Con Build de Linux (por defecto, recomendado)**
```bash
cd Agent
./train_custom_ppo.sh --time-scale 50.0
```

El script usa automáticamente el build de Linux (`../Build/RunCar.x86_64`). Puedes ver la simulación en tiempo real.

**Opción B: Modo headless (sin ventana, más rápido)**
```bash
cd Agent
./train_custom_ppo.sh --no-graphics --time-scale 50.0
```

Útil para entrenamientos largos donde no necesitas ver la simulación.

**Opción C: Con Unity Editor (desarrollo)**
```bash
cd Agent
./train_custom_ppo.sh --env editor
```

Luego, cuando el script indique, **presiona Play en Unity Editor**.

**Opción D: Especificar build personalizado**
```bash
cd Agent
./train_custom_ppo.sh --env /ruta/a/tu/build.x86_64 --time-scale 50.0
```

#### Opciones disponibles:
```bash
./train_custom_ppo.sh [opciones]

Opciones:
  --env PATH          Ruta al ejecutable de Unity (opcional)
                       - Por defecto: usa ../Build/RunCar.x86_64
                       - Usa 'editor' para Unity Editor
  --no-graphics       Ejecutar en modo headless (sin ventana, más rápido)
  --max-steps N       Número máximo de pasos (default: 5000000)
  --time-scale F      Time scale de Unity (default: 20.0)
  --save-dir PATH     Directorio para guardar modelos (default: results/custom_ppo)
  --save-freq N       Frecuencia de guardado en pasos (default: 100000)
  --seed N            Seed para reproducibilidad (default: 1)
```

### Búsqueda de hiperparámetros con Optuna

Para ajustar el algoritmo automáticamente se puede usar el flujo basado en Optuna:

```bash
cd Agent
./train_optuna_ppo.sh \
     --trials 20 \      
     --trial-steps 750000 \    
     --env "../Build/Build2.app" \
     --time-scale 50.0
```

Características principales:

- Ejecuta múltiples trials secuenciales conectados al mismo entorno de Unity.
- Cada trial entrena durante `--trial-steps` pasos y reporta el promedio de recompensa a Optuna para maximizarlo.
- Los modelos y métricas se guardan en `results/optuna_ppo/` (subcarpetas `trials/`, `best_models/` y `tensorboard/`).
- Puedes activar `--storage sqlite:///optuna.db` para reanudar búsquedas largas y `--save-top-k` para conservar los mejores checkpoints.

Flags más útiles:

| Flag | Descripción |
|------|-------------|
| `--trials` | Número de configuraciones a evaluar (`<=0` = infinito) |
| `--trial-steps` | Pasos por trial antes de medir la recompensa |
| `--metric-window` | Episodios promediados para el score |
| `--report-every` | Frecuencia (en pasos) de los reportes a Optuna/pruner |
| `--sampler` / `--pruner` | Estrategias de búsqueda (`tpe`/`random`, `median`/`none`) |
| `--env`, `--no-graphics`, `--time-scale` | Igual que en `train_custom_ppo.py` |

> **Tip:** Después de traer estos cambios corre `uv sync` para instalar la dependencia adicional `optuna`.

#### Configuración del algoritmo:

El algoritmo PPO se configura en `envs/train_custom_ppo.py`. Puedes modificar:
- Learning rate
- Batch size
- Número de pasos antes de actualizar
- Arquitectura de la red neuronal
- Y más...

#### Modelos guardados:

Los modelos se guardan en `results/custom_ppo/`:
- `ppo_step_100000.pt`
- `ppo_step_200000.pt`
- `ppo_final.pt`

### Evaluación del Modelo (Inference)

Para ejecutar Unity con un modelo entrenado sin entrenar (solo ver cómo se comporta):

**Opción A: Con Build de Linux (recomendado)**
```bash
cd Agent
./eval_car_ppo.sh --weights results/custom_ppo/ppo_final.pt --episodes 10
```

**Opción B: Con Unity Editor**
```bash
cd Agent
./eval_car_ppo.sh --weights results/custom_ppo/ppo_final.pt --env editor --episodes 10
```

**Opciones disponibles:**
```bash
./eval_car_ppo.sh [opciones]

Opciones:
  --weights PATH      Ruta al archivo de pesos del modelo (.pt) [REQUERIDO]
  --episodes N       Número de episodios a evaluar (default: 10)
  --env PATH          Ruta al ejecutable de Unity (opcional)
                       - Por defecto: usa ../Build/RunCar.x86_64
                       - Usa 'editor' para Unity Editor
  --time-scale F      Time scale de Unity (default: 1.0, velocidad normal)
  --seed N            Seed para reproducibilidad (default: 1)
```

**Ejemplo:**
```bash
# Evaluar el modelo final con 5 episodios
./eval_car_ppo.sh --weights results/custom_ppo/ppo_final.pt --episodes 5

# Evaluar un checkpoint específico
./eval_car_ppo.sh --weights results/custom_ppo/ppo_step_4100000.pt --episodes 10
```

## 📊 Monitoreo del Entrenamiento

### TensorBoard

Para visualizar el progreso del entrenamiento:

```bash
cd Agent
tensorboard --logdir results/custom_ppo
```

Luego abre `http://localhost:6006` en tu navegador.

### Métricas importantes:

- **Cumulative Reward**: Recompensa acumulada por episodio
- **Policy Loss**: Pérdida de la política
- **Value Loss**: Pérdida del crítico
- **Entropy**: Entropía de la política (exploración)

## 🏗️ Arquitectura

### Componentes en el Prefab Car:

```
Car (GameObject)
├── CarController      → Coordina el carro (checkpoints, muerte, estado)
├── CarMovement        → Maneja el movimiento físico (velocidad, rotación)
├── CarAgent           → Conecta con ML-Agents (recibe acciones del modelo)
└── Behavior Parameters → Configuración de ML-Agents (nombre, observaciones, acciones)
```

### Flujo de Control:

```
PPO Personalizado (Python)
    ↓
    ↓ [envía acciones]
    ↓
CarAgent.OnActionReceived()
    ↓
    ↓ [llama a]
    ↓
CarMovement.SetInputs(turn, throttle)
    ↓
    ↓ [aplica físicamente]
    ↓
Car se mueve
```

### Flujo de Observaciones:

```
Sensores del Car
    ↓
    ↓ [datos]
    ↓
CarAgent.CollectObservations()
    ↓
    ↓ [envía a]
    ↓
PPO Personalizado (Python)
```

### Flujo de Recompensas:

```
CarController (detecta eventos)
    ↓
    ↓ [notifica]
    ↓
CarAgent.OnCheckpointCaptured() / OnWallHit()
    ↓
    ↓ [envía recompensa]
    ↓
PPO Personalizado (Python)
```

### Responsabilidades:

| Componente | Responsabilidad |
|------------|----------------|
| **CarAgent** | Conecta con el PPO personalizado, recibe acciones, envía observaciones y recompensas |
| **CarController** | Coordina el carro, detecta eventos, notifica a CarAgent |
| **CarMovement** | Aplica el movimiento físico (velocidad, rotación) |
| **Behavior Parameters** | Configuración de ML-Agents (nombre, tamaño de observaciones/acciones) |

## 🔍 Solución de Problemas

### Error: "Connection timeout" o "No connection"

**Causa**: Unity no está en Play cuando ML-Agents intenta conectarse.

**Solución**:
1. Asegúrate de que Unity esté en **Play mode** ANTES de que ML-Agents se conecte
2. Verifica que el **Behavior Type** esté en `Default`
3. Verifica que el **Behavior Name** sea exactamente `CarAgent`
4. Verifica que haya al menos un auto activo en la escena

### Error: "More observations (6) made than vector observation size (1)"

**Causa**: El Vector Observation Space Size está configurado incorrectamente (debería ser `6`).

**Solución** (si el error persiste):
1. Abre el prefab `Car` en Unity Editor
2. En `Behavior Parameters` → `Vector Observation`
3. Verifica que **Space Size** sea `6` (5 sensores + 1 velocidad)
4. Guarda el prefab

**Nota**: El proyecto ya viene configurado con `Space Size: 6`, así que este error no debería ocurrir.

### Error: "Behavior name mismatch"

**Causa**: El Behavior Name en Unity no coincide con el esperado.

**Solución**:
1. Verifica que el Behavior Name sea exactamente `CarAgent` (case-sensitive)

### El auto aparece pero no se mueve

**Posibles causas**:

1. **ML-Agents no está conectado**:
   - Verifica que el script de entrenamiento esté ejecutándose
   - Verifica que Unity esté en Play mode
   - Revisa la consola de Unity para errores

2. **Behavior Type incorrecto**:
   - Debe ser `Default` (NO `Heuristic Only` o `Inference Only`)

3. **Componentes deshabilitados**:
   - Verifica que `CarMovement`, `CarController` y `CarAgent` estén habilitados

4. **CarMovement no aplica inputs**:
   - Verifica que `CarController.UseUserInput` esté en `false` (para ML-Agents)

### No aparecen autos y no entrena

**Causa**: ML-Agents necesita que haya al menos un agente activo en la escena.

**Solución** (si el error persiste):
1. **Verificar que la escena tenga autos**:
   - Las escenas ya vienen configuradas con autos
   - Si no aparecen, verifica que la escena esté cargada correctamente

2. **Verificar TrackManager**:
   - El `TrackManager` ya está configurado con el prefab `Car`
   - Si es necesario, verifica que `PrototypeCar` esté asignado

3. **Si necesitas agregar más autos manualmente**:
   - Arrastra el prefab `Car` a la escena
   - Posiciónalo en la posición inicial
   - Asegúrate de que esté activo

### Timeout después de conexión exitosa

**Causa**: El agente no está respondiendo correctamente a ML-Agents.

**Solución**:
1. Verifica que el prefab `Car` tenga `CarAgent` y `Behavior Parameters` configurados correctamente
2. Verifica que el auto esté activo en la escena
3. Revisa la consola de Unity para errores en `OnEpisodeBegin()` o `CollectObservations()`
4. Verifica que el auto tenga 5 sensores configurados


### El entrenamiento es muy lento

**Soluciones**:
1. Aumenta el `--time-scale` (ej: `--time-scale 50.0`)
2. Usa un build de Unity en lugar del Editor

### Error: "No module named 'torch'"

**Solución**:
```bash
cd Agent
uv sync
```

## 📁 Estructura del Proyecto

```
Agent/
├── envs/                    # Scripts de entrenamiento
│   ├── train_custom_ppo.py  # Entrenamiento con PPO personalizado
│   └── ...
├── PPO/                     # Implementación de PPO
│   ├── ppo.py              # Algoritmo PPO
│   └── rollout.py          # Rollout buffer
├── results/                 # Resultados del entrenamiento
│   └── custom_ppo/         # Resultados de PPO personalizado
├── train_custom_ppo.sh     # Script para entrenamiento
├── pyproject.toml          # Dependencias del proyecto
└── README.md               # Este archivo
```

## 📝 Notas Importantes

1. **Orden de ejecución**: 
   - Abre Unity Editor y carga la escena principal (ya configurada)
   - NO presiones Play todavía
   - Ejecuta el script de entrenamiento
   - Cuando el script indique, presiona Play en Unity

2. **Unity como entorno**: Unity solo proporciona la simulación. Tu código Python es el que aprende usando tu implementación de PPO.

3. **Asíncrono**: Unity es asíncrono. Los agentes pueden terminar en diferentes momentos, y los scripts manejan esto automáticamente.

4. **Múltiples agentes**: Los scripts actualmente manejan un agente a la vez. Para múltiples agentes, necesitarás modificar el código.

5. **Recompensas**: Las recompensas vienen de `CarAgent.cs` (métodos `OnCheckpointCaptured()`, `OnWallHit()`, etc.).

## 🔗 Referencias

- [Documentación oficial de ML-Agents](https://github.com/Unity-Technologies/ml-agents) (para la API de bajo nivel)
- [Proximal Policy Optimization (PPO) - Paper original](https://arxiv.org/abs/1707.06347)

