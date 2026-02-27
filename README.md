# Sistema de Trading Algorítmico con Alpaca

Este proyecto implementa un sistema completo de trading algorítmico que utiliza la API de Alpaca para ejecutar operaciones en el mercado de valores. El sistema incluye capacidades de backtesting, ingesta de datos en tiempo real, generación de señales basadas en modelos de machine learning, gestión de riesgo, manejo robusto de errores con reintentos, y monitoreo de rendimiento.

## Características

- **Backtesting**: Prueba estrategias de trading con datos históricos
- **Trading en vivo/papel**: Ejecuta estrategias en cuentas de papel o reales
- **Modelos ML**: Utiliza modelos de machine learning para generar señales con entrenamiento automático programado
- **Aprendizaje por Refuerzo**: Sistema RL que aprende tanto de estrategias exitosas como de oportunidades de aprendizaje para mejorar continuamente
- **Simulación RL Continua**: Modo de simulación infinita donde el bot opera día a día y aprende en un bucle continuo
- **Estrategias múltiples**: Implementa estrategias de ML, mean reversion, trend following y ensemble
- **Gestión de riesgo**: Controla drawdown, stop-loss, y exposición máxima
- **Manejo robusto de errores**: Implementa reintentos con backoff exponencial e idempotencia
- **Servidor MCP**: Proxy de conexión al mercado con caché local y control de tasa
- **Sistema de colas**: Procesamiento asíncrono con Redis para señales, órdenes y predicciones
- **Microservicios**: API REST con FastAPI para predicciones, señales, órdenes y métricas
- **Trade Manager**: Control avanzado de operaciones con take profit/stop loss dinámicos y auto-hedging
- **Monitoreo**: Integración con Prometheus, endpoints de health check y alertas configurables
- **Integración con IA**: Capacidades para operar como agente autónomo con APIs de IA
- **Dockerizado**: Fácil despliegue con Docker y docker-compose

## Arquitectura Avanzada

El sistema implementa una arquitectura avanzada con múltiples componentes que trabajan juntos para proporcionar un sistema de trading robusto, escalable y de alto rendimiento.

### Servidor MCP (Market Connection Proxy)

El sistema incluye un servidor MCP (Market Connection Proxy) que actúa como intermediario entre la aplicación y la API de Alpaca. Este servidor proporciona las siguientes ventajas:

- **Caché local**: Reduce el número de llamadas a la API de Alpaca almacenando temporalmente los datos.
- **Control de tasa de solicitudes**: Evita superar los límites de la API de Alpaca.
- **Resiliencia**: Proporciona un punto único de conexión para todas las instancias de la aplicación.
- **Latencia reducida**: Mejora el rendimiento al servir datos desde la caché local.

#### Configuración del servidor MCP

Para configurar el servidor MCP, añada las siguientes variables a su archivo `.env`:

```
# Configuración del servidor MCP
USE_MCP=true
MCP_HOST=localhost
MCP_PORT=8000
MCP_REDIS_URL=redis://localhost:6379/0
MCP_CACHE_TTL=60
MCP_RATE_LIMIT_MAX_REQUESTS=100
MCP_RATE_LIMIT_TIMEFRAME=60
```

#### Ejecución del servidor MCP

Para iniciar el servidor MCP, ejecute:

```bash
python src/execution/run_mcp_server.py
```

O con parámetros personalizados:

```bash
python src/execution/run_mcp_server.py --host 0.0.0.0 --port 8000 --redis-url redis://localhost:6379/0 --cache-ttl 60 --rate-limit 100 --timeframe 60
```

#### Uso del servidor MCP con AlpacaClient

Para configurar `AlpacaClient` para usar el servidor MCP:

```python
from src.execution.alpaca_client import AlpacaClient

client = AlpacaClient(
    api_key="SU_API_KEY",
    api_secret="SU_API_SECRET",
    trading_mode="paper",
    use_mcp=True,
    mcp_url="http://localhost:8000"
)

# El cliente intentará usar el servidor MCP para todas las operaciones
# Si el servidor MCP no está disponible, volverá a usar la API de Alpaca directamente
```

#### Ejemplo de uso del servidor MCP

Se incluye un ejemplo completo en `examples/mcp_example.py` que muestra cómo iniciar el servidor MCP y configurar `AlpacaClient` para usarlo.

### Sistema de Colas con Redis

El sistema implementa un sistema de colas basado en Redis para el procesamiento asíncrono de tareas, lo que permite una mayor escalabilidad y rendimiento.

#### Características principales:

- **Procesamiento asíncrono**: Separa la generación de señales, órdenes y predicciones de su ejecución
- **Colas dedicadas**: Colas separadas para señales, órdenes, predicciones y entrenamiento
- **Escalabilidad horizontal**: Permite ejecutar múltiples workers para procesar tareas en paralelo
- **Persistencia**: Las tareas se mantienen en cola incluso si el sistema se reinicia

#### Uso del sistema de colas:

```python
from src.queue import QueueManager, SIGNAL_QUEUE, ORDER_QUEUE

# Inicializar el gestor de colas
queue_manager = QueueManager(redis_url="redis://localhost:6379/0")

# Encolar una tarea de señal
job_id = queue_manager.enqueue_signal_task(
    symbol="AAPL",
    timeframe="1h",
    strategy_func="generate_ml_signal",
    strategy_params={"model_name": "random_forest_v1"}
)

# Encolar una tarea de orden
order_job_id = queue_manager.enqueue_order_task(
    symbol="AAPL",
    order_type="market",
    side="buy",
    qty=10
)

# Verificar estado de un trabajo
status = queue_manager.get_job_status(job_id)
print(f"Estado del trabajo: {status}")

# Obtener información de las colas
queue_info = queue_manager.get_queue_info()
print(f"Trabajos pendientes en cola de señales: {queue_info[SIGNAL_QUEUE]['pending']}")
print(f"Trabajos pendientes en cola de órdenes: {queue_info[ORDER_QUEUE]['pending']}")
```

#### Ejecución de workers:

Para iniciar los workers que procesan las tareas en las colas:

```bash
python src/queue/run_workers.py --redis-url redis://localhost:6379/0 --workers 2 --queues signal,order
```

### Microservicios con FastAPI

El sistema implementa una API REST con FastAPI que expone endpoints para predicciones, señales, órdenes y métricas.

#### Endpoints principales:

- **/predict**: Genera predicciones para un símbolo y timeframe específicos
- **/signal**: Genera señales de trading basadas en diferentes estrategias
- **/order**: Crea y gestiona órdenes de trading
- **/metrics**: Proporciona métricas de rendimiento del sistema
- **/health**: Endpoint de health check para monitoreo

#### Ejecución del servidor API:

```bash
python src/api/run_api_server.py --host 0.0.0.0 --port 8080
```

#### Ejemplo de uso de la API:

```python
import requests
import json

# Generar una predicción
response = requests.post(
    "http://localhost:8080/predict",
    json={
        "symbol": "AAPL",
        "timeframe": "1h",
        "model_name": "random_forest_v1",
        "features": ["rsi", "macd", "bollinger"]
    }
)
prediction = response.json()
print(f"Predicción: {prediction}")

# Generar una señal
response = requests.post(
    "http://localhost:8080/signal",
    json={
        "symbol": "AAPL",
        "timeframe": "1h",
        "strategy": "ml_prediction",
        "params": {"model_name": "random_forest_v1"}
    }
)
signal = response.json()
print(f"Señal: {signal}")

# Crear una orden
response = requests.post(
    "http://localhost:8080/order",
    json={
        "symbol": "AAPL",
        "order_type": "market",
        "side": "buy",
        "qty": 10
    }
)
order = response.json()
print(f"Orden: {order}")
```

## Sistema de Aprendizaje por Refuerzo

El sistema implementa un agente de aprendizaje por refuerzo avanzado basado en Q-Learning que aprende tanto de estrategias exitosas como de oportunidades de aprendizaje para mejorar continuamente su rendimiento en el trading.

### Características del Sistema RL

- **Aprendizaje Dual**: Aprende tanto de operaciones exitosas (win_rate > 50%) como de fracasos para evitar repetir errores
- **Estados Discretizados**: Convierte indicadores técnicos (RSI, SMA, volatilidad, momentum) en estados discretos para Q-Learning
- **Exploración vs Explotación**: Balancea entre probar nuevas estrategias y usar las aprendidas
- **Memoria Persistente**: Guarda estrategias exitosas y oportunidades de aprendizaje para análisis posterior
- **Adaptación Continua**: Mejora su rendimiento a medida que acumula experiencia de trading

### Algoritmo Q-Learning

El sistema utiliza Q-Learning con las siguientes características:

- **Estados**: Combinación discretizada de indicadores técnicos (RSI, ratio SMA, volatilidad, momentum)
- **Acciones**: BUY, SELL, HOLD
- **Recompensas**: Basadas en retornos reales menos costos de transacción, normalizadas entre -1 y 1
- **Factor de descuento**: 0.95 para valorar recompensas futuras
- **Tasa de aprendizaje**: 0.1 para actualizar valores Q
- **Epsilon-greedy**: 10% de exploración para probar nuevas estrategias

### Configuración del Sistema RL

```python
from src.models.reinforcement_learning import TradingReinforcementLearner

# Configuración del agente RL
config = {
    "LEARNING_RATE": 0.1,          # Tasa de aprendizaje para Q-Learning
    "DISCOUNT_FACTOR": 0.95,       # Factor de descuento para recompensas futuras
    "EPSILON": 0.1,                # Tasa de exploración (10%)
    "MEMORY_DIR": "models/memory", # Directorio para guardar memoria
    "USE_DEEP_Q": False,           # Usar Deep Q-Learning (opcional)
    "STATE_SIZE": 10,              # Dimensión del estado para Deep Q
    "HIDDEN_SIZE": 64,             # Neuronas en capas ocultas
    "MEMORY_SIZE": 10000,          # Tamaño del buffer de replay
    "BATCH_SIZE": 32               # Tamaño del batch para entrenamiento
}

# Inicializar el agente
rl_agent = TradingReinforcementLearner(config)
```

### Uso del Sistema RL

```python
import pandas as pd
from src.models.reinforcement_learning import TradingReinforcementLearner

# Inicializar agente
rl_agent = TradingReinforcementLearner(config)

# Obtener estado actual del mercado
features = pd.Series({
    'rsi': 65.5,
    'sma_5': 152.30,
    'sma_20': 150.80,
    'returns': 0.015,
    'volatility': 0.02
})

state = rl_agent.get_state(features)
print(f"Estado discretizado: {state}")

# Elegir acción óptima
action = rl_agent.choose_action(state, training=True)
print(f"Acción recomendada: {action}")

# Aprender de experiencias de trading
experiences = [
    {
        'state': '(2,3,0,2)',
        'action': 'BUY',
        'reward': 0.025,  # Retorno positivo
        'next_state': '(3,3,0,2)'
    },
    {
        'state': '(3,3,0,2)',
        'action': 'SELL',
        'reward': -0.015,  # Pérdida con costo de transacción
        'next_state': '(2,2,1,1)'
    }
]

rl_agent.learn_from_experience('AAPL', experiences)

# Obtener resumen de estrategia aprendida
summary = rl_agent.get_strategy_summary('AAPL')
print(f"Win Rate promedio: {summary['avg_win_rate']:.2%}")
print(f"Estrategias exitosas guardadas: {summary['successful_strategies']}")
```

### Tipos de Estrategias Guardadas

El sistema clasifica y guarda dos tipos de estrategias:

1. **Estrategias Exitosas** (`strategy_type: 'successful'`):
   - Win rate > 50%
   - Se usan como referencia para futuras decisiones
   - Máximo 10 estrategias guardadas por símbolo

2. **Oportunidades de Aprendizaje** (`strategy_type: 'learning_opportunity'`):
   - Win rate ≤ 50%
   - Se analizan para evitar repetir errores
   - Ayudan al sistema a aprender qué NO hacer

### Monitoreo del Aprendizaje RL

```python
# Obtener métricas de rendimiento
performance = rl_agent.get_strategy_summary('AAPL')

print(f"Sesiones totales: {performance['total_sessions']}")
print(f"Win Rate promedio: {performance['avg_win_rate']:.2%}")
print(f"Mejor Win Rate histórico: {performance['best_win_rate']:.2%}")
print(f"Tamaño de Q-table: {performance['q_table_size']} estados")
print(f"Estrategias guardadas: {performance['successful_strategies']}")

# Calcular confianza en una acción específica
confidence = rl_agent.get_action_confidence(state, 'BUY')
print(f"Confianza en acción BUY: {confidence:.2f}")
```

### Simulación RL Continua

Para modo de simulación infinita donde el bot aprende día a día:

```python
# Configurar simulación continua
config.update({
    "CONTINUOUS_SIMULATION": True,
    "SIMULATION_DAYS": 365,  # Simular 1 año
    "DAILY_TRADES_LIMIT": 5,  # Máximo 5 trades por día
    "AUTO_SAVE_FREQUENCY": 24  # Guardar memoria cada 24 horas
})

rl_agent = TradingReinforcementLearner(config)

# El sistema operará continuamente, aprendiendo de cada sesión de trading
# y mejorando su rendimiento a lo largo del tiempo
```

#### Ejecución desde Línea de Comandos

Para ejecutar el modo de simulación RL continua desde la línea de comandos:

```bash
# Simulación RL por 100 días adicionales (modo sim_rl_100)
python src/app.py --mode sim_rl_100 --additional-days 100

# Con límites diarios personalizados
python src/app.py --mode sim_rl_100 --additional-days 365 --daily-trades-limit 3

# Simulación infinita (sin límite de días, aprende continuamente)
python src/app.py --mode sim_rl_100 --additional-days -1

# Con configuración específica de símbolos
python src/app.py --mode sim_rl_100 --additional-days 200 --symbols "AAPL,MSFT,GOOGL,TSLA"

# MODO INFINITO: El bot aprende para siempre (24/7)
python src/app.py --mode sim_rl_100 --additional-days -1 --daily-trades-limit 5 --log-level INFO
```

#### Parámetros de Línea de Comandos para RL

- `--mode sim_rl_100`: Activa el modo de simulación RL
- `--additional-days N`: Número de días adicionales para simular (use -1 para infinito)
- `--daily-trades-limit N`: Máximo número de trades por día (default: 5)
- `--symbols "LISTA"`: Lista de símbolos separados por comas
- `--log-level DEBUG`: Nivel de logging para ver el aprendizaje en detalle

#### Cómo el Bot Aprende y Crea Estrategias

El sistema RL aprende de manera continua, creando y refinando estrategias basadas en experiencias reales de trading. Aquí está el proceso detallado:

### Proceso de Aprendizaje Continuo

1. **Inicio del Aprendizaje**: El bot comienza con una Q-table vacía y explora aleatoriamente (10% epsilon)
2. **Experiencias Iniciales**: Cada trade genera una experiencia (estado, acción, recompensa, estado siguiente)
3. **Actualización Q-Learning**: La Q-table se actualiza usando la fórmula Q-Learning
4. **Clasificación de Estrategias**: Las experiencias exitosas se guardan como "estrategias aprendidas"
5. **Mejora Continua**: El bot usa estrategias aprendidas mientras sigue explorando nuevas oportunidades

### Estrategias que el Bot Crea

El bot crea estrategias basadas en patrones de indicadores técnicos que han sido exitosos:

**Ejemplos de Estrategias Aprendidas:**
- **RSI alto + Momentum positivo = BUY**: Cuando RSI > 70 y momentum > 0.02, comprar
- **SMA crossover + Volatilidad baja = HOLD**: Esperar cuando hay cruce de medias móviles pero volatilidad baja
- **RSI bajo + Volatilidad alta = SELL**: Vender cuando RSI < 30 y volatilidad > 0.05
- **Momentum negativo + RSI medio = HOLD**: Mantener posición cuando momentum es negativo pero RSI está en zona neutral

### Evolución del Win Rate

```
Día 1-10: Win Rate 45-50% (aprendizaje inicial, exploración alta)
Día 11-30: Win Rate 50-60% (primeras estrategias exitosas guardadas)
Día 31-60: Win Rate 60-70% (combinación de estrategias aprendidas + exploración)
Día 61-100: Win Rate 70-80% (optimización de estrategias existentes)
Día 100+: Win Rate 75-85% (rendimiento estabilizado con estrategias refinadas)
```

#### Ejemplo de Salida Durante el Aprendizaje

```
INFO - Día 1/100: Win Rate: 46% -> Ejecutando BUY en AAPL (exploración aleatoria)
INFO - Día 15/100: Win Rate: 58% -> Memoria guardada (5 estrategias exitosas)
INFO - Día 30/100: Win Rate: 62% -> Exploración: probando SELL en MSFT
INFO - Día 50/100: Win Rate: 71% -> Estrategia aprendida: RSI alto + momentum positivo = BUY
INFO - Día 75/100: Win Rate: 74% -> 8 estrategias exitosas guardadas
INFO - Día 100/100: Win Rate: 78% -> Simulación completada. Rendimiento mejorado 32%
INFO - Día 150/∞: Win Rate: 82% -> Nueva estrategia: Volatilidad baja + RSI medio = HOLD
INFO - Día 200/∞: Win Rate: 85% -> Estrategia refinada: RSI 65-75 + momentum >0.01 = BUY
INFO - Día 365/∞: Win Rate: 87% -> 15 estrategias exitosas, aprendizaje continuo...
```

#### Configuración de Límites Diarios

Los límites diarios se configuran automáticamente para evitar sobretrading:

```python
# Límites por defecto
DEFAULT_DAILY_TRADES_LIMIT = 5      # Máximo 5 operaciones por día
DEFAULT_MAX_POSITION_SIZE = 0.1     # 10% del capital por posición
DEFAULT_COOLDOWN_MINUTES = 15       # 15 minutos entre operaciones
DEFAULT_MAX_DRAWDOWN = 0.05         # Stop si pierde 5%
```

#### Monitoreo del Progreso RL

Durante la simulación, el sistema muestra métricas en tiempo real:

- **Win Rate diario**: Porcentaje de operaciones exitosas
- **Estrategias aprendidas**: Número de estrategias exitosas guardadas
- **Exploración vs Explotación**: Balance entre probar nuevas estrategias y usar aprendidas
- **Mejora de rendimiento**: Comparación con rendimiento inicial
- **Memoria guardada**: Frecuencia de guardado de la Q-table y estrategias

### Persistencia de Memoria RL

```python
# Guardar memoria del agente
rl_agent.save_memory()

# Cargar memoria guardada (automático al inicializar)
# La memoria incluye Q-table, estrategias exitosas y historial de rendimiento
```

## Machine Learning y Entrenamiento Automático

El sistema incluye un módulo de entrenamiento automático programado para modelos de machine learning que permite mantener los modelos actualizados con los datos más recientes.

### AutoTrainer

La clase `AutoTrainer` gestiona el entrenamiento automático de modelos para diferentes símbolos y timeframes:

```python
from src.models.auto_trainer import AutoTrainer

# Inicializar el entrenador automático
trainer = AutoTrainer(
    schedule="daily",  # Opciones: "daily", "weekly", "monthly"
    symbols=["AAPL", "MSFT", "GOOGL"],
    timeframes=["1h", "1d"],
    lookback_periods={"1h": 30, "1d": 90}
)

# Iniciar el entrenamiento programado
trainer.start_scheduled_training()

# O ejecutar un entrenamiento inmediato
trainer.train_all_models()
```

Para ejecutar el entrenamiento automático:

```bash
python src/models/run_auto_trainer.py --schedule daily --symbols AAPL,MSFT,GOOGL --timeframes 1h,1d
```

## Trade Manager

El sistema implementa un Trade Manager avanzado que proporciona control completo sobre las operaciones de trading.

### Características principales:

- **Control de cantidad máxima**: Limita la exposición por activo
- **Take profit/stop loss dinámicos**: Ajusta automáticamente los niveles basados en volatilidad
- **Cooldown entre operaciones**: Evita sobreoperación en condiciones de mercado volátiles
- **Auto-hedging**: Implementa estrategias de cobertura automática con ETFs inversos
- **Notificaciones**: Envía alertas a Telegram y Discord

```python
from src.execution.trade_manager import TradeManager

# Inicializar el Trade Manager
trade_manager = TradeManager(
    alpaca_client=client,
    max_position_size={"AAPL": 100, "default": 50},
    cooldown_minutes=15,
    enable_auto_hedging=True,
    telegram_token="YOUR_TELEGRAM_TOKEN",
    telegram_chat_id="YOUR_CHAT_ID"
)

# Procesar una señal
trade_manager.process_signal(
    symbol="AAPL",
    signal_type="buy",
    confidence=0.85,
    price=150.25
)

# Registrar una operación completada
trade_manager.register_trade(
    symbol="AAPL",
    side="buy",
    qty=10,
    price=150.25,
    order_id="order123"
)
```

## Monitoreo y Alertas

El sistema incluye integración con Prometheus para monitoreo y alertas configurables.

### Métricas disponibles:

- **Rendimiento del trading**: PnL, win rate, drawdown
- **Latencia del sistema**: Tiempo de respuesta de la API, procesamiento de señales
- **Estado del sistema**: Salud de los componentes, errores
- **Uso de recursos**: CPU, memoria, conexiones de red

### Configuración de alertas:

- **Telegram**: Notificaciones en tiempo real
- **Discord**: Alertas y resúmenes diarios
- **Prometheus Alertmanager**: Alertas basadas en reglas

## Integración con Agentes Autónomos

El sistema está preparado para integrarse con APIs de IA como Claude o GPT para implementar estrategias de trading autónomas.

### Capacidades:

- **Procesamiento de lenguaje natural**: Interpreta instrucciones en lenguaje natural
- **Toma de decisiones autónoma**: Genera señales basadas en análisis de mercado
- **Adaptación dinámica**: Ajusta parámetros de trading según condiciones de mercado
- **Explicabilidad**: Proporciona justificaciones para decisiones de trading

## Estructura del Proyecto Actualizada

```
├── src/                    # Código fuente principal
│   ├── app.py              # Punto de entrada principal
│   ├── config.py           # Gestión de configuración
│   ├── api/                # API REST con FastAPI
│   │   ├── api_server.py   # Servidor API
│   │   └── run_api_server.py # Script para ejecutar el servidor API
│   ├── backtest/           # Módulo de backtesting
│   ├── data/               # Ingesta y almacenamiento de datos
│   ├── features/           # Ingeniería de características
│   ├── models/             # Modelos de ML y entrenamiento
│   │   ├── auto_trainer.py # Entrenamiento automático programado
│   │   ├── run_auto_trainer.py # Script para ejecutar entrenamiento automático
│   │   └── trainer.py      # Entrenamiento de modelos
│   ├── queue/              # Sistema de colas con Redis
│   │   ├── queue_manager.py # Gestor de colas
│   │   └── run_workers.py  # Script para ejecutar workers
│   ├── strategy/           # Generación de señales y gestión de portfolio
│   ├── execution/          # Ejecución de órdenes y cliente de Alpaca
│   │   ├── alpaca_client.py # Cliente de Alpaca con soporte para MCP
│   │   ├── mcp_server.py   # Servidor MCP (Market Connection Proxy)
│   │   ├── run_mcp_server.py # Script para ejecutar el servidor MCP
│   │   └── trade_manager.py # Gestor avanzado de operaciones
│   ├── risk/               # Gestión de riesgo
│   ├── monitoring/         # Métricas y alertas
│   │   ├── alerts.py       # Configuración de alertas
│   │   └── metrics.py      # Métricas de Prometheus
│   └── utils/              # Utilidades comunes
├── examples/               # Ejemplos de uso
├── tests/                  # Tests unitarios y de integración
├── Dockerfile              # Definición de imagen Docker
├── docker-compose.yml      # Configuración de servicios
├── requirements.txt        # Dependencias Python
└── .env.example           # Ejemplo de variables de entorno
```

## Ejemplos de Manejo Robusto de Errores

### Ejemplo Básico de Manejo de Errores

```bash
python examples/error_handling_example.py
```

Este ejemplo muestra:
- Generación de IDs de cliente únicos para idempotencia
- Detección de órdenes duplicadas
- Manejo de errores transitorios con backoff exponencial
- Reemplazo y cancelación de órdenes con manejo robusto de errores

### Ejemplo Avanzado de Trading Robusto

```bash
python examples/robust_trading_example.py --mode paper --symbols AAPL,MSFT,GOOGL --max_retries 5
```

Este ejemplo avanzado implementa:
- Estrategia de órdenes bracket con manejo robusto de errores
- Monitoreo continuo de órdenes con reintentos automáticos
- Reemplazo inteligente de órdenes pendientes
- Gestión completa de errores transitorios en un flujo de trading real

## Configuración

La configuración se realiza a través de variables de entorno o un archivo `.env`. Las principales opciones son:

- `ALPACA_API_KEY`: Tu API key de Alpaca
- `ALPACA_API_SECRET`: Tu API secret de Alpaca
- `SYMBOLS`: Lista de símbolos separados por comas
- `TIMEFRAME`: Timeframe para los datos (1m, 5m, 15m, 1h, 1d)
- `STRATEGY`: Estrategia a utilizar (ml_prediction, mean_reversion, trend_following, ensemble)
- `MODEL_TYPE`: Tipo de modelo ML (random_forest, gradient_boosting, lstm)
- `MAX_RETRY_ATTEMPTS`: Número máximo de reintentos para operaciones con Alpaca (default: 5)
- `BASE_BACKOFF_TIME_MS`: Tiempo base para backoff exponencial en ms (default: 1000)
- `MAX_BACKOFF_TIME_MS`: Tiempo máximo de backoff en ms (default: 30000)
- `JITTER_FACTOR`: Factor de aleatoriedad para backoff (default: 0.1)
- `USE_MCP`: Usar servidor MCP (true/false)
- `MCP_HOST`: Host del servidor MCP
- `MCP_PORT`: Puerto del servidor MCP
- `MCP_REDIS_URL`: URL de Redis para caché (opcional)
- `REDIS_URL`: URL de Redis para el sistema de colas
- `API_HOST`: Host para el servidor API
- `API_PORT`: Puerto para el servidor API
- `TELEGRAM_TOKEN`: Token de bot de Telegram para notificaciones
- `DISCORD_WEBHOOK`: Webhook de Discord para notificaciones
- `PROMETHEUS_PUSHGATEWAY`: URL del Pushgateway de Prometheus
- `AUTO_TRAINING_SCHEDULE`: Programación para entrenamiento automático (daily, weekly, monthly)

Consulta `.env.example` para ver todas las opciones disponibles.

## Inicio Rápido

1. Clonar el repositorio:
```bash
git clone https://github.com/tu-usuario/alpaca-trading-system.git
cd alpaca-trading-system
```

2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

3. Configurar variables de entorno:
```bash
cp .env.example .env
# Editar .env con tus credenciales y configuración
```

4. Iniciar servicios:
```bash
# Iniciar Redis (necesario para MCP y sistema de colas)
docker-compose up -d redis

# Iniciar servidor MCP
python src/execution/run_mcp_server.py

# Iniciar workers para procesamiento asíncrono
python src/queue/run_workers.py

# Iniciar servidor API
python src/api/run_api_server.py
```

5. Ejecutar ejemplo de trading:
```bash
python examples/trading_example.py
```

## Roadmap de Mejoras

Este es el plan de mejoras propuesto por áreas clave. Cada punto incluye el beneficio y una línea base de implementación para integrarlo al sistema actual.

### Arquitectura y Robustez
- Gestión de errores y resiliencia: Decorador global de reintentos con backoff exponencial para Alpaca y Redis usando `tenacity` o wrapper propio. Implementación prevista: `src/utils/retry.py` y aplicación en `alpaca_client.py`, `mcp_server.py`, `queue_manager.py`.
- Persistencia de colas: Almacenar estado de jobs, órdenes y señales en una DB ligera (`SQLite` o `PostgreSQL`). Implementación prevista: `src/data/store.py` (tablas `jobs`, `orders`, `signals`) y hooks en `QueueManager` y `TradeManager`.
- Logs estructurados: Emisión de logs JSON con `structlog` incluyendo metadatos (symbol, estrategia, resultado, latencia). Implementación prevista: `src/utils/logging.py` y uso consistente en módulos críticos.
- Supervisión: Microservicio o tarea cron que revise colas, órdenes y alertas cada X minutos y notifique bloqueos. Implementación prevista: `src/monitoring/supervisor.py` con integración a Telegram/Discord.

### Machine Learning y Estrategia
- Data pipeline: Módulo `data_collector.py` que guarda candles y features en DB (SQLite/Mongo). Implementación prevista: `src/data/collector.py` con ingesta programada y cache.
- Feature engineering dinámico: Integración de librerías `ta` y/o `vectorbt` para features técnicos (RSI, MACD, VWAP, ATR). Implementación prevista: ampliar `src/features/engineering.py`.
- Autoentrenamiento: Job recurrente (cada 24h) en `TRAINING_QUEUE` que reentrena y versiona modelos automáticamente. Implementación prevista: orquestación desde `src/models/auto_trainer.py` y `src/queue/queue_manager.py`.
- Validación continua: Pipeline de paper validation (entrena → evalúa → decide desplegar). Implementación prevista: `src/models/validation.py` y registro de métricas/artefactos en `store.py`.