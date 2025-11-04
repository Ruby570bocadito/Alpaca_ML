# Sistema de Trading Algorítmico con Alpaca

Este proyecto implementa un sistema completo de trading algorítmico que utiliza la API de Alpaca para ejecutar operaciones en el mercado de valores. El sistema incluye capacidades de backtesting, ingesta de datos en tiempo real, generación de señales basadas en modelos de machine learning, gestión de riesgo, manejo robusto de errores con reintentos, y monitoreo de rendimiento.

## Características

- **Backtesting**: Prueba estrategias de trading con datos históricos
- **Trading en vivo/papel**: Ejecuta estrategias en cuentas de papel o reales
- **Modelos ML**: Utiliza modelos de machine learning para generar señales con entrenamiento automático programado
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

### Ejecución y Riesgo
- Smart Order Routing: Evaluar profundidad de mercado (L2, si Alpaca lo permite) para evitar slippage. Implementación prevista: extender `src/execution/order_manager.py` con reglas SOR.
- Trailing Stops: Stop loss dinámico por porcentaje/ATR en función del profit. Implementación prevista: `TradeManager` soporta trailing y actualización periódica.
- Rebalanceo automático: Job semanal que equilibre exposición por símbolo/sector. Implementación prevista: `src/strategy/portfolio.py` con tarea programada.
- Gestión de volatilidad: “Volatility filter” (ATR/desviación estándar) para limitar trades en alta volatilidad. Implementación prevista: filtros en `signals.py` y `trade_manager.py`.

### Integraciones útiles
- API de control (FastAPI): Endpoints para estado de colas, órdenes y envío de señales manuales. Implementación prevista: ampliar `src/api/api_server.py` con `/queues`, `/orders`, `/signals/manual`.
- Dashboard visual: `Streamlit` o frontend React conectado a la API para visualizar PnL, órdenes activas, señales, latencia. Implementación prevista: `dash/` (Streamlit) o `frontend/` (React) con despliegue vía Docker.
- Versionado de estrategias: Guardar cada estrategia/versión con métricas de performance. Implementación prevista: tabla `strategies` en `store.py` y utilidades en `strategy/`.

### Escalabilidad futura
- Orquestación: Docker Compose con servicios (Redis, Worker, API, ML) y health checks. Implementación prevista: ampliar `docker-compose.yml` con `healthcheck` y dependencias.
- Monitoreo de rendimiento: Prometheus + Grafana (o NewRelic) para CPU, latencia y resultados de colas. Implementación prevista: `src/monitoring/metrics.py`, dashboards y alertas.
- Failover automático: Redundancia de Redis + Backup Worker (supervisor). Implementación prevista: `supervisor` en despliegue o `pm2`/`watchdog` equivalente.

### Seguridad y autenticación
- Gestión segura de claves: Cargar claves Alpaca y Discord/Telegram desde `.env` cifrado o AWS Secrets Manager. Implementación prevista: soporte opcional en `config.py`.
- Verificación de señales externas: Firmar digitalmente señales vía API/MCP (JWT o HMAC). Implementación prevista: middleware en `api_server.py` y validación en `TradeManager`.

Estado actual: varias piezas están parcial o totalmente implementadas (API, colas, Trade Manager, autoentrenamiento, monitoreo básico). El resto queda priorizado para las próximas iteraciones.

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir cambios importantes antes de enviar un pull request.