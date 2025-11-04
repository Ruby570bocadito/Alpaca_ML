"""
API Server con FastAPI para endpoints de predicción, señales, órdenes y métricas.
"""
import os
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn
from dotenv import load_dotenv
import logging
from datetime import datetime
import sys

# Añadir directorio raíz al path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importar componentes del sistema
from src.queue.queue_manager import QueueManager, SIGNAL_QUEUE, PREDICTION_QUEUE, ORDER_QUEUE
from src.execution.alpaca_client import AlpacaClient

# Cargar variables de entorno
load_dotenv()

# Configuración
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Trading System API",
    description="API para sistema de trading algorítmico con Alpaca",
    version="1.0.0"
)

# Inicializar Queue Manager
queue_manager = QueueManager(redis_url=REDIS_URL)

# Modelos de datos
class PredictionRequest(BaseModel):
    symbol: str
    timeframe: str = "1D"
    features: Optional[List[str]] = None
    model_id: Optional[str] = "default"

class SignalRequest(BaseModel):
    symbol: str
    timeframe: str = "1D"
    strategy: str = "default"
    parameters: Optional[Dict[str, Any]] = None

class OrderRequest(BaseModel):
    symbol: str
    qty: float
    side: str = Field(..., description="buy o sell")
    type: str = Field("market", description="market, limit, stop, stop_limit")
    time_in_force: str = "day"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    client_order_id: Optional[str] = None
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None

# Endpoints de salud
@app.get("/health")
async def health_check():
    """Endpoint para verificar la salud del sistema."""
    # Verificar conexión a Redis
    try:
        redis_status = queue_manager.redis_conn.ping()
    except Exception as e:
        redis_status = False
        logger.error(f"Error de conexión a Redis: {str(e)}")
    
    # Verificar colas
    queue_info = queue_manager.get_queue_info()
    
    # Estado general
    status = "healthy" if redis_status else "degraded"
    
    return {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "up",
            "redis": "up" if redis_status else "down",
            "queues": queue_info
        }
    }

# Endpoints de predicción
@app.post("/predict")
async def create_prediction(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """Endpoint para generar predicciones de precios."""
    try:
        # Encolar tarea de predicción (aquí se simula la función que se ejecutaría)
        job_id = queue_manager.enqueue_task(
            PREDICTION_QUEUE,
            lambda x, y, z, w: {"symbol": x, "timeframe": y, "model": z, "features": w},
            request.symbol,
            request.timeframe,
            request.model_id,
            request.features
        )
        
        if not job_id:
            raise HTTPException(status_code=500, detail="Error al encolar tarea de predicción")
        
        return {
            "job_id": job_id,
            "status": "enqueued",
            "message": f"Predicción para {request.symbol} encolada correctamente"
        }
    except Exception as e:
        logger.error(f"Error en endpoint /predict: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoints de señales
@app.post("/signal")
async def create_signal(request: SignalRequest):
    """Endpoint para generar señales de trading."""
    try:
        # Encolar tarea de generación de señal
        job_id = queue_manager.enqueue_task(
            SIGNAL_QUEUE,
            lambda x, y, z, w: {"symbol": x, "timeframe": y, "strategy": z, "parameters": w},
            request.symbol,
            request.timeframe,
            request.strategy,
            request.parameters
        )
        
        if not job_id:
            raise HTTPException(status_code=500, detail="Error al encolar tarea de señal")
        
        return {
            "job_id": job_id,
            "status": "enqueued",
            "message": f"Generación de señal para {request.symbol} encolada correctamente"
        }
    except Exception as e:
        logger.error(f"Error en endpoint /signal: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoints de órdenes
@app.post("/order")
async def submit_order(request: OrderRequest):
    """Endpoint para enviar órdenes de trading."""
    try:
        # Encolar tarea de envío de orden
        job_id = queue_manager.enqueue_task(
            ORDER_QUEUE,
            lambda x, y, z, w, a, b, c, d, e: {
                "symbol": x, "qty": y, "side": z, "type": w,
                "time_in_force": a, "limit_price": b, "stop_price": c,
                "client_order_id": d, "take_profit_stop_loss": e
            },
            request.symbol,
            request.qty,
            request.side,
            request.type,
            request.time_in_force,
            request.limit_price,
            request.stop_price,
            request.client_order_id,
            {"take_profit": request.take_profit, "stop_loss": request.stop_loss}
        )
        
        if not job_id:
            raise HTTPException(status_code=500, detail="Error al encolar tarea de orden")
        
        return {
            "job_id": job_id,
            "status": "enqueued",
            "message": f"Orden para {request.symbol} encolada correctamente"
        }
    except Exception as e:
        logger.error(f"Error en endpoint /order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint para verificar estado de trabajos
@app.get("/job/{job_id}")
async def get_job_status(job_id: str):
    """Endpoint para verificar el estado de un trabajo."""
    try:
        status = queue_manager.get_job_status(job_id)
        return status
    except Exception as e:
        logger.error(f"Error al obtener estado del trabajo {job_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=f"Trabajo no encontrado: {job_id}")

# Endpoint para métricas
@app.get("/metrics")
async def get_metrics():
    """Endpoint para obtener métricas del sistema."""
    # Aquí se implementaría la lógica para obtener métricas
    # Por ahora devolvemos datos de ejemplo
    return {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "queue_depth": sum(q["jobs"] for q in queue_manager.get_queue_info().values()),
            "failed_jobs": sum(q["failed"] for q in queue_manager.get_queue_info().values()),
            "active_workers": sum(q["workers"] for q in queue_manager.get_queue_info().values())
        },
        "trading": {
            "signals_generated_today": 42,
            "orders_executed_today": 15,
            "success_rate": 0.95
        }
    }

def start_server():
    """Inicia el servidor FastAPI."""
    uvicorn.run(app, host=API_HOST, port=API_PORT)

if __name__ == "__main__":
    start_server()