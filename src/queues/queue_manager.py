"""
Queue Manager para procesamiento asíncrono de señales y órdenes.
Utiliza Redis y RQ para gestionar colas de tareas.
"""
import os
import time
from typing import Any, Dict, List, Optional, Callable
import redis
from rq import Queue, Worker, Connection
from rq.job import Job
import logging
from dotenv import load_dotenv

# Configurar logging
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Configuración de Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
REDIS_QUEUE_TIMEOUT = int(os.getenv("REDIS_QUEUE_TIMEOUT", "180"))  # 3 minutos por defecto

# Nombres de colas
SIGNAL_QUEUE = "signals"
ORDER_QUEUE = "orders"
PREDICTION_QUEUE = "predictions"
TRAINING_QUEUE = "training"

class QueueManager:
    """Gestor de colas para procesamiento asíncrono de tareas."""
    
    def __init__(self, redis_url: str = REDIS_URL):
        """
        Inicializa el gestor de colas.
        
        Args:
            redis_url: URL de conexión a Redis
        """
        self.redis_conn = redis.from_url(redis_url)
        
        # Inicializar colas
        self.signal_queue = Queue(SIGNAL_QUEUE, connection=self.redis_conn)
        self.order_queue = Queue(ORDER_QUEUE, connection=self.redis_conn)
        self.prediction_queue = Queue(PREDICTION_QUEUE, connection=self.redis_conn)
        self.training_queue = Queue(TRAINING_QUEUE, connection=self.redis_conn)
        
        # Mapeo de colas por nombre
        self.queues = {
            SIGNAL_QUEUE: self.signal_queue,
            ORDER_QUEUE: self.order_queue,
            PREDICTION_QUEUE: self.prediction_queue,
            TRAINING_QUEUE: self.training_queue
        }
        
        logger.info(f"QueueManager inicializado con Redis en {redis_url}")
    
    def enqueue_task(self, queue_name: str, task_func: Callable, 
                    *args, **kwargs) -> Optional[str]:
        """
        Encola una tarea para ejecución asíncrona.
        
        Args:
            queue_name: Nombre de la cola (signals, orders, predictions, training)
            task_func: Función a ejecutar
            *args, **kwargs: Argumentos para la función
            
        Returns:
            ID del trabajo encolado o None si hay error
        """
        if queue_name not in self.queues:
            logger.error(f"Cola no válida: {queue_name}")
            return None
            
        try:
            job = self.queues[queue_name].enqueue(
                task_func, 
                *args, 
                **kwargs,
                job_timeout=REDIS_QUEUE_TIMEOUT
            )
            logger.info(f"Tarea encolada en {queue_name}, job_id: {job.id}")
            return job.id
        except Exception as e:
            logger.error(f"Error al encolar tarea: {str(e)}")
            return None
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Obtiene el estado de un trabajo.
        
        Args:
            job_id: ID del trabajo
            
        Returns:
            Diccionario con información del estado del trabajo
        """
        try:
            job = Job.fetch(job_id, connection=self.redis_conn)
            result = None
            
            if job.is_finished:
                result = job.result
                
            return {
                "id": job_id,
                "status": job.get_status(),
                "queue": job.origin,
                "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "ended_at": job.ended_at.isoformat() if job.ended_at else None,
                "result": result,
                "exc_info": job.exc_info
            }
        except Exception as e:
            logger.error(f"Error al obtener estado del trabajo {job_id}: {str(e)}")
            return {"id": job_id, "status": "error", "error": str(e)}
    
    def get_queue_info(self) -> Dict[str, Dict[str, int]]:
        """
        Obtiene información sobre el estado de todas las colas.
        
        Returns:
            Diccionario con información de cada cola
        """
        info = {}
        for name, queue in self.queues.items():
            info[name] = {
                "jobs": queue.count,
                "failed": queue.failed_job_registry.count,
                "workers": len(Worker.all(queue=queue))
            }
        return info

# Ejemplo de uso:
# queue_manager = QueueManager()
# job_id = queue_manager.enqueue_task(SIGNAL_QUEUE, generate_trading_signal, symbol="AAPL")
# status = queue_manager.get_job_status(job_id)