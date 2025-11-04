"""
Script para ejecutar workers que procesan tareas de las colas.
"""
import os
import sys
import argparse
import redis
from rq import Worker, Queue, Connection
from dotenv import load_dotenv
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Añadir directorio raíz al path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importar definiciones de colas
from src.queue import SIGNAL_QUEUE, ORDER_QUEUE, PREDICTION_QUEUE, TRAINING_QUEUE

# Cargar variables de entorno
load_dotenv()

# Configuración de Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

def main():
    """Función principal para ejecutar los workers."""
    parser = argparse.ArgumentParser(description='Ejecutar workers para procesar tareas de las colas')
    parser.add_argument(
        '--queues', 
        nargs='+', 
        default=[SIGNAL_QUEUE, ORDER_QUEUE, PREDICTION_QUEUE, TRAINING_QUEUE],
        help='Colas a procesar (por defecto: todas)'
    )
    parser.add_argument(
        '--redis-url', 
        default=REDIS_URL,
        help=f'URL de conexión a Redis (por defecto: {REDIS_URL})'
    )
    parser.add_argument(
        '--workers', 
        type=int, 
        default=2,
        help='Número de workers por cola (por defecto: 2)'
    )
    
    args = parser.parse_args()
    
    # Validar colas
    valid_queues = [SIGNAL_QUEUE, ORDER_QUEUE, PREDICTION_QUEUE, TRAINING_QUEUE]
    for queue in args.queues:
        if queue not in valid_queues:
            logger.error(f"Cola no válida: {queue}")
            sys.exit(1)
    
    logger.info(f"Iniciando {args.workers} workers para las colas: {', '.join(args.queues)}")
    
    # Conectar a Redis y ejecutar workers
    with Connection(redis.from_url(args.redis_url)):
        queues = [Queue(queue_name) for queue_name in args.queues]
        workers = []
        
        for i in range(args.workers):
            worker = Worker(queues, name=f'worker-{i+1}')
            workers.append(worker)
            logger.info(f"Worker {worker.name} iniciado para {', '.join(args.queues)}")
        
        # Iniciar workers
        for worker in workers:
            worker.work(with_scheduler=True)

if __name__ == '__main__':
    main()