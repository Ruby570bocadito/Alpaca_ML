"""
Script para ejecutar el entrenamiento automático de modelos.
"""
import os
import sys
import argparse
from dotenv import load_dotenv
import logging

# Añadir directorio raíz al path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importaciones locales
from src.models.auto_trainer import AutoTrainer
from src.config import load_config

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Función principal para ejecutar el entrenamiento automático."""
    parser = argparse.ArgumentParser(description='Ejecutar entrenamiento automático de modelos')
    parser.add_argument(
        '--schedule', 
        choices=['daily', 'weekly', 'monthly', 'now'],
        default=os.getenv("TRAINING_SCHEDULE", "weekly"),
        help='Frecuencia de entrenamiento (por defecto: weekly)'
    )
    parser.add_argument(
        '--symbols', 
        nargs='+',
        default=os.getenv("TRAINING_SYMBOLS", "SPY,QQQ,AAPL,MSFT,GOOGL").split(","),
        help='Símbolos para entrenar (por defecto: SPY,QQQ,AAPL,MSFT,GOOGL)'
    )
    parser.add_argument(
        '--timeframes', 
        nargs='+',
        default=os.getenv("TRAINING_TIMEFRAMES", "1D,1H").split(","),
        help='Timeframes para entrenar (por defecto: 1D,1H)'
    )
    parser.add_argument(
        '--lookback', 
        type=int,
        default=int(os.getenv("TRAINING_LOOKBACK_DAYS", "365")),
        help='Días de histórico para entrenamiento (por defecto: 365)'
    )
    parser.add_argument(
        '--models-dir', 
        default=os.getenv("MODELS_DIR", "models"),
        help='Directorio para guardar modelos (por defecto: models)'
    )
    
    args = parser.parse_args()
    
    # Cargar configuración
    config = load_config()

    # Inicializar AutoTrainer (internamente crea sus componentes)
    auto_trainer = AutoTrainer(
        models_dir=args.models_dir,
        training_schedule=args.schedule,
        symbols=args.symbols,
        timeframes=args.timeframes,
        lookback_days=args.lookback,
        config=config,
    )
    
    # Si se especifica 'now', entrenar inmediatamente
    if args.schedule == 'now':
        logger.info("Iniciando entrenamiento inmediato")
        results = auto_trainer.train_all_models()
        logger.info(f"Entrenamiento completado: {results['models_trained']} modelos entrenados, "
                   f"{results['errors']} errores")
    else:
        # Configurar y ejecutar scheduler
        logger.info(f"Iniciando scheduler con frecuencia: {args.schedule}")
        auto_trainer.run_scheduler()

if __name__ == '__main__':
    main()