#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para entrenar modelos de ML para el sistema de trading
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta

# Añadir el directorio raíz al path para importar los módulos del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config
from src.data.ingest import DataIngestionManager
from src.features.engineering import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.utils.logging import setup_logging

# Configurar logging
setup_logging(log_level="INFO", console=True, log_file="model_training.log")
logger = logging.getLogger(__name__)

def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Entrenamiento de modelos ML para trading")
    parser.add_argument(
        "--config", 
        type=str, 
        default=".env",
        help="Ruta al archivo de configuración"
    )
    parser.add_argument(
        "--symbols", 
        type=str, 
        help="Lista de símbolos separados por comas (sobreescribe la configuración)"
    )
    parser.add_argument(
        "--start-date", 
        type=str, 
        default="2020-01-01",
        help="Fecha de inicio para datos de entrenamiento (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date", 
        type=str, 
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Fecha de fin para datos de entrenamiento (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--model-type", 
        type=str, 
        choices=["random_forest", "gradient_boosting", "lstm"],
        help="Tipo de modelo a entrenar (sobreescribe la configuración)"
    )
    parser.add_argument(
        "--timeframe", 
        type=str, 
        choices=["1m", "5m", "15m", "1h", "1d"],
        help="Timeframe para los datos (sobreescribe la configuración)"
    )
    parser.add_argument(
        "--force", 
        action="store_true",
        help="Forzar reentrenamiento incluso si el modelo ya existe"
    )
    return parser.parse_args()

def main():
    # Parsear argumentos
    args = parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Sobreescribir configuración con argumentos
    if args.symbols:
        config["SYMBOLS"] = args.symbols
    
    if args.model_type:
        config["MODEL_TYPE"] = args.model_type
    
    if args.timeframe:
        config["TIMEFRAME"] = args.timeframe
    
    # Obtener lista de símbolos
    symbols = config["SYMBOLS"].split(",")
    timeframe = config["TIMEFRAME"]
    model_type = config["MODEL_TYPE"]
    start_date = args.start_date
    end_date = args.end_date
    
    logger.info(f"Iniciando entrenamiento de modelos para {len(symbols)} símbolos")
    logger.info(f"Período: {start_date} a {end_date}")
    logger.info(f"Timeframe: {timeframe}")
    logger.info(f"Tipo de modelo: {model_type}")
    
    # Inicializar componentes
    data_manager = DataIngestionManager(config, mode="backtest")
    feature_engineer = FeatureEngineer(config)
    model_trainer = ModelTrainer(config)
    
    # Crear directorio para modelos si no existe
    model_dir = config.get("MODEL_DIR", "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Entrenar modelo para cada símbolo
    for symbol in symbols:
        try:
            logger.info(f"Procesando {symbol}...")
            
            # Verificar si el modelo ya existe
            model_path = os.path.join(model_dir, f"{symbol}_{model_type}_{timeframe}.pkl")
            if os.path.exists(model_path) and not args.force:
                logger.info(f"El modelo para {symbol} ya existe. Usa --force para reentrenar.")
                continue
            
            # Obtener datos históricos
            logger.info(f"Obteniendo datos históricos para {symbol}...")
            df = data_manager.get_historical_bars(
                symbol=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date
            )
            
            if df is None or df.empty:
                logger.warning(f"No se pudieron obtener datos para {symbol}. Saltando.")
                continue
            
            logger.info(f"Datos obtenidos: {len(df)} barras")
            
            # Generar características
            logger.info(f"Generando características para {symbol}...")
            df_features = feature_engineer.generate_features(df, symbol)
            
            if df_features is None or df_features.empty:
                logger.warning(f"No se pudieron generar características para {symbol}. Saltando.")
                continue
            
            logger.info(f"Características generadas: {len(df_features.columns)} columnas")
            
            # Entrenar modelo
            logger.info(f"Entrenando modelo para {symbol}...")
            model, metrics = model_trainer.train_model(
                symbol=symbol,
                data=df_features,
                model_type=model_type,
                save_model=True
            )
            
            # Mostrar métricas
            logger.info(f"Entrenamiento completado para {symbol}")
            logger.info(f"Métricas de entrenamiento:")
            for metric_name, metric_value in metrics.items():
                logger.info(f"  {metric_name}: {metric_value}")
        
        except Exception as e:
            logger.error(f"Error al procesar {symbol}: {e}", exc_info=True)
    
    logger.info("Entrenamiento de modelos completado")

if __name__ == "__main__":
    main()