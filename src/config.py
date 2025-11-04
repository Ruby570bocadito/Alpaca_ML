#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo de configuración para el sistema de trading.
Carga variables de entorno y secretos.
"""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Configurar logger
logger = logging.getLogger(__name__)


def load_config() -> Dict[str, Any]:
    """Carga la configuración del sistema desde variables de entorno.

    Returns:
        Dict[str, Any]: Diccionario con la configuración
    """
    config = {
        # Credenciales de Alpaca
        "ALPACA_API_KEY": os.getenv("ALPACA_API_KEY"),
        "ALPACA_API_SECRET": os.getenv("ALPACA_API_SECRET"),
        "ALPACA_BASE_URL": os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        
        # Configuración de Alpaca MCP Server
        "ALPACA_MCP_ENABLED": os.getenv("ALPACA_MCP_ENABLED", "false").lower() == "true",
        "ALPACA_MCP_HOST": os.getenv("ALPACA_MCP_HOST", "localhost"),
        "ALPACA_MCP_PORT": int(os.getenv("ALPACA_MCP_PORT", "5000")),
        
        # Configuración de trading
        "TRADING_MODE": os.getenv("TRADING_MODE", "paper"),
        "STRATEGY": os.getenv("STRATEGY", "ml_strategy"),
        "symbols": os.getenv("SYMBOLS", "AAPL,MSFT,GOOGL").split(","),
        "SYMBOLS": os.getenv("SYMBOLS", "AAPL,MSFT,GOOGL,AMZN,TSLA"),
        "MAX_POSITION_SIZE": float(os.getenv("MAX_POSITION_SIZE", "0.1")),
        "MAX_DRAWDOWN": float(os.getenv("MAX_DRAWDOWN", "0.05")),
        "TRADING_INTERVAL_SECONDS": int(os.getenv("TRADING_INTERVAL_SECONDS", "60")),
        "BACKTEST_START": os.getenv("BACKTEST_START", "2023-01-01"),
        "BACKTEST_END": os.getenv("BACKTEST_END", "2023-12-31"),
        "INITIAL_EQUITY": float(os.getenv("INITIAL_EQUITY", "100000")),
        
        # Configuración de base de datos
        "DB_HOST": os.getenv("DB_HOST", "localhost"),
        "DB_PORT": int(os.getenv("DB_PORT", "5432")),
        "DB_NAME": os.getenv("DB_NAME", "trading_bot"),
        "DB_USER": os.getenv("DB_USER", "postgres"),
        "DB_PASSWORD": os.getenv("DB_PASSWORD", "password"),
        
        # Configuración de monitoreo y logging
        "SENTRY_DSN": os.getenv("SENTRY_DSN", ""),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
        "LOG_FORMAT": os.getenv("LOG_FORMAT", "text"),  # text | json
        "USE_STRUCTLOG": os.getenv("USE_STRUCTLOG", "false").lower() == "true",
        "LOG_TO_CONSOLE": os.getenv("LOG_TO_CONSOLE", "true"),
        "LOG_TO_FILE": os.getenv("LOG_TO_FILE", "true"),
        "LOG_DIR": os.getenv("LOG_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")),
        "LOG_FILE": os.getenv("LOG_FILE", "trading_bot.log"),
        "LOG_MAX_SIZE": os.getenv("LOG_MAX_SIZE", "10485760"),
        "LOG_BACKUP_COUNT": os.getenv("LOG_BACKUP_COUNT", "5"),
        
        # Configuración de entrenamiento automático y validación de modelos
        "TRAINING_SCHEDULE": os.getenv("TRAINING_SCHEDULE", "weekly"),
        "TRAINING_SYMBOLS": os.getenv("TRAINING_SYMBOLS", "SPY,QQQ,AAPL,MSFT,GOOGL"),
        "TRAINING_TIMEFRAMES": os.getenv("TRAINING_TIMEFRAMES", "1D,1H"),
        "TRAINING_LOOKBACK_DAYS": os.getenv("TRAINING_LOOKBACK_DAYS", "365"),
        "MODEL_MIN_F1": float(os.getenv("MODEL_MIN_F1", "0.55")),
        "MODEL_MIN_ACCURACY": float(os.getenv("MODEL_MIN_ACCURACY", "0.55")),
        "MODEL_MAX_MSE": float(os.getenv("MODEL_MAX_MSE", "0.25")),
        "MODEL_ACTIVATE_ON_TRAIN": os.getenv("MODEL_ACTIVATE_ON_TRAIN", "true").lower() == "true",

        # Configuración de aprendizaje continuo (modo learn)
        "LEARN_RETRAIN_INTERVAL_HOURS": int(os.getenv("LEARN_RETRAIN_INTERVAL_HOURS", "24")),
        "LEARN_EVALUATION_WINDOW_DAYS": int(os.getenv("LEARN_EVALUATION_WINDOW_DAYS", "30")),
        "LEARN_MIN_TRADES_FOR_RETRAIN": int(os.getenv("LEARN_MIN_TRADES_FOR_RETRAIN", "10")),
        "LEARN_MODEL_VERSIONS_TO_KEEP": int(os.getenv("LEARN_MODEL_VERSIONS_TO_KEEP", "5")),
        "LEARN_PERFORMANCE_THRESHOLD": float(os.getenv("LEARN_PERFORMANCE_THRESHOLD", "0.6")),

        # Configuración de noticias y sentimiento
        "NEWSAPI_KEY": os.getenv("NEWSAPI_KEY"),
        "USE_NEWS_SENTIMENT": os.getenv("USE_NEWS_SENTIMENT", "true").lower() == "true",
        "NEWS_SOURCES": os.getenv("NEWS_SOURCES", "bloomberg,reuters,cnbc,financial-times"),
        "NEWS_LANGUAGE": os.getenv("NEWS_LANGUAGE", "en"),
        "NEWS_SORT_BY": os.getenv("NEWS_SORT_BY", "publishedAt"),
        "MAX_ARTICLES_PER_REQUEST": int(os.getenv("MAX_ARTICLES_PER_REQUEST", "100")),
        "NEWS_CACHE_EXPIRY_HOURS": int(os.getenv("NEWS_CACHE_EXPIRY_HOURS", "1")),
        "NEWS_CACHE_DIR": os.getenv("NEWS_CACHE_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "news_cache")),
        "MIN_TEXT_LENGTH": int(os.getenv("MIN_TEXT_LENGTH", "10")),
        "USE_VADER": os.getenv("USE_VADER", "true").lower() == "true",
        "COMPOUND_THRESHOLD": float(os.getenv("COMPOUND_THRESHOLD", "0.05")),
        "AGGREGATION_WINDOW_HOURS": int(os.getenv("AGGREGATION_WINDOW_HOURS", "24")),
        
        # Seguridad y control API
        "JWT_SECRET": os.getenv("JWT_SECRET", ""),
        
        # Rutas de archivos
        "DATA_DIR": os.getenv("DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")),
        "MODELS_DIR": os.getenv("MODELS_DIR", os.path.join(os.path.dirname(__file__), "models", "artifacts")),
        "LOGS_DIR": os.getenv("LOGS_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")),
    }
    
    # Validar configuración crítica
    _validate_config(config)
    
    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """Valida la configuración crítica.

    Args:
        config: Diccionario con la configuración

    Raises:
        ValueError: Si falta alguna configuración crítica
    """
    # Validar credenciales de Alpaca
    if not config["ALPACA_API_KEY"] or not config["ALPACA_API_SECRET"]:
        logger.warning("Credenciales de Alpaca no configuradas. Algunas funcionalidades pueden no estar disponibles.")
    
    # Validar símbolos
    if not config["SYMBOLS"]:
        raise ValueError("No se han configurado símbolos para operar")
    
    # Validar directorios
    for dir_key in ["DATA_DIR", "MODELS_DIR", "LOGS_DIR"]:
        os.makedirs(config[dir_key], exist_ok=True)
    
    logger.info("Configuración validada correctamente")


def get_config_value(key: str, default: Optional[Any] = None) -> Any:
    """Obtiene un valor de configuración específico.

    Args:
        key: Clave de configuración
        default: Valor por defecto si no existe

    Returns:
        Any: Valor de configuración
    """
    config = load_config()
    return config.get(key, default)