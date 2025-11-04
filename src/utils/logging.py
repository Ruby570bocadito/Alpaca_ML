#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuración y utilidades de logging para el sistema de trading.
"""

import logging
import logging.handlers
import os
import sys
import json
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Union, List
import structlog

# Configuración de colores para consola
COLORS = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Verde
    "WARNING": "\033[33m",   # Amarillo
    "ERROR": "\033[31m",     # Rojo
    "CRITICAL": "\033[35m",  # Magenta
    "RESET": "\033[0m"       # Reset
}


class ColoredFormatter(logging.Formatter):
    """Formateador de logs con colores para la consola."""

    def format(self, record):
        levelname = record.levelname
        if levelname in COLORS:
            record.levelname = f"{COLORS[levelname]}{levelname}{COLORS['RESET']}"
        return super().format(record)


class JSONFormatter(logging.Formatter):
    """Formateador de logs en formato JSON."""

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "line": record.lineno,
            "process": record.process,
            "thread": record.thread,
        }
        
        # Añadir excepción si existe
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Añadir datos extra
        if hasattr(record, "data") and record.data:
            log_data["data"] = record.data
        
        return json.dumps(log_data)


class TradeLogger(logging.Logger):
    """Logger personalizado para el sistema de trading."""

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
    
    def trade(self, msg, *args, **kwargs):
        """Log de operaciones de trading."""
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, f"[TRADE] {msg}", args, **kwargs)
    
    def signal(self, msg, *args, **kwargs):
        """Log de señales de trading."""
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, f"[SIGNAL] {msg}", args, **kwargs)
    
    def model(self, msg, *args, **kwargs):
        """Log de actividad del modelo."""
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, f"[MODEL] {msg}", args, **kwargs)
    
    def risk(self, msg, *args, **kwargs):
        """Log de gestión de riesgo."""
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, f"[RISK] {msg}", args, **kwargs)
    
    def metric(self, msg, *args, **kwargs):
        """Log de métricas."""
        if self.isEnabledFor(logging.DEBUG):
            self._log(logging.DEBUG, f"[METRIC] {msg}", args, **kwargs)
    
    def data(self, msg, *args, **kwargs):
        """Log de datos."""
        if self.isEnabledFor(logging.DEBUG):
            self._log(logging.DEBUG, f"[DATA] {msg}", args, **kwargs)
    
    def structured(self, msg, data=None, level=logging.INFO, *args, **kwargs):
        """Log estructurado con datos adicionales."""
        if self.isEnabledFor(level):
            record = self.makeRecord(
                self.name, level, kwargs.get("fn", ""), 
                kwargs.get("lno", 0), msg, args, 
                kwargs.get("exc_info", None),
                kwargs.get("func", None), kwargs.get("extra", None)
            )
            record.data = data
            self.handle(record)


# Registrar el logger personalizado
logging.setLoggerClass(TradeLogger)


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Configura el sistema de logging.

    Args:
        config: Configuración del sistema

    Returns:
        logging.Logger: Logger configurado
    """
    # Obtener configuración de logging
    log_level = config.get("LOG_LEVEL", "INFO").upper()
    log_dir = config.get("LOG_DIR", "logs")
    log_file = config.get("LOG_FILE", "trading_bot.log")
    log_to_console = config.get("LOG_TO_CONSOLE", "true").lower() == "true"
    log_to_file = config.get("LOG_TO_FILE", "true").lower() == "true"
    log_format = config.get("LOG_FORMAT", "text").lower()  # text o json
    use_structlog = str(config.get("USE_STRUCTLOG", "false")).lower() == "true"
    log_max_size = int(config.get("LOG_MAX_SIZE", "10485760"))  # 10MB
    log_backup_count = int(config.get("LOG_BACKUP_COUNT", "5"))
    
    # Crear directorio de logs si no existe
    if log_to_file and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Obtener logger raíz
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))
    
    # Limpiar handlers existentes
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Configurar formato según configuración
    if use_structlog:
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.stdlib.add_log_level,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.JSONRenderer(),
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        formatter = logging.Formatter("%(message)s")  # structlog ya renderiza JSON
    elif log_format == "json":
        formatter = JSONFormatter()
    else:
        log_format_str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        formatter = logging.Formatter(log_format_str)
    
    # Configurar handler de consola
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        # Configurar encoding UTF-8 para evitar errores con caracteres Unicode
        console_handler.stream.reconfigure(encoding='utf-8')
        logger.addHandler(console_handler)
    
    # Configurar handler de archivo
    if log_to_file:
        log_file_path = os.path.join(log_dir, log_file)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file_path,
            maxBytes=log_max_size,
            backupCount=log_backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Configurar logger específico para el bot
    bot_logger = logging.getLogger("trading_bot")
    if use_structlog:
        # Retornar un logger de structlog enlazado con nombre
        return structlog.get_logger("trading_bot")
    
    # Configurar loggers de librerías externas
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("websocket").setLevel(logging.WARNING)
    
    return bot_logger


def get_logger(name: str = "trading_bot") -> logging.Logger:
    """Obtiene un logger configurado.

    Args:
        name: Nombre del logger

    Returns:
        logging.Logger: Logger configurado
    """
    return logging.getLogger(name)


def log_exception(logger: logging.Logger, e: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Registra una excepción con contexto adicional.

    Args:
        logger: Logger a utilizar
        e: Excepción a registrar
        context: Contexto adicional (opcional)
    """
    if context is None:
        context = {}
    
    error_data = {
        "exception_type": type(e).__name__,
        "exception_message": str(e),
        **context
    }
    
    if isinstance(logger, TradeLogger):
        logger.structured(f"Error: {str(e)}", data=error_data, level=logging.ERROR, exc_info=True)
    else:
        logger.error(f"Error: {str(e)}", exc_info=True, extra={"data": error_data})


def log_trade(logger: logging.Logger, trade_data: Dict[str, Any]) -> None:
    """Registra información de una operación de trading.

    Args:
        logger: Logger a utilizar
        trade_data: Datos de la operación
    """
    symbol = trade_data.get("symbol", "UNKNOWN")
    side = trade_data.get("side", "UNKNOWN")
    qty = trade_data.get("qty", 0)
    price = trade_data.get("price", 0)
    
    message = f"{side.upper()} {symbol}: {qty} @ ${price:.2f}"
    
    if isinstance(logger, TradeLogger):
        logger.trade(message)
        logger.structured("Trade executed", data=trade_data, level=logging.INFO)
    else:
        logger.info(f"[TRADE] {message}")


def log_signal(logger: logging.Logger, signal_data: Dict[str, Any]) -> None:
    """Registra información de una señal de trading.

    Args:
        logger: Logger a utilizar
        signal_data: Datos de la señal
    """
    symbol = signal_data.get("symbol", "UNKNOWN")
    direction = signal_data.get("direction", "UNKNOWN")
    confidence = signal_data.get("confidence", 0)
    strategy = signal_data.get("strategy", "UNKNOWN")
    
    message = f"{symbol} {direction.upper()} signal ({confidence:.2f}) from {strategy}"
    
    if isinstance(logger, TradeLogger):
        logger.signal(message)
        logger.structured("Signal generated", data=signal_data, level=logging.INFO)
    else:
        logger.info(f"[SIGNAL] {message}")


def log_model_prediction(logger: logging.Logger, prediction_data: Dict[str, Any]) -> None:
    """Registra información de una predicción del modelo.

    Args:
        logger: Logger a utilizar
        prediction_data: Datos de la predicción
    """
    symbol = prediction_data.get("symbol", "UNKNOWN")
    prediction = prediction_data.get("prediction", 0)
    probability = prediction_data.get("probability", 0)
    model_name = prediction_data.get("model_name", "UNKNOWN")
    
    direction = "UP" if prediction > 0 else "DOWN"
    message = f"{symbol} predicted {direction} ({probability:.2f}) by {model_name}"
    
    if isinstance(logger, TradeLogger):
        logger.model(message)
        logger.structured("Model prediction", data=prediction_data, level=logging.DEBUG)
    else:
        logger.debug(f"[MODEL] {message}")


def log_risk_event(logger: logging.Logger, risk_data: Dict[str, Any]) -> None:
    """Registra información de un evento de riesgo.

    Args:
        logger: Logger a utilizar
        risk_data: Datos del evento de riesgo
    """
    event_type = risk_data.get("event_type", "UNKNOWN")
    message = risk_data.get("message", "Risk event")
    
    if isinstance(logger, TradeLogger):
        logger.risk(message)
        logger.structured("Risk event", data=risk_data, level=logging.WARNING)
    else:
        logger.warning(f"[RISK] {event_type}: {message}")


def log_performance_metrics(logger: logging.Logger, metrics: Dict[str, Any]) -> None:
    """Registra métricas de rendimiento.

    Args:
        logger: Logger a utilizar
        metrics: Métricas de rendimiento
    """
    # Formatear métricas principales
    equity = metrics.get("equity", 0)
    daily_pnl = metrics.get("daily_pnl", 0)
    total_pnl = metrics.get("total_pnl", 0)
    drawdown = metrics.get("drawdown", 0)
    sharpe = metrics.get("sharpe_ratio", 0)
    
    message = (f"Equity: ${equity:.2f} | Daily P&L: ${daily_pnl:.2f} | "
              f"Total P&L: ${total_pnl:.2f} | DD: {drawdown:.2f}% | Sharpe: {sharpe:.2f}")
    
    if isinstance(logger, TradeLogger):
        logger.metric(message)
        logger.structured("Performance metrics", data=metrics, level=logging.INFO)
    else:
        logger.info(f"[METRICS] {message}")
