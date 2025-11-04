#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ejemplo de uso del sistema de manejo robusto de errores con backoff exponencial e idempotencia
"""

import os
import sys
import logging
import time
from datetime import datetime

# Añadir el directorio raíz al path para importar los módulos del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config
from src.execution.alpaca_client import AlpacaClient
from src.execution.order_manager import OrderManager
from src.utils.logging import setup_logging

# Configurar logging
setup_logging(log_level="INFO", console=True, log_file="error_handling_example.log")
logger = logging.getLogger(__name__)

def main():
    """Ejemplo de uso del manejo robusto de errores."""
    # Cargar configuración
    config = load_config()
    
    # Configuración específica para el manejo de errores
    error_handling_config = {
        "MAX_RETRY_ATTEMPTS": 5,
        "BASE_BACKOFF_TIME_MS": 1000,
        "MAX_BACKOFF_TIME_MS": 30000,
        "JITTER_FACTOR": 0.1
    }
    
    # Actualizar la configuración con los parámetros de manejo de errores
    config.update(error_handling_config)
    
    # Inicializar cliente de Alpaca
    alpaca_client = AlpacaClient(
        api_key=config["ALPACA_API_KEY"],
        api_secret=config["ALPACA_API_SECRET"],
        base_url=config["ALPACA_BASE_URL"],
        data_url=config["ALPACA_DATA_URL"]
    )
    
    # Inicializar el gestor de órdenes con la configuración de manejo de errores
    order_manager = OrderManager(config, alpaca_client=alpaca_client)
    
    # Ejemplo 1: Envío de orden con idempotencia
    logger.info("\n\n=== Ejemplo 1: Envío de orden con idempotencia ===")
    try:
        # Generar un ID de cliente único para la orden
        client_order_id = order_manager._generate_client_order_id()
        logger.info(f"ID de cliente generado: {client_order_id}")
        
        # Enviar orden con el ID de cliente generado
        order = order_manager.submit_order(
            symbol="AAPL",
            qty=1,
            side="buy",
            type="market",
            time_in_force="day",
            client_order_id=client_order_id
        )
        logger.info(f"Orden enviada correctamente: {order.id}")
        
        # Intentar enviar la misma orden de nuevo (debería detectar la duplicación)
        logger.info("Intentando enviar la misma orden de nuevo...")
        duplicate_order = order_manager.submit_order(
            symbol="AAPL",
            qty=1,
            side="buy",
            type="market",
            time_in_force="day",
            client_order_id=client_order_id
        )
        logger.info(f"Resultado de la orden duplicada: {duplicate_order.id}")
    except Exception as e:
        logger.error(f"Error en el ejemplo 1: {str(e)}")
    
    # Ejemplo 2: Manejo de errores transitorios con backoff exponencial
    logger.info("\n\n=== Ejemplo 2: Manejo de errores transitorios con backoff exponencial ===")
    try:
        # Simular un error transitorio (esto es solo para demostración)
        # En un caso real, los errores transitorios ocurrirían naturalmente debido a
        # problemas de red, límites de tasa, etc.
        logger.info("Obteniendo órdenes abiertas con posible error transitorio...")
        open_orders = order_manager.get_open_orders()
        logger.info(f"Órdenes abiertas obtenidas: {len(open_orders)}")
    except Exception as e:
        logger.error(f"Error en el ejemplo 2: {str(e)}")
    
    # Ejemplo 3: Reemplazo de orden con manejo de errores
    logger.info("\n\n=== Ejemplo 3: Reemplazo de orden con manejo de errores ===")
    try:
        # Primero creamos una orden límite
        limit_order = order_manager.submit_order(
            symbol="AAPL",
            qty=1,
            side="buy",
            type="limit",
            time_in_force="day",
            limit_price=150.0  # Ajustar según el precio actual
        )
        logger.info(f"Orden límite creada: {limit_order.id}")
        
        # Esperar un momento para asegurarnos de que la orden se ha procesado
        time.sleep(2)
        
        # Reemplazar la orden con una nueva orden límite
        replaced_order = order_manager.replace_order(
            order_id=limit_order.id,
            qty=1,
            limit_price=151.0  # Nuevo precio límite
        )
        logger.info(f"Orden reemplazada correctamente: {replaced_order.id}")
    except Exception as e:
        logger.error(f"Error en el ejemplo 3: {str(e)}")
    
    # Ejemplo 4: Cancelación de todas las órdenes con manejo de errores
    logger.info("\n\n=== Ejemplo 4: Cancelación de todas las órdenes con manejo de errores ===")
    try:
        # Cancelar todas las órdenes abiertas
        cancelled_orders = order_manager.cancel_all_orders()
        logger.info(f"Órdenes canceladas: {len(cancelled_orders)}")
    except Exception as e:
        logger.error(f"Error en el ejemplo 4: {str(e)}")

if __name__ == "__main__":
    main()
    logger.info("Ejemplo completado.")