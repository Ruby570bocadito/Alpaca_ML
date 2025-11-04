#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ejemplo avanzado de trading con manejo robusto de errores, backoff exponencial e idempotencia
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime, timedelta

# Añadir el directorio raíz al path para importar los módulos del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config
from src.data.ingest import DataIngestionManager
from src.execution.alpaca_client import AlpacaClient
from src.execution.order_manager import OrderManager
from src.risk.risk_manager import RiskManager
from src.utils.logging import setup_logging
from src.utils.time_utils import is_market_hours, get_next_market_open

# Configurar logging
setup_logging(log_level="INFO", console=True, log_file="robust_trading.log")
logger = logging.getLogger(__name__)

def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Ejemplo de trading robusto con Alpaca")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["paper", "live"], 
        default="paper",
        help="Modo de trading: paper o live"
    )
    parser.add_argument(
        "--symbols", 
        type=str, 
        default="AAPL,MSFT,GOOGL",
        help="Lista de símbolos separados por comas"
    )
    parser.add_argument(
        "--max_retries", 
        type=int, 
        default=5,
        help="Número máximo de reintentos para operaciones con Alpaca"
    )
    return parser.parse_args()

def setup_components(config):
    """Configura los componentes necesarios para el trading."""
    # Inicializar cliente de Alpaca
    alpaca_client = AlpacaClient(
        api_key=config["ALPACA_API_KEY"],
        api_secret=config["ALPACA_API_SECRET"],
        base_url=config["ALPACA_BASE_URL"],
        data_url=config["ALPACA_DATA_URL"]
    )
    
    # Verificar conexión con Alpaca
    try:
        account = alpaca_client.get_account()
        logger.info(f"Conectado a Alpaca. Cuenta: {account.id}, Saldo: ${float(account.equity):.2f}")
    except Exception as e:
        logger.error(f"Error al conectar con Alpaca: {str(e)}")
        sys.exit(1)
    
    # Inicializar componentes con manejo robusto de errores
    data_manager = DataIngestionManager(config, alpaca_client=alpaca_client)
    order_manager = OrderManager(config, alpaca_client=alpaca_client)
    risk_manager = RiskManager(config, alpaca_client=alpaca_client, order_manager=order_manager)
    
    return alpaca_client, data_manager, order_manager, risk_manager

def execute_bracket_order_strategy(symbols, order_manager, risk_manager):
    """Ejecuta una estrategia simple de órdenes bracket con manejo robusto de errores."""
    logger.info("Ejecutando estrategia de órdenes bracket...")
    
    for symbol in symbols:
        try:
            # Verificar si ya tenemos posiciones abiertas para este símbolo
            positions = order_manager.alpaca_client.list_positions()
            if any(p.symbol == symbol for p in positions):
                logger.info(f"Ya existe una posición abierta para {symbol}, omitiendo...")
                continue
            
            # Obtener datos de mercado actuales
            current_quote = order_manager.alpaca_client.get_latest_quote(symbol)
            current_price = (current_quote.bidprice + current_quote.askprice) / 2
            
            # Calcular parámetros para la orden bracket
            qty = risk_manager.calculate_position_size(symbol, current_price)
            take_profit_price = current_price * 1.02  # 2% de ganancia
            stop_loss_price = current_price * 0.99   # 1% de pérdida
            
            logger.info(f"Creando orden bracket para {symbol} a precio {current_price:.2f}")
            logger.info(f"Cantidad: {qty}, Take Profit: {take_profit_price:.2f}, Stop Loss: {stop_loss_price:.2f}")
            
            # Crear orden bracket con manejo robusto de errores
            bracket_order = order_manager.create_bracket_order(
                symbol=symbol,
                qty=qty,
                side="buy",
                take_profit_limit_price=take_profit_price,
                stop_loss_stop_price=stop_loss_price,
                stop_loss_limit_price=stop_loss_price * 0.99
            )
            
            logger.info(f"Orden bracket creada exitosamente para {symbol}")
            
        except Exception as e:
            logger.error(f"Error al procesar {symbol}: {str(e)}")

def monitor_and_manage_orders(order_manager, interval_seconds=60, duration_minutes=60):
    """Monitorea y gestiona órdenes con manejo robusto de errores."""
    logger.info(f"Iniciando monitoreo de órdenes por {duration_minutes} minutos...")
    
    end_time = datetime.now() + timedelta(minutes=duration_minutes)
    
    while datetime.now() < end_time:
        try:
            # Obtener órdenes abiertas con manejo robusto de errores
            open_orders = order_manager.get_open_orders()
            logger.info(f"Órdenes abiertas: {len(open_orders)}")
            
            # Procesar órdenes pendientes en cola
            order_manager.process_pending_orders()
            
            # Verificar estado de órdenes
            for order in open_orders:
                # Obtener estado actualizado de la orden
                updated_order = order_manager.get_order_status(order.id)
                logger.info(f"Orden {updated_order.id} para {updated_order.symbol}: {updated_order.status}")
                
                # Si la orden lleva mucho tiempo pendiente, considerar reemplazarla
                if updated_order.status == "new" and \
                   (datetime.now() - updated_order.submitted_at).total_seconds() > 300:  # 5 minutos
                    
                    if updated_order.type == "limit":
                        # Obtener precio actual
                        current_quote = order_manager.alpaca_client.get_latest_quote(updated_order.symbol)
                        current_price = (current_quote.bidprice + current_quote.askprice) / 2
                        
                        # Ajustar el precio límite para aumentar probabilidad de ejecución
                        new_limit_price = current_price if updated_order.side == "buy" else current_price * 0.99
                        
                        logger.info(f"Reemplazando orden {updated_order.id} con nuevo precio: {new_limit_price:.2f}")
                        
                        # Reemplazar orden con manejo robusto de errores
                        order_manager.replace_order(
                            order_id=updated_order.id,
                            qty=updated_order.qty,
                            limit_price=new_limit_price
                        )
            
            # Esperar antes de la siguiente verificación
            time.sleep(interval_seconds)
            
        except Exception as e:
            logger.error(f"Error en el monitoreo de órdenes: {str(e)}")
            # Esperar un poco antes de reintentar
            time.sleep(10)

def main():
    """Función principal del ejemplo de trading robusto."""
    # Parsear argumentos
    args = parse_args()
    
    # Cargar configuración
    config = load_config()
    
    # Actualizar configuración con argumentos
    config["TRADING_MODE"] = args.mode
    config["SYMBOLS"] = args.symbols
    config["MAX_RETRY_ATTEMPTS"] = args.max_retries
    config["BASE_BACKOFF_TIME_MS"] = 1000
    config["MAX_BACKOFF_TIME_MS"] = 30000
    config["JITTER_FACTOR"] = 0.1
    
    # Configurar componentes
    alpaca_client, data_manager, order_manager, risk_manager = setup_components(config)
    
    # Verificar si el mercado está abierto
    if not is_market_hours():
        next_open = get_next_market_open()
        logger.info(f"El mercado está cerrado. Próxima apertura: {next_open}")
        logger.info("Este ejemplo solo funciona durante horas de mercado.")
        return
    
    # Cancelar órdenes existentes para empezar limpio
    try:
        cancelled = order_manager.cancel_all_orders()
        logger.info(f"Canceladas {len(cancelled)} órdenes existentes")
    except Exception as e:
        logger.error(f"Error al cancelar órdenes existentes: {str(e)}")
    
    # Ejecutar estrategia de órdenes bracket
    symbols = config["SYMBOLS"].split(",")
    execute_bracket_order_strategy(symbols, order_manager, risk_manager)
    
    # Monitorear y gestionar órdenes
    monitor_and_manage_orders(order_manager, interval_seconds=30, duration_minutes=30)
    
    logger.info("Ejemplo de trading robusto completado.")

if __name__ == "__main__":
    main()