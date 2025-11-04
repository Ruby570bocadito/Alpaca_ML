#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ejemplo de uso del servidor MCP (Market Connection Proxy) con AlpacaClient.
Este ejemplo muestra cómo iniciar el servidor MCP y configurar AlpacaClient para usarlo.
"""

import os
import sys
import time
import logging
import argparse
from dotenv import load_dotenv

# Añadir directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.execution.alpaca_client import AlpacaClient
from src.execution.mcp_server import create_mcp_server
import uvicorn
import threading

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_mcp_server(host, port):
    """Ejecuta el servidor MCP en un hilo separado."""
    mcp_app = create_mcp_server(
        alpaca_api_key=os.getenv("ALPACA_API_KEY"),
        alpaca_api_secret=os.getenv("ALPACA_API_SECRET"),
        redis_url=None,  # Sin caché Redis para este ejemplo
        cache_ttl=60,
        rate_limit_max_requests=100,
        rate_limit_timeframe=60
    )
    
    uvicorn.run(mcp_app, host=host, port=port)

def main():
    """Función principal del ejemplo."""
    # Cargar variables de entorno
    load_dotenv()
    
    # Verificar credenciales de Alpaca
    if not os.getenv("ALPACA_API_KEY") or not os.getenv("ALPACA_API_SECRET"):
        logger.error("Las credenciales de Alpaca no están configuradas en el archivo .env")
        sys.exit(1)
    
    # Configuración del servidor MCP
    host = "localhost"
    port = 8000
    mcp_url = f"http://{host}:{port}"
    
    # Iniciar servidor MCP en un hilo separado
    logger.info(f"Iniciando servidor MCP en {mcp_url}...")
    mcp_thread = threading.Thread(target=run_mcp_server, args=(host, port))
    mcp_thread.daemon = True
    mcp_thread.start()
    
    # Esperar a que el servidor esté listo
    time.sleep(2)
    
    try:
        # Crear cliente Alpaca con MCP
        logger.info("Creando cliente Alpaca con MCP...")
        client = AlpacaClient(
            api_key=os.getenv("ALPACA_API_KEY"),
            api_secret=os.getenv("ALPACA_API_SECRET"),
            trading_mode="paper",
            use_mcp=True,
            mcp_url=mcp_url
        )
        
        # Verificar conexión
        logger.info("Verificando conexión...")
        account = client.get_account()
        logger.info(f"Cuenta: {account.get('id')}, Status: {account.get('status')}")
        logger.info(f"Equity: ${account.get('equity')}, Cash: ${account.get('cash')}")
        
        # Obtener posiciones
        logger.info("Obteniendo posiciones...")
        positions = client.get_positions()
        logger.info(f"Número de posiciones: {len(positions)}")
        
        # Obtener órdenes abiertas
        logger.info("Obteniendo órdenes abiertas...")
        orders = client.get_orders(status="open")
        logger.info(f"Número de órdenes abiertas: {len(orders)}")
        
        # Ejemplo de envío de orden (comentado para evitar envíos accidentales)
        """
        logger.info("Enviando orden de prueba...")
        order = client.submit_order({
            "symbol": "AAPL",
            "qty": 1,
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
            "client_order_id": f"test-order-{int(time.time())}"
        })
        logger.info(f"Orden enviada: {order}")
        """
        
        logger.info("Ejemplo completado con éxito")
        
    except Exception as e:
        logger.error(f"Error en el ejemplo: {e}", exc_info=True)
    
    # Mantener el programa en ejecución para que el servidor MCP siga funcionando
    logger.info("Presiona Ctrl+C para salir...")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Saliendo...")

if __name__ == "__main__":
    main()