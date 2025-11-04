#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para ejecutar el servidor MCP (Market Connection Proxy) para Alpaca.
"""

import os
import sys
import logging
import argparse
from dotenv import load_dotenv

from mcp_server import create_mcp_server

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Función principal para ejecutar el servidor MCP."""
    # Cargar variables de entorno
    load_dotenv()
    
    # Parsear argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Ejecutar servidor MCP para Alpaca')
    parser.add_argument('--port', type=int, default=int(os.environ.get('MCP_PORT', 5000)),
                        help='Puerto para el servidor MCP')
    parser.add_argument('--host', type=str, default=os.environ.get('MCP_HOST', '0.0.0.0'),
                        help='Host para el servidor MCP')
    parser.add_argument('--redis-url', type=str, default=os.environ.get('REDIS_URL'),
                        help='URL de Redis para caché')
    parser.add_argument('--cache-ttl', type=int, default=int(os.environ.get('MCP_CACHE_TTL', 60)),
                        help='Tiempo de vida de la caché en segundos')
    parser.add_argument('--rate-limit-window', type=int, 
                        default=int(os.environ.get('MCP_RATE_LIMIT_WINDOW', 60)),
                        help='Ventana de tiempo para límites de tasa en segundos')
    parser.add_argument('--rate-limit-max-calls', type=int,
                        default=int(os.environ.get('MCP_RATE_LIMIT_MAX_CALLS', 200)),
                        help='Máximo de llamadas en la ventana de tiempo')
    
    args = parser.parse_args()
    
    # Verificar variables de entorno requeridas
    required_env_vars = ['ALPACA_API_KEY', 'ALPACA_API_SECRET']
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"Faltan variables de entorno requeridas: {', '.join(missing_vars)}")
        logger.error("Por favor, configura estas variables en el archivo .env o como variables de entorno")
        sys.exit(1)
    
    # Configuración del servidor
    config = {
        "alpaca_api_key": os.environ.get("ALPACA_API_KEY"),
        "alpaca_api_secret": os.environ.get("ALPACA_API_SECRET"),
        "alpaca_base_url": os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        "alpaca_data_url": os.environ.get("ALPACA_DATA_URL", "https://data.alpaca.markets"),
        "redis_url": args.redis_url,
        "cache_ttl": args.cache_ttl,
        "rate_limit_window": args.rate_limit_window,
        "rate_limit_max_calls": args.rate_limit_max_calls,
        "port": args.port,
        "host": args.host
    }
    
    logger.info(f"Iniciando servidor MCP en {args.host}:{args.port}")
    
    try:
        # Crear y ejecutar el servidor
        server = create_mcp_server(config)
        server.run()
    except KeyboardInterrupt:
        logger.info("Servidor detenido por el usuario")
    except Exception as e:
        logger.error(f"Error al ejecutar el servidor: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()