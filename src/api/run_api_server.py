"""
Script para ejecutar el servidor API.
"""
import os
import argparse
from dotenv import load_dotenv
import uvicorn
import sys

# Añadir directorio raíz al path para importaciones
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Importar la aplicación FastAPI
from src.api.api_server import app

# Cargar variables de entorno
load_dotenv()

def main():
    """Función principal para ejecutar el servidor API."""
    parser = argparse.ArgumentParser(description='Ejecutar servidor API')
    parser.add_argument(
        '--host', 
        default=os.getenv("API_HOST", "0.0.0.0"),
        help='Host para el servidor API'
    )
    parser.add_argument(
        '--port', 
        type=int, 
        default=int(os.getenv("API_PORT", "8000")),
        help='Puerto para el servidor API'
    )
    parser.add_argument(
        '--reload',
        action='store_true',
        help='Activar recarga automática para desarrollo'
    )
    
    args = parser.parse_args()
    
    print(f"Iniciando servidor API en {args.host}:{args.port}")
    uvicorn.run(
        "src.api.api_server:app", 
        host=args.host, 
        port=args.port,
        reload=args.reload
    )

if __name__ == '__main__':
    main()