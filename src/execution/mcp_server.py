#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Servidor MCP (Market Connection Proxy) para Alpaca.
Actúa como intermediario entre el sistema de trading y la API de Alpaca,
proporcionando caché local, manejo de límites de tasa y resiliencia.
"""

import os
import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import queue

import alpaca_trade_api as tradeapi
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import redis
from redis.exceptions import RedisError
import jwt

# Configurar logging
logger = logging.getLogger(__name__)

class MCPConfig(BaseModel):
    """Configuración para el servidor MCP."""
    alpaca_api_key: str
    alpaca_api_secret: str
    alpaca_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_data_url: str = "https://data.alpaca.markets"
    redis_url: Optional[str] = None
    cache_ttl: int = 60  # Tiempo de vida de la caché en segundos
    rate_limit_window: int = 60  # Ventana de tiempo para límites de tasa en segundos
    rate_limit_max_calls: int = 200  # Máximo de llamadas en la ventana de tiempo
    port: int = 5000
    host: str = "0.0.0.0"

class MarketDataRequest(BaseModel):
    """Modelo para solicitudes de datos de mercado."""
    symbols: List[str]
    timeframe: str = "1Min"
    start: Optional[str] = None
    end: Optional[str] = None
    limit: Optional[int] = 100

class OrderRequest(BaseModel):
    """Modelo para solicitudes de órdenes."""
    symbol: str
    qty: float
    side: str
    type: str = "market"
    time_in_force: str = "day"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    client_order_id: Optional[str] = None
    extended_hours: bool = False
    order_class: Optional[str] = None
    take_profit: Optional[Dict[str, Any]] = None
    stop_loss: Optional[Dict[str, Any]] = None

class MCPServer:
    """
    Servidor MCP (Market Connection Proxy) para Alpaca.
    
    Proporciona:
    - Caché local para datos de mercado
    - Manejo de límites de tasa
    - Resiliencia ante errores transitorios
    - API REST para interactuar con Alpaca
    """
    
    def __init__(self, config: MCPConfig):
        """
        Inicializa el servidor MCP.
        
        Args:
            config: Configuración del servidor
        """
        self.config = config
        self.app = FastAPI(title="Alpaca MCP Server", 
                          description="Market Connection Proxy para Alpaca API",
                          version="1.0.0")
        self.paused = False
        self.jwt_secret = os.environ.get("JWT_SECRET")
        
        # Configurar CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Cliente de Alpaca
        self.alpaca = tradeapi.REST(
            key_id=config.alpaca_api_key,
            secret_key=config.alpaca_api_secret,
            base_url=config.alpaca_base_url,
            api_version="v2"
        )
        
        # Cliente de Redis para caché (opcional)
        self.redis = None
        if config.redis_url:
            try:
                self.redis = redis.from_url(config.redis_url)
                logger.info(f"Conectado a Redis en {config.redis_url}")
            except RedisError as e:
                logger.warning(f"No se pudo conectar a Redis: {str(e)}")
        
        # Control de límites de tasa
        self.api_calls = queue.Queue()
        self.api_call_lock = threading.Lock()
        
        # Registrar rutas
        self._register_routes()
        
        logger.info("Servidor MCP inicializado")
    
    def _register_routes(self):
        """Registra las rutas de la API."""
        # Rutas de estado
        self.app.get("/health")(self.health_check)
        # Rutas de control (protegidass por JWT si está configurado)
        self.app.post("/control/pause")(self.pause)
        self.app.post("/control/resume")(self.resume)
        self.app.get("/control/status")(self.status)
        
        # Rutas de datos de mercado
        self.app.get("/v2/market/bars")(self.get_bars)
        self.app.get("/v2/market/quotes")(self.get_quotes)
        self.app.get("/v2/market/trades")(self.get_trades)
        
        # Rutas de cuenta y posiciones
        self.app.get("/v2/account")(self.get_account)
        self.app.get("/v2/positions")(self.get_positions)
        self.app.get("/v2/positions/{symbol}")(self.get_position)
        
        # Rutas de órdenes
        self.app.get("/v2/orders")(self.get_orders)
        self.app.post("/v2/orders")(self.submit_order)
        self.app.get("/v2/orders/{order_id}")(self.get_order)
        self.app.delete("/v2/orders/{order_id}")(self.cancel_order)
        self.app.delete("/v2/orders")(self.cancel_all_orders)
    
    async def health_check(self):
        """Endpoint para verificar el estado del servidor."""
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "alpaca_connected": self._check_alpaca_connection()
        }

    def _verify_jwt(self, authorization: Optional[str]) -> bool:
        if not self.jwt_secret:
            return True
        if not authorization or not authorization.lower().startswith("bearer "):
            return False
        token = authorization.split(" ", 1)[1]
        try:
            jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
            return True
        except Exception:
            return False

    async def pause(self, authorization: Optional[str] = Header(default=None)):
        if not self._verify_jwt(authorization):
            raise HTTPException(status_code=401, detail="Unauthorized")
        self.paused = True
        return {"status": "paused", "timestamp": datetime.now().isoformat()}

    async def resume(self, authorization: Optional[str] = Header(default=None)):
        if not self._verify_jwt(authorization):
            raise HTTPException(status_code=401, detail="Unauthorized")
        self.paused = False
        return {"status": "running", "timestamp": datetime.now().isoformat()}

    async def status(self):
        return {"status": "paused" if self.paused else "running"}
    
    def _check_alpaca_connection(self) -> bool:
        """Verifica la conexión con Alpaca."""
        try:
            self.alpaca.get_account()
            return True
        except Exception as e:
            logger.error(f"Error al conectar con Alpaca: {str(e)}")
            return False
    
    def _check_rate_limit(self) -> bool:
        """
        Verifica si se ha alcanzado el límite de tasa.
        
        Returns:
            bool: True si se puede hacer la llamada, False si se ha alcanzado el límite
        """
        with self.api_call_lock:
            # Eliminar llamadas antiguas fuera de la ventana de tiempo
            current_time = time.time()
            window_start = current_time - self.config.rate_limit_window
            
            # Mantener solo las llamadas dentro de la ventana de tiempo
            calls_in_window = []
            while not self.api_calls.empty():
                call_time = self.api_calls.get()
                if call_time >= window_start:
                    calls_in_window.append(call_time)
            
            # Volver a poner las llamadas en la ventana en la cola
            for call_time in calls_in_window:
                self.api_calls.put(call_time)
            
            # Verificar si se ha alcanzado el límite
            if len(calls_in_window) >= self.config.rate_limit_max_calls:
                return False
            
            # Registrar la nueva llamada
            self.api_calls.put(current_time)
            return True
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """
        Obtiene datos de la caché.
        
        Args:
            key: Clave de la caché
            
        Returns:
            Datos almacenados o None si no están en caché
        """
        if not self.redis:
            return None
        
        try:
            data = self.redis.get(key)
            if data:
                return json.loads(data)
        except (RedisError, json.JSONDecodeError) as e:
            logger.warning(f"Error al obtener datos de caché: {str(e)}")
        
        return None
    
    def _store_in_cache(self, key: str, data: Any) -> bool:
        """
        Almacena datos en la caché.
        
        Args:
            key: Clave de la caché
            data: Datos a almacenar
            
        Returns:
            bool: True si se almacenó correctamente, False en caso contrario
        """
        if not self.redis:
            return False
        
        try:
            self.redis.setex(
                key,
                self.config.cache_ttl,
                json.dumps(data)
            )
            return True
        except (RedisError, TypeError) as e:
            logger.warning(f"Error al almacenar datos en caché: {str(e)}")
            return False
    
    # Implementación de endpoints de datos de mercado
    
    async def get_bars(self, symbols: str, timeframe: str = "1Min", 
                      start: Optional[str] = None, end: Optional[str] = None, 
                      limit: int = 100):
        """Endpoint para obtener barras de precios."""
        if self.paused:
            raise HTTPException(status_code=503, detail="Service paused")
        if not self._check_rate_limit():
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        symbols_list = symbols.split(",")
        cache_key = f"bars:{timeframe}:{symbols}:{start}:{end}:{limit}"
        
        # Intentar obtener de caché
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Obtener datos de Alpaca
            bars = self.alpaca.get_bars(
                symbols_list,
                timeframe,
                start=start,
                end=end,
                limit=limit
            ).df.reset_index()
            
            # Convertir a formato JSON
            result = {}
            for symbol in symbols_list:
                symbol_bars = bars[bars['symbol'] == symbol]
                if not symbol_bars.empty:
                    result[symbol] = symbol_bars.to_dict(orient='records')
            
            # Almacenar en caché
            self._store_in_cache(cache_key, result)
            
            return result
        except Exception as e:
            logger.error(f"Error al obtener barras: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_quotes(self, symbols: str):
        """Endpoint para obtener cotizaciones."""
        if self.paused:
            raise HTTPException(status_code=503, detail="Service paused")
        if not self._check_rate_limit():
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        symbols_list = symbols.split(",")
        cache_key = f"quotes:{symbols}"
        
        # Intentar obtener de caché
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Obtener datos de Alpaca
            quotes = {}
            for symbol in symbols_list:
                quote = self.alpaca.get_latest_quote(symbol)
                quotes[symbol] = {
                    "ask_price": float(quote.askprice),
                    "ask_size": float(quote.asksize),
                    "bid_price": float(quote.bidprice),
                    "bid_size": float(quote.bidsize),
                    "timestamp": quote.timestamp.isoformat()
                }
            
            # Almacenar en caché (con TTL corto para cotizaciones)
            self._store_in_cache(cache_key, quotes)
            
            return quotes
        except Exception as e:
            logger.error(f"Error al obtener cotizaciones: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_trades(self, symbols: str):
        """Endpoint para obtener operaciones recientes."""
        if self.paused:
            raise HTTPException(status_code=503, detail="Service paused")
        if not self._check_rate_limit():
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        symbols_list = symbols.split(",")
        cache_key = f"trades:{symbols}"
        
        # Intentar obtener de caché
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data
        
        try:
            # Obtener datos de Alpaca
            trades = {}
            for symbol in symbols_list:
                trade = self.alpaca.get_latest_trade(symbol)
                trades[symbol] = {
                    "price": float(trade.price),
                    "size": float(trade.size),
                    "timestamp": trade.timestamp.isoformat()
                }
            
            # Almacenar en caché (con TTL corto para operaciones)
            self._store_in_cache(cache_key, trades)
            
            return trades
        except Exception as e:
            logger.error(f"Error al obtener operaciones: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Implementación de endpoints de cuenta y posiciones
    
    async def get_account(self):
        """Endpoint para obtener información de la cuenta."""
        if self.paused:
            raise HTTPException(status_code=503, detail="Service paused")
        if not self._check_rate_limit():
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        try:
            account = self.alpaca.get_account()
            return {
                "id": account.id,
                "status": account.status,
                "currency": account.currency,
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "buying_power": float(account.buying_power),
                "initial_margin": float(account.initial_margin),
                "maintenance_margin": float(account.maintenance_margin),
                "daytrading_buying_power": float(account.daytrading_buying_power),
                "last_equity": float(account.last_equity),
                "last_maintenance_margin": float(account.last_maintenance_margin),
                "created_at": account.created_at.isoformat()
            }
        except Exception as e:
            logger.error(f"Error al obtener información de cuenta: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_positions(self):
        """Endpoint para obtener todas las posiciones."""
        if self.paused:
            raise HTTPException(status_code=503, detail="Service paused")
        if not self._check_rate_limit():
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        try:
            positions = self.alpaca.list_positions()
            result = []
            for position in positions:
                result.append({
                    "symbol": position.symbol,
                    "qty": float(position.qty),
                    "avg_entry_price": float(position.avg_entry_price),
                    "market_value": float(position.market_value),
                    "cost_basis": float(position.cost_basis),
                    "unrealized_pl": float(position.unrealized_pl),
                    "unrealized_plpc": float(position.unrealized_plpc),
                    "current_price": float(position.current_price),
                    "lastday_price": float(position.lastday_price),
                    "change_today": float(position.change_today)
                })
            return result
        except Exception as e:
            logger.error(f"Error al obtener posiciones: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_position(self, symbol: str):
        """Endpoint para obtener una posición específica."""
        if self.paused:
            raise HTTPException(status_code=503, detail="Service paused")
        if not self._check_rate_limit():
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        try:
            position = self.alpaca.get_position(symbol)
            return {
                "symbol": position.symbol,
                "qty": float(position.qty),
                "avg_entry_price": float(position.avg_entry_price),
                "market_value": float(position.market_value),
                "cost_basis": float(position.cost_basis),
                "unrealized_pl": float(position.unrealized_pl),
                "unrealized_plpc": float(position.unrealized_plpc),
                "current_price": float(position.current_price),
                "lastday_price": float(position.lastday_price),
                "change_today": float(position.change_today)
            }
        except Exception as e:
            logger.error(f"Error al obtener posición para {symbol}: {str(e)}")
            raise HTTPException(status_code=404, detail=f"Position not found for {symbol}")
    
    # Implementación de endpoints de órdenes
    
    async def get_orders(self, status: str = "open", limit: int = 100, 
                        after: Optional[str] = None, until: Optional[str] = None):
        """Endpoint para obtener órdenes."""
        if self.paused:
            raise HTTPException(status_code=503, detail="Service paused")
        if not self._check_rate_limit():
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        try:
            orders = self.alpaca.list_orders(
                status=status,
                limit=limit,
                after=after,
                until=until
            )
            
            result = []
            for order in orders:
                result.append({
                    "id": order.id,
                    "client_order_id": order.client_order_id,
                    "symbol": order.symbol,
                    "qty": float(order.qty),
                    "filled_qty": float(order.filled_qty),
                    "side": order.side,
                    "type": order.type,
                    "time_in_force": order.time_in_force,
                    "limit_price": float(order.limit_price) if order.limit_price else None,
                    "stop_price": float(order.stop_price) if order.stop_price else None,
                    "status": order.status,
                    "created_at": order.created_at.isoformat(),
                    "updated_at": order.updated_at.isoformat() if order.updated_at else None,
                    "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
                    "filled_at": order.filled_at.isoformat() if order.filled_at else None,
                    "expired_at": order.expired_at.isoformat() if order.expired_at else None,
                    "canceled_at": order.canceled_at.isoformat() if order.canceled_at else None
                })
            
            return result
        except Exception as e:
            logger.error(f"Error al obtener órdenes: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def submit_order(self, order: OrderRequest):
        """Endpoint para enviar una orden."""
        if self.paused:
            raise HTTPException(status_code=503, detail="Service paused")
        if not self._check_rate_limit():
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        try:
            # Convertir el modelo Pydantic a un diccionario y eliminar valores None
            order_data = order.dict(exclude_none=True)
            
            # Enviar orden a Alpaca
            submitted_order = self.alpaca.submit_order(**order_data)
            
            return {
                "id": submitted_order.id,
                "client_order_id": submitted_order.client_order_id,
                "symbol": submitted_order.symbol,
                "qty": float(submitted_order.qty),
                "side": submitted_order.side,
                "type": submitted_order.type,
                "time_in_force": submitted_order.time_in_force,
                "limit_price": float(submitted_order.limit_price) if submitted_order.limit_price else None,
                "stop_price": float(submitted_order.stop_price) if submitted_order.stop_price else None,
                "status": submitted_order.status,
                "created_at": submitted_order.created_at.isoformat()
            }
        except Exception as e:
            logger.error(f"Error al enviar orden: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_order(self, order_id: str):
        """Endpoint para obtener una orden específica."""
        if not self._check_rate_limit():
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        try:
            order = self.alpaca.get_order(order_id)
            
            return {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "qty": float(order.qty),
                "filled_qty": float(order.filled_qty),
                "side": order.side,
                "type": order.type,
                "time_in_force": order.time_in_force,
                "limit_price": float(order.limit_price) if order.limit_price else None,
                "stop_price": float(order.stop_price) if order.stop_price else None,
                "status": order.status,
                "created_at": order.created_at.isoformat(),
                "updated_at": order.updated_at.isoformat() if order.updated_at else None,
                "submitted_at": order.submitted_at.isoformat() if order.submitted_at else None,
                "filled_at": order.filled_at.isoformat() if order.filled_at else None,
                "expired_at": order.expired_at.isoformat() if order.expired_at else None,
                "canceled_at": order.canceled_at.isoformat() if order.canceled_at else None
            }
        except Exception as e:
            logger.error(f"Error al obtener orden {order_id}: {str(e)}")
            raise HTTPException(status_code=404, detail=f"Order not found: {order_id}")
    
    async def cancel_order(self, order_id: str):
        """Endpoint para cancelar una orden específica."""
        if not self._check_rate_limit():
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        try:
            self.alpaca.cancel_order(order_id)
            return {"message": f"Order {order_id} cancelled successfully"}
        except Exception as e:
            logger.error(f"Error al cancelar orden {order_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def cancel_all_orders(self):
        """Endpoint para cancelar todas las órdenes."""
        if not self._check_rate_limit():
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        try:
            self.alpaca.cancel_all_orders()
            return {"message": "All orders cancelled successfully"}
        except Exception as e:
            logger.error(f"Error al cancelar todas las órdenes: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def run(self):
        """Inicia el servidor MCP."""
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port
        )

def create_mcp_server(config_dict: Dict[str, Any]) -> MCPServer:
    """
    Crea una instancia del servidor MCP.
    
    Args:
        config_dict: Diccionario de configuración
        
    Returns:
        Instancia del servidor MCP
    """
    config = MCPConfig(**config_dict)
    return MCPServer(config)

if __name__ == "__main__":
    # Configuración de ejemplo
    config = {
        "alpaca_api_key": os.environ.get("ALPACA_API_KEY"),
        "alpaca_api_secret": os.environ.get("ALPACA_API_SECRET"),
        "alpaca_base_url": os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
        "redis_url": os.environ.get("REDIS_URL"),
        "port": int(os.environ.get("MCP_PORT", 5000))
    }
    
    # Crear y ejecutar el servidor
    server = create_mcp_server(config)
    server.run()