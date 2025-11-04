#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para la integración con la API de Alpaca.
Soporta conexión directa a Alpaca o a través del servidor MCP.
"""

import logging
import os
import time
import pandas as pd
import numpy as np
import requests
import json
import random
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta

# Importar cliente de Alpaca
try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import REST, TimeFrame
    from alpaca_trade_api.stream import Stream
except ImportError:
    logging.error("No se pudo importar alpaca_trade_api. Instálalo con: pip install alpaca-trade-api")

from src.utils.retry import retry_network

logger = logging.getLogger(__name__)


class AlpacaClient:
    """Cliente para interactuar con la API de Alpaca, con soporte para MCP."""

    def __init__(
        self,
        api_key: str = None,
        api_secret: str = None,
        base_url: str = None,
        data_url: str = None,
        trading_mode: str = "paper",
        use_mcp: bool = False,
        mcp_url: str = "http://localhost:8000",
        max_retry_attempts: int = 3,
        base_backoff_time_ms: int = 300,
        max_backoff_time_ms: int = 10000,
        jitter_factor: float = 0.1
    ):
        """Inicializa el cliente de Alpaca.

        Args:
            api_key: API key de Alpaca
            api_secret: API secret de Alpaca
            base_url: URL base de la API de Alpaca
            data_url: URL de la API de datos de Alpaca
            trading_mode: Modo de trading (paper, live, backtest)
            use_mcp: Usar el servidor MCP para las peticiones
            mcp_url: URL del servidor MCP
            max_retry_attempts: Número máximo de intentos de reintento
            base_backoff_time_ms: Tiempo base de espera entre reintentos (ms)
            max_backoff_time_ms: Tiempo máximo de espera entre reintentos (ms)
            jitter_factor: Factor de aleatoriedad para el tiempo de espera
        """
        # Configuración de Alpaca
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET")
        self.trading_mode = trading_mode
        
        # Configuración de MCP
        self.use_mcp = use_mcp
        self.mcp_url = mcp_url
        
        # Configuración de reintentos
        self.max_retry_attempts = max_retry_attempts
        self.base_backoff_time_ms = base_backoff_time_ms
        self.max_backoff_time_ms = max_backoff_time_ms
        self.jitter_factor = jitter_factor
        
        # Verificar credenciales
        if not self.api_key or not self.api_secret:
            logger.error("API key y API secret son requeridos")
            raise ValueError("API key y API secret son requeridos")
        
        # Configurar URLs según el modo de trading
        if not base_url:
            if trading_mode == "live":
                base_url = "https://api.alpaca.markets"
            else:  # paper o backtest
                base_url = "https://paper-api.alpaca.markets"
        
        if not data_url:
            data_url = "https://data.alpaca.markets"
        
        # Inicializar cliente REST
        self.api = tradeapi.REST(
            key_id=self.api_key,
            secret_key=self.api_secret,
            base_url=base_url,
            api_version="v2"
        )
        
        # Inicializar cliente de streaming
        self.stream = tradeapi.Stream(
            key_id=self.api_key,
            secret_key=self.api_secret,
            base_url=base_url,
            data_feed="iex"
        )
        
        # Verificar conexión con Alpaca
        try:
            self.api.get_account()
            logger.info(f"Conexión establecida con Alpaca ({trading_mode})")
        except Exception as e:
            logger.error(f"Error al conectar con Alpaca: {e}")
            raise ConnectionError(f"Error al conectar con Alpaca: {e}")
        
        # Verificar conexión con MCP si está habilitado
        if self.use_mcp:
            self._check_mcp_connection()
    
    @retry_network()
    def _check_mcp_connection(self) -> bool:
        """Verifica la conexión con el servidor MCP.
        
        Returns:
            bool: True si la conexión es exitosa
        """
        try:
            response = requests.get(f"{self.mcp_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("status") == "ok" and health_data.get("alpaca_connected"):
                    logger.info("Conexión con servidor MCP verificada")
                    return True
                else:
                    logger.warning(f"Servidor MCP no está conectado a Alpaca: {health_data}")
            else:
                logger.warning(f"Error al verificar servidor MCP: {response.status_code}")
            
            # Si hay problemas con MCP, desactivarlo
            logger.warning("Desactivando uso de MCP debido a problemas de conexión")
            self.use_mcp = False
            return False
        except Exception as e:
            logger.warning(f"No se pudo conectar al servidor MCP: {str(e)}")
            self.use_mcp = False
            return False

    @retry_network()
    def get_account(self) -> Dict[str, Any]:
        """Obtiene información de la cuenta.

        Returns:
            Dict[str, Any]: Información de la cuenta
        """
        try:
            if self.use_mcp:
                try:
                    response = requests.get(f"{self.mcp_url}/v2/account", timeout=10)
                    if response.status_code == 200:
                        return response.json()
                    else:
                        logger.warning(f"Error al obtener cuenta desde MCP: {response.status_code}")
                        # Fallback a API directa
                        self.use_mcp = False
                except Exception as e:
                    logger.warning(f"Error al conectar con MCP: {str(e)}")
                    self.use_mcp = False
            
            # Usar API directa de Alpaca
            account = self.api.get_account()
            return {
                "id": account.id,
                "status": account.status,
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "long_market_value": float(account.long_market_value),
                "short_market_value": float(account.short_market_value),
                "initial_margin": float(account.initial_margin),
                "maintenance_margin": float(account.maintenance_margin),
                "last_equity": float(account.last_equity),
                "daytrade_count": int(account.daytrade_count),
                "daytrading_buying_power": float(account.daytrading_buying_power),
            }
        except Exception as e:
            logger.error(f"Error al obtener información de cuenta: {e}", exc_info=True)
            return {}

    @retry_network()
    def get_positions(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene las posiciones actuales.

        Returns:
            Dict[str, Dict[str, Any]]: Posiciones actuales por símbolo
        """
        positions = {}
        
        try:
            if self.use_mcp:
                try:
                    response = requests.get(f"{self.mcp_url}/v2/positions", timeout=10)
                    if response.status_code == 200:
                        positions_list = response.json()
                        for position in positions_list:
                            positions[position["symbol"]] = position
                        logger.info(f"Obtenidas {len(positions)} posiciones desde MCP")
                        return positions
                    else:
                        logger.warning(f"Error al obtener posiciones desde MCP: {response.status_code}")
                        # Fallback a API directa
                        self.use_mcp = False
                except Exception as e:
                    logger.warning(f"Error al conectar con MCP: {str(e)}")
                    self.use_mcp = False
            
            # Usar API directa de Alpaca
            alpaca_positions = self.api.list_positions()
            
            for position in alpaca_positions:
                positions[position.symbol] = {
                    "symbol": position.symbol,
                    "qty": int(position.qty),
                    "side": "long" if int(position.qty) > 0 else "short",
                    "avg_entry_price": float(position.avg_entry_price),
                    "market_value": float(position.market_value),
                    "cost_basis": float(position.cost_basis),
                    "unrealized_pl": float(position.unrealized_pl),
                    "unrealized_plpc": float(position.unrealized_plpc),
                    "current_price": float(position.current_price),
                    "lastday_price": float(position.lastday_price),
                    "change_today": float(position.change_today),
                }
            
            logger.info(f"Obtenidas {len(positions)} posiciones")
            return positions
            
        except Exception as e:
            logger.error(f"Error al obtener posiciones: {e}", exc_info=True)
            return {}

    def get_orders(self, status: str = "open") -> List[Dict[str, Any]]:
        """Obtiene órdenes según su estado.

        Args:
            status: Estado de las órdenes (open, closed, all)

        Returns:
            List[Dict[str, Any]]: Lista de órdenes
        """
        orders = []
        
        try:
            if self.use_mcp:
                try:
                    response = requests.get(f"{self.mcp_url}/v2/orders?status={status}", timeout=10)
                    if response.status_code == 200:
                        return response.json()
                    else:
                        logger.warning(f"Error al obtener órdenes desde MCP: {response.status_code}")
                        # Fallback a API directa
                        self.use_mcp = False
                except Exception as e:
                    logger.warning(f"Error al conectar con MCP: {str(e)}")
                    self.use_mcp = False
            
            # Usar API directa de Alpaca
            if status == "open":
                alpaca_orders = self.api.list_orders(status="open")
            elif status == "closed":
                alpaca_orders = self.api.list_orders(status="closed", limit=100)
            else:  # all
                alpaca_orders = self.api.list_orders(status="all", limit=100)
            
            for order in alpaca_orders:
                orders.append({
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
                    "created_at": order.created_at.isoformat() if order.created_at else None,
                    "filled_at": order.filled_at.isoformat() if order.filled_at else None,
                    "expired_at": order.expired_at.isoformat() if order.expired_at else None,
                    "canceled_at": order.canceled_at.isoformat() if order.canceled_at else None,
                    "failed_at": order.failed_at.isoformat() if order.failed_at else None,
                    "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                })
            
            logger.info(f"Obtenidas {len(orders)} órdenes con estado {status}")
            return orders
            
        except Exception as e:
            logger.error(f"Error al obtener órdenes: {e}", exc_info=True)
            return []

    @retry_network()
    def submit_order(self, order_params: Dict[str, Any]) -> Dict[str, Any]:
        """Envía una orden a Alpaca.

        Args:
            order_params: Parámetros de la orden

        Returns:
            Dict[str, Any]: Información de la orden enviada
        """
        # Verificar modo de trading
        if self.trading_mode == "backtest":
            logger.warning("No se pueden enviar órdenes en modo backtest")
            return {"status": "error", "message": "No se pueden enviar órdenes en modo backtest"}
        
        # Verificar parámetros mínimos
        required_params = ["symbol", "qty", "side", "type", "time_in_force"]
        for param in required_params:
            if param not in order_params:
                logger.error(f"Falta parámetro requerido: {param}")
                return {"status": "error", "message": f"Falta parámetro requerido: {param}"}
        
        try:
            if self.use_mcp:
                try:
                    response = requests.post(
                        f"{self.mcp_url}/v2/orders",
                        json=order_params,
                        timeout=10
                    )
                    if response.status_code == 200:
                        return response.json()
                    else:
                        logger.warning(f"Error al enviar orden a MCP: {response.status_code}, {response.text}")
                        # Fallback a API directa
                        self.use_mcp = False
                except Exception as e:
                    logger.warning(f"Error al conectar con MCP: {str(e)}")
                    self.use_mcp = False
            
            # Usar API directa de Alpaca
            # Preparar parámetros
            symbol = order_params["symbol"]
            qty = abs(float(order_params["qty"]))  # Asegurar que es positivo
            side = order_params["side"]
            order_type = order_params["type"]
            time_in_force = order_params["time_in_force"]
            
            # Parámetros opcionales
            limit_price = order_params.get("limit_price")
            stop_price = order_params.get("stop_price")
            client_order_id = order_params.get("client_order_id")
            
            # Enviar orden
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side=side,
                type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
                client_order_id=client_order_id
            )
            
            # Convertir a diccionario
            order_dict = {
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
                "created_at": order.created_at.isoformat() if order.created_at else None,
            }
            
            logger.info(f"Orden enviada: {order_dict}")
            return order_dict
            
        except Exception as e:
            logger.error(f"Error al enviar orden: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancela una orden.

        Args:
            order_id: ID de la orden a cancelar

        Returns:
            Dict[str, Any]: Resultado de la cancelación
        """
        # Verificar modo de trading
        if self.trading_mode == "backtest":
            logger.warning("No se pueden cancelar órdenes en modo backtest")
            return {"status": "error", "message": "No se pueden cancelar órdenes en modo backtest"}
        
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Orden cancelada: {order_id}")
            return {"status": "cancelled", "order_id": order_id}
        except Exception as e:
            logger.error(f"Error al cancelar orden {order_id}: {e}", exc_info=True)
            return {"status": "error", "message": str(e), "order_id": order_id}

    def cancel_all_orders(self) -> Dict[str, Any]:
        """Cancela todas las órdenes abiertas.

        Returns:
            Dict[str, Any]: Resultado de la cancelación
        """
        # Verificar modo de trading
        if self.trading_mode == "backtest":
            logger.warning("No se pueden cancelar órdenes en modo backtest")
            return {"status": "error", "message": "No se pueden cancelar órdenes en modo backtest"}
        
        try:
            self.api.cancel_all_orders()
            logger.info("Todas las órdenes canceladas")
            return {"status": "success", "message": "Todas las órdenes canceladas"}
        except Exception as e:
            logger.error(f"Error al cancelar todas las órdenes: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def get_historical_bars(self, symbol: str, timeframe: str, start: str, end: str, limit: int = 1000) -> pd.DataFrame:
        """Obtiene barras históricas para un símbolo.

        Args:
            symbol: Símbolo
            timeframe: Intervalo de tiempo (1Min, 5Min, 15Min, 1H, 1D)
            start: Fecha de inicio (YYYY-MM-DD)
            end: Fecha de fin (YYYY-MM-DD)
            limit: Límite de barras

        Returns:
            pd.DataFrame: DataFrame con barras históricas
        """
        try:
            # Convertir timeframe al formato de Alpaca
            tf_map = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame.Minute,
                "15Min": TimeFrame.Minute,
                "1H": TimeFrame.Hour,
                "1D": TimeFrame.Day,
            }
            
            # Multiplicador para timeframes específicos
            multiplier = 1
            if timeframe == "5Min":
                multiplier = 5
            elif timeframe == "15Min":
                multiplier = 15
            
            # Obtener barras
            if timeframe in tf_map:
                bars = self.api.get_bars(
                    symbol,
                    tf_map[timeframe],
                    start=start,
                    end=end,
                    limit=limit,
                    adjustment='raw',
                    multiplier=multiplier
                ).df
            else:
                logger.error(f"Timeframe no soportado: {timeframe}")
                return pd.DataFrame()
            
            if bars.empty:
                logger.warning(f"No se encontraron barras para {symbol} ({timeframe})")
                return pd.DataFrame()
            
            logger.info(f"Obtenidas {len(bars)} barras para {symbol} ({timeframe})")
            return bars
            
        except Exception as e:
            logger.error(f"Error al obtener barras históricas para {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    def get_latest_quote(self, symbol: str) -> Dict[str, Any]:
        """Obtiene la última cotización para un símbolo.

        Args:
            symbol: Símbolo

        Returns:
            Dict[str, Any]: Información de la cotización
        """
        try:
            quote = self.api.get_latest_quote(symbol)
            
            return {
                "symbol": symbol,
                "bid_price": float(quote.bp),
                "bid_size": int(quote.bs),
                "ask_price": float(quote.ap),
                "ask_size": int(quote.as_),
                "timestamp": quote.t.isoformat() if hasattr(quote, 't') else None,
            }
            
        except Exception as e:
            logger.error(f"Error al obtener cotización para {symbol}: {e}", exc_info=True)
            return {}

    def get_latest_trade(self, symbol: str) -> Dict[str, Any]:
        """Obtiene el último trade para un símbolo.

        Args:
            symbol: Símbolo

        Returns:
            Dict[str, Any]: Información del trade
        """
        try:
            trade = self.api.get_latest_trade(symbol)
            
            return {
                "symbol": symbol,
                "price": float(trade.p),
                "size": int(trade.s),
                "exchange": trade.x,
                "timestamp": trade.t.isoformat() if hasattr(trade, 't') else None,
            }
            
        except Exception as e:
            logger.error(f"Error al obtener trade para {symbol}: {e}", exc_info=True)
            return {}

    def get_clock(self) -> Dict[str, Any]:
        """Obtiene información del reloj de mercado.

        Returns:
            Dict[str, Any]: Información del reloj
        """
        try:
            clock = self.api.get_clock()
            
            return {
                "timestamp": clock.timestamp.isoformat(),
                "is_open": clock.is_open,
                "next_open": clock.next_open.isoformat(),
                "next_close": clock.next_close.isoformat(),
            }
            
        except Exception as e:
            logger.error(f"Error al obtener reloj de mercado: {e}", exc_info=True)
            return {}

    def get_calendar(self, start: str = None, end: str = None) -> List[Dict[str, Any]]:
        """Obtiene el calendario de mercado.

        Args:
            start: Fecha de inicio (YYYY-MM-DD)
            end: Fecha de fin (YYYY-MM-DD)

        Returns:
            List[Dict[str, Any]]: Calendario de mercado
        """
        try:
            # Si no se especifican fechas, usar la semana actual
            if not start:
                start = datetime.now().strftime("%Y-%m-%d")
            if not end:
                end = (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d")
            
            calendar = self.api.get_calendar(start=start, end=end)
            
            result = []
            for day in calendar:
                result.append({
                    "date": day.date.isoformat(),
                    "open": day.open.isoformat(),
                    "close": day.close.isoformat(),
                    "session_open": day.session_open.isoformat(),
                    "session_close": day.session_close.isoformat(),
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error al obtener calendario de mercado: {e}", exc_info=True)
            return []

    def start_stream(self, symbols: List[str], handlers: Dict[str, Any] = None) -> bool:
        """Inicia el streaming de datos.

        Args:
            symbols: Lista de símbolos
            handlers: Manejadores de eventos

        Returns:
            bool: True si se inició correctamente, False en caso contrario
        """
        if self.stream_running:
            logger.warning("El streaming ya está en ejecución")
            return True
        
        try:
            # Inicializar cliente de streaming
            self.stream = Stream(
                key_id=self.api_key,
                secret_key=self.api_secret,
                base_url=self.base_url,
                data_feed="iex"  # Usar IEX como fuente de datos
            )
            
            # Configurar manejadores de eventos
            if handlers:
                if "trade" in handlers and callable(handlers["trade"]):
                    self.stream.subscribe_trades(handlers["trade"], *symbols)
                    logger.info(f"Suscrito a trades para {len(symbols)} símbolos")
                
                if "quote" in handlers and callable(handlers["quote"]):
                    self.stream.subscribe_quotes(handlers["quote"], *symbols)
                    logger.info(f"Suscrito a quotes para {len(symbols)} símbolos")
                
                if "bar" in handlers and callable(handlers["bar"]):
                    self.stream.subscribe_bars(handlers["bar"], *symbols)
                    logger.info(f"Suscrito a barras para {len(symbols)} símbolos")
                
                if "updated_bar" in handlers and callable(handlers["updated_bar"]):
                    self.stream.subscribe_updated_bars(handlers["updated_bar"], *symbols)
                    logger.info(f"Suscrito a barras actualizadas para {len(symbols)} símbolos")
            
            # Iniciar streaming en un hilo separado
            self.stream.run_async()
            self.stream_running = True
            
            logger.info("Streaming iniciado correctamente")
            return True
            
        except Exception as e:
            logger.error(f"Error al iniciar streaming: {e}", exc_info=True)
            return False

    def stop_stream(self) -> bool:
        """Detiene el streaming de datos.

        Returns:
            bool: True si se detuvo correctamente, False en caso contrario
        """
        if not self.stream_running or not self.stream:
            logger.warning("El streaming no está en ejecución")
            return True
        
        try:
            self.stream.stop()
            self.stream_running = False
            logger.info("Streaming detenido correctamente")
            return True
        except Exception as e:
            logger.error(f"Error al detener streaming: {e}", exc_info=True)
            return False

    def is_market_open(self) -> bool:
        """Verifica si el mercado está abierto.

        Returns:
            bool: True si el mercado está abierto, False en caso contrario
        """
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            logger.error(f"Error al verificar estado del mercado: {e}", exc_info=True)
            return False

    def get_time_to_next_market_event(self) -> Dict[str, Any]:
        """Obtiene el tiempo hasta el próximo evento de mercado.

        Returns:
            Dict[str, Any]: Información sobre el próximo evento
        """
        try:
            clock = self.api.get_clock()
            now = datetime.now(clock.timestamp.tzinfo)
            
            if clock.is_open:
                # Mercado abierto, calcular tiempo hasta cierre
                next_event = "close"
                time_to_event = (clock.next_close - now).total_seconds()
            else:
                # Mercado cerrado, calcular tiempo hasta apertura
                next_event = "open"
                time_to_event = (clock.next_open - now).total_seconds()
            
            # Convertir a formato legible
            hours = int(time_to_event // 3600)
            minutes = int((time_to_event % 3600) // 60)
            seconds = int(time_to_event % 60)
            
            return {
                "next_event": next_event,
                "time_to_event_seconds": time_to_event,
                "time_to_event_formatted": f"{hours}h {minutes}m {seconds}s",
                "timestamp": now.isoformat(),
                "is_open": clock.is_open,
            }
            
        except Exception as e:
            logger.error(f"Error al calcular tiempo hasta próximo evento: {e}", exc_info=True)
            return {}