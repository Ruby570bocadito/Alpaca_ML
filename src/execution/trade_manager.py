"""
Trade Manager para gestión avanzada de operaciones.
Combina señales y ejecución con controles avanzados.
"""
import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from datetime import datetime, timedelta
import threading
import requests
from dotenv import load_dotenv

# Importaciones locales
from .alpaca_client import AlpacaClient
from ..risk.risk_manager import RiskManager
from ..monitoring.alerts import AlertManager

# Cargar variables de entorno
load_dotenv()

# Configuración
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "0.1"))  # % del portfolio
COOLDOWN_MINUTES = int(os.getenv("TRADE_COOLDOWN_MINUTES", "15"))
TRAILING_ACTIVATION_PCT = float(os.getenv("TRAILING_ACTIVATION_PCT", "0.5"))
TRAILING_STOP_PCT = float(os.getenv("TRAILING_STOP_PCT", "0.8"))
VOLATILITY_MAX_PCT = float(os.getenv("VOLATILITY_MAX_PCT", "5.0"))
TELEGRAM_ENABLED = os.getenv("TELEGRAM_ENABLED", "false").lower() == "true"
DISCORD_ENABLED = os.getenv("DISCORD_ENABLED", "false").lower() == "true"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")

# Configurar logging
logger = logging.getLogger(__name__)

class TradeManager:
    """
    Gestor avanzado de operaciones que combina señales y ejecución.
    Incluye control de cantidad máxima por activo, take profit/stop loss dinámicos,
    cooldown entre operaciones y notificaciones.
    """
    
    def __init__(
        self,
        alpaca_client: AlpacaClient,
        risk_manager: Optional[RiskManager] = None,
        max_position_size: float = MAX_POSITION_SIZE,
        cooldown_minutes: int = COOLDOWN_MINUTES
    ):
        """
        Inicializa el Trade Manager.
        
        Args:
            alpaca_client: Cliente de Alpaca para ejecución de órdenes
            risk_manager: Gestor de riesgo (opcional)
            max_position_size: Tamaño máximo de posición como % del portfolio
            cooldown_minutes: Tiempo mínimo entre operaciones del mismo símbolo
        """
        self.alpaca_client = alpaca_client
        self.risk_manager = risk_manager
        self.max_position_size = max_position_size
        self.cooldown_minutes = cooldown_minutes
        
        # Estado interno
        self.last_trades = {}  # Registro de últimas operaciones por símbolo
        self.active_orders = {}  # Órdenes activas
        self.lock = threading.RLock()  # Lock para operaciones thread-safe
        
        # Inicializar gestor de alertas
        self.alert_manager = AlertManager()
        
        logger.info(f"TradeManager inicializado con max_position_size={max_position_size}, "
                   f"cooldown_minutes={cooldown_minutes}")
    
    def process_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa una señal de trading y ejecuta la operación correspondiente.
        
        Args:
            signal: Diccionario con la señal de trading
                {
                    "symbol": str,
                    "side": "buy" o "sell",
                    "strength": float,  # 0.0 a 1.0
                    "strategy": str,
                    "timeframe": str,
                    "take_profit_pct": float,  # opcional
                    "stop_loss_pct": float,  # opcional
                    "quantity": float,  # opcional
                    "order_type": str,  # opcional (market, limit, etc.)
                    "limit_price": float,  # opcional
                    "metadata": dict  # opcional
                }
        
        Returns:
            Diccionario con el resultado de la operación
        """
        with self.lock:
            symbol = signal.get("symbol")
            side = signal.get("side")
            
            if not symbol or not side:
                error_msg = "Señal inválida: se requiere symbol y side"
                logger.error(error_msg)
                return {"success": False, "error": error_msg}
            
            # Verificar cooldown
            if not self._check_cooldown(symbol):
                cooldown_msg = f"Cooldown activo para {symbol}, operación ignorada"
                logger.info(cooldown_msg)
                return {"success": False, "error": cooldown_msg, "cooldown": True}

            # Filtro de volatilidad
            try:
                bars = self.alpaca_client.get_bars(symbol, "1Day", None, None, limit=30)
                df = pd.DataFrame(bars)
                if not df.empty and "close" in df.columns:
                    vol = df["close"].pct_change().rolling(20).std().iloc[-1] * 100
                    if vol and vol > VOLATILITY_MAX_PCT:
                        msg = f"Volatilidad {vol:.2f}% > umbral {VOLATILITY_MAX_PCT}% para {symbol}"
                        logger.warning(msg)
                        return {"success": False, "error": msg, "volatility": vol}
            except Exception as e:
                logger.warning(f"No se pudo calcular volatilidad para {symbol}: {e}")
            
            # Verificar restricciones de riesgo
            if self.risk_manager:
                risk_check = self.risk_manager.check_trade(symbol, side, signal.get("quantity"))
                if not risk_check["allowed"]:
                    logger.warning(f"Operación rechazada por gestor de riesgo: {risk_check['reason']}")
                    return {"success": False, "error": risk_check["reason"], "risk_check": risk_check}
            
            # Calcular cantidad
            qty = self._calculate_quantity(symbol, side, signal)
            if qty <= 0:
                return {"success": False, "error": "Cantidad calculada es cero o negativa"}
            
            # Preparar parámetros de la orden
            order_params = self._prepare_order_params(symbol, side, qty, signal)
            
            # Ejecutar orden
            try:
                order_result = self.alpaca_client.submit_order(**order_params)
                
                # Registrar operación
                self._register_trade(symbol, side, qty, order_result)
                
                # Enviar notificación
                self._send_trade_notification(symbol, side, qty, order_params, order_result)

                # Configurar trailing stop
                try:
                    if self.risk_manager:
                        position_qty = qty if side == "buy" else -qty
                        self.risk_manager.set_trailing_stop(
                            symbol,
                            position_qty,
                            activation_pct=signal.get("trailing_activation_pct", TRAILING_ACTIVATION_PCT),
                            trail_pct=signal.get("trailing_stop_pct", TRAILING_STOP_PCT),
                        )
                except Exception as e:
                    logger.error(f"Error al configurar trailing stop para {symbol}: {e}")
                
                logger.info(f"Orden ejecutada: {symbol} {side} {qty} - ID: {order_result.get('id')}")
                return {
                    "success": True,
                    "order": order_result,
                    "params": order_params
                }
            
            except Exception as e:
                error_msg = f"Error al ejecutar orden: {str(e)}"
                logger.error(error_msg)
                
                # Notificar error
                self._send_error_notification(symbol, side, qty, order_params, str(e))
                
                return {"success": False, "error": error_msg}
    
    def _check_cooldown(self, symbol: str) -> bool:
        """
        Verifica si ha pasado suficiente tiempo desde la última operación del símbolo.
        
        Args:
            symbol: Símbolo a verificar
            
        Returns:
            True si se puede operar, False si está en cooldown
        """
        if symbol not in self.last_trades:
            return True
        
        last_trade_time = self.last_trades[symbol]["timestamp"]
        cooldown_delta = timedelta(minutes=self.cooldown_minutes)
        
        return datetime.now() - last_trade_time > cooldown_delta
    
    def _calculate_quantity(self, symbol: str, side: str, signal: Dict[str, Any]) -> float:
        """
        Calcula la cantidad a operar basada en el tamaño máximo de posición.
        
        Args:
            symbol: Símbolo a operar
            side: Dirección de la operación (buy/sell)
            signal: Señal de trading
            
        Returns:
            Cantidad a operar
        """
        # Si la señal especifica una cantidad, usarla (con validación)
        if "quantity" in signal and signal["quantity"] > 0:
            return float(signal["quantity"])
        
        try:
            # Obtener valor del portfolio
            account = self.alpaca_client.get_account()
            portfolio_value = float(account["portfolio_value"])
            
            # Obtener precio actual
            latest_quote = self.alpaca_client.get_latest_quote(symbol)
            current_price = latest_quote["askprice"] if side == "buy" else latest_quote["bidprice"]
            
            # Calcular cantidad basada en % máximo del portfolio
            max_position_value = portfolio_value * self.max_position_size
            
            # Ajustar por "strength" de la señal si está presente (0.0-1.0)
            if "strength" in signal and 0 <= signal["strength"] <= 1:
                max_position_value *= signal["strength"]
            
            # Calcular cantidad
            qty = max_position_value / current_price
            
            # Redondear a 2 decimales para acciones, entero para otros activos
            if symbol.startswith("BTC") or symbol.startswith("ETH"):
                qty = round(qty, 4)  # Crypto
            else:
                qty = round(qty, 2)  # Acciones
            
            return qty
        
        except Exception as e:
            logger.error(f"Error al calcular cantidad: {str(e)}")
            return 0
    
    def _prepare_order_params(self, symbol: str, side: str, qty: float, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepara los parámetros para la orden.
        
        Args:
            symbol: Símbolo a operar
            side: Dirección de la operación (buy/sell)
            qty: Cantidad a operar
            signal: Señal de trading
            
        Returns:
            Diccionario con parámetros para submit_order
        """
        # Parámetros base
        params = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": signal.get("order_type", "market"),
            "time_in_force": signal.get("time_in_force", "day"),
            "client_order_id": f"signal_{int(time.time())}_{symbol}"
        }
        
        # Añadir precio límite si es necesario
        if params["type"] in ["limit", "stop_limit"] and "limit_price" in signal:
            params["limit_price"] = signal["limit_price"]
        
        # Añadir precio stop si es necesario
        if params["type"] in ["stop", "stop_limit"] and "stop_price" in signal:
            params["stop_price"] = signal["stop_price"]
        
        # Configurar take profit y stop loss si están especificados
        take_profit_pct = signal.get("take_profit_pct")
        stop_loss_pct = signal.get("stop_loss_pct")
        
        if take_profit_pct or stop_loss_pct:
            # Obtener precio actual
            latest_quote = self.alpaca_client.get_latest_quote(symbol)
            current_price = latest_quote["askprice"] if side == "buy" else latest_quote["bidprice"]
            
            # Calcular precios de take profit y stop loss
            if take_profit_pct:
                if side == "buy":
                    params["take_profit"] = round(current_price * (1 + take_profit_pct/100), 2)
                else:
                    params["take_profit"] = round(current_price * (1 - take_profit_pct/100), 2)
            
            if stop_loss_pct:
                if side == "buy":
                    params["stop_loss"] = round(current_price * (1 - stop_loss_pct/100), 2)
                else:
                    params["stop_loss"] = round(current_price * (1 + stop_loss_pct/100), 2)
        
        return params
    
    def _register_trade(self, symbol: str, side: str, qty: float, order_result: Dict[str, Any]) -> None:
        """
        Registra una operación en el historial interno.
        
        Args:
            symbol: Símbolo operado
            side: Dirección de la operación
            qty: Cantidad operada
            order_result: Resultado de la orden
        """
        self.last_trades[symbol] = {
            "timestamp": datetime.now(),
            "side": side,
            "qty": qty,
            "order_id": order_result.get("id"),
            "price": order_result.get("filled_avg_price", order_result.get("limit_price", 0))
        }
        
        # Guardar en órdenes activas si es necesario
        if order_result.get("status") in ["new", "accepted", "held"]:
            self.active_orders[order_result.get("id")] = {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "timestamp": datetime.now(),
                "order": order_result
            }
    
    def _send_trade_notification(self, symbol: str, side: str, qty: float, 
                               params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Envía notificación de operación ejecutada.
        
        Args:
            symbol: Símbolo operado
            side: Dirección de la operación
            qty: Cantidad operada
            params: Parámetros de la orden
            result: Resultado de la orden
        """
        # Crear mensaje
        emoji = "🟢" if side == "buy" else "🔴"
        message = (
            f"{emoji} *OPERACIÓN EJECUTADA*\n"
            f"Símbolo: `{symbol}`\n"
            f"Acción: `{side.upper()}`\n"
            f"Cantidad: `{qty}`\n"
            f"Tipo: `{params.get('type', 'market')}`\n"
            f"Estado: `{result.get('status', 'desconocido')}`\n"
            f"ID: `{result.get('id', 'N/A')}`\n"
        )
        
        if "take_profit" in params:
            message += f"Take Profit: `{params['take_profit']}`\n"
        
        if "stop_loss" in params:
            message += f"Stop Loss: `{params['stop_loss']}`\n"
        
        # Enviar notificaciones
        self._send_notification(message)
    
    def _send_error_notification(self, symbol: str, side: str, qty: float, 
                               params: Dict[str, Any], error: str) -> None:
        """
        Envía notificación de error en operación.
        
        Args:
            symbol: Símbolo operado
            side: Dirección de la operación
            qty: Cantidad operada
            params: Parámetros de la orden
            error: Mensaje de error
        """
        # Crear mensaje
        message = (
            f"⚠️ *ERROR EN OPERACIÓN*\n"
            f"Símbolo: `{symbol}`\n"
            f"Acción: `{side.upper()}`\n"
            f"Cantidad: `{qty}`\n"
            f"Tipo: `{params.get('type', 'market')}`\n"
            f"Error: `{error}`\n"
        )
        
        # Enviar notificaciones
        self._send_notification(message)
    
    def _send_notification(self, message: str) -> None:
        """
        Envía notificación a los canales configurados.
        
        Args:
            message: Mensaje a enviar
        """
        # Enviar a Telegram si está habilitado
        if TELEGRAM_ENABLED and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
            try:
                self.alert_manager.send_telegram_message(message)
            except Exception as e:
                logger.error(f"Error al enviar notificación a Telegram: {str(e)}")
        
        # Enviar a Discord si está habilitado
        if DISCORD_ENABLED and DISCORD_WEBHOOK_URL:
            try:
                self.alert_manager.send_discord_message(message)
            except Exception as e:
                logger.error(f"Error al enviar notificación a Discord: {str(e)}")
    
    def update_orders_status(self) -> Dict[str, Any]:
        """
        Actualiza el estado de las órdenes activas.
        
        Returns:
            Diccionario con resumen de actualizaciones
        """
        with self.lock:
            if not self.active_orders:
                return {"updated": 0, "completed": 0, "canceled": 0}
            
            orders_to_remove = []
            stats = {"updated": 0, "completed": 0, "canceled": 0}
            
            for order_id, order_info in self.active_orders.items():
                try:
                    # Obtener estado actualizado
                    updated_order = self.alpaca_client.get_order(order_id)
                    
                    # Actualizar información
                    order_info["order"] = updated_order
                    stats["updated"] += 1
                    
                    # Verificar si la orden está completa o cancelada
                    if updated_order["status"] in ["filled", "canceled", "expired", "rejected"]:
                        orders_to_remove.append(order_id)
                        
                        if updated_order["status"] == "filled":
                            stats["completed"] += 1
                        else:
                            stats["canceled"] += 1
                
                except Exception as e:
                    logger.error(f"Error al actualizar orden {order_id}: {str(e)}")
            
            # Eliminar órdenes completadas o canceladas
            for order_id in orders_to_remove:
                self.active_orders.pop(order_id, None)
            
            return stats
    
    def get_active_orders(self) -> List[Dict[str, Any]]:
        """
        Obtiene la lista de órdenes activas.
        
        Returns:
            Lista de órdenes activas
        """
        with self.lock:
            return [
                {
                    "order_id": order_id,
                    "symbol": info["symbol"],
                    "side": info["side"],
                    "qty": info["qty"],
                    "status": info["order"].get("status", "unknown"),
                    "type": info["order"].get("type", "unknown"),
                    "submitted_at": info["order"].get("submitted_at", "unknown")
                }
                for order_id, info in self.active_orders.items()
            ]
    
    def cancel_all_orders(self) -> Dict[str, Any]:
        """
        Cancela todas las órdenes activas.
        
        Returns:
            Resultado de la operación
        """
        try:
            result = self.alpaca_client.cancel_all_orders()
            self.active_orders = {}
            return {"success": True, "canceled": result}
        except Exception as e:
            logger.error(f"Error al cancelar órdenes: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def auto_hedge(self, symbol: str, hedge_symbol: str = None) -> Dict[str, Any]:
        """
        Implementa una estrategia de auto-hedging usando ETFs inversos.
        
        Args:
            symbol: Símbolo principal
            hedge_symbol: Símbolo para hedging (opcional)
            
        Returns:
            Resultado de la operación de hedging
        """
        # Implementación básica - se puede expandir según necesidades
        try:
            # Obtener posición actual
            position = self.alpaca_client.get_position(symbol)
            
            # Si no hay posición, no hay nada que hacer
            if not position:
                return {"success": True, "message": "No hay posición para hacer hedging"}
            
            # Determinar símbolo de hedging si no se proporciona
            if not hedge_symbol:
                # Mapeo simple de símbolos a ETFs inversos
                hedge_map = {
                    "SPY": "SH",
                    "QQQ": "PSQ",
                    "IWM": "RWM",
                    # Añadir más mapeos según sea necesario
                }
                hedge_symbol = hedge_map.get(symbol)
                
                if not hedge_symbol:
                    return {"success": False, "error": f"No se encontró ETF inverso para {symbol}"}
            
            # Calcular cantidad para hedging (misma exposición en dólares)
            qty = float(position["qty"])
            market_value = float(position["market_value"])
            
            # Obtener precio del instrumento de hedging
            hedge_quote = self.alpaca_client.get_latest_quote(hedge_symbol)
            hedge_price = hedge_quote["askprice"]
            
            # Calcular cantidad de hedging
            hedge_qty = abs(market_value) / hedge_price
            hedge_qty = round(hedge_qty, 2)
            
            # Determinar dirección del hedging (opuesta a la posición principal)
            hedge_side = "buy" if float(position["qty"]) < 0 else "sell"
            
            # Crear orden de hedging
            hedge_params = {
                "symbol": hedge_symbol,
                "qty": hedge_qty,
                "side": hedge_side,
                "type": "market",
                "time_in_force": "day",
                "client_order_id": f"hedge_{int(time.time())}_{symbol}"
            }
            
            # Ejecutar orden
            hedge_order = self.alpaca_client.submit_order(**hedge_params)
            
            # Notificar
            self._send_notification(
                f"🛡️ *AUTO-HEDGING*\n"
                f"Posición: `{symbol} {position['qty']}`\n"
                f"Hedge: `{hedge_symbol} {hedge_qty} {hedge_side}`\n"
                f"Orden ID: `{hedge_order.get('id', 'N/A')}`"
            )
            
            return {
                "success": True,
                "position": position,
                "hedge_order": hedge_order,
                "hedge_params": hedge_params
            }
            
        except Exception as e:
            logger.error(f"Error en auto-hedging: {str(e)}")
            return {"success": False, "error": str(e)}