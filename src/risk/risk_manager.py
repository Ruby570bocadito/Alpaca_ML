#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para la gestión de riesgo en operaciones de trading.
"""

import logging
import time
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

# Importaciones internas
from ..execution.alpaca_client import AlpacaClient
from ..execution.order_manager import OrderManager

logger = logging.getLogger(__name__)


class RiskManager:
    """Clase para gestionar el riesgo en operaciones de trading."""

    def __init__(self, config: Dict[str, Any], alpaca_client: AlpacaClient, order_manager: OrderManager):
        """Inicializa el gestor de riesgo.

        Args:
            config: Configuración del sistema
            alpaca_client: Cliente de Alpaca
            order_manager: Gestor de órdenes
        """
        self.config = config
        self.alpaca_client = alpaca_client
        self.order_manager = order_manager
        
        # Configuración de riesgo
        self.risk_config = {
            "max_position_size_pct": float(config.get("MAX_POSITION_SIZE_PCT", "5.0")),
            "max_single_order_size_pct": float(config.get("MAX_SINGLE_ORDER_SIZE_PCT", "2.0")),
            "max_total_positions": int(config.get("MAX_TOTAL_POSITIONS", "10")),
            "max_drawdown_pct": float(config.get("MAX_DRAWDOWN_PCT", "5.0")),
            "max_daily_loss_pct": float(config.get("MAX_DAILY_LOSS_PCT", "3.0")),
            "max_daily_trades": int(config.get("MAX_DAILY_TRADES", "20")),
            "max_position_concentration_pct": float(config.get("MAX_POSITION_CONCENTRATION_PCT", "20.0")),
            "stop_loss_pct": float(config.get("DEFAULT_STOP_LOSS_PCT", "2.0")),
            "trailing_stop_activation_pct": float(config.get("TRAILING_STOP_ACTIVATION_PCT", "1.0")),
            "trailing_stop_distance_pct": float(config.get("TRAILING_STOP_DISTANCE_PCT", "0.5")),
            "circuit_breaker_enabled": config.get("CIRCUIT_BREAKER_ENABLED", "true").lower() == "true",
        }
        
        # Estado del gestor de riesgo
        self.risk_state = {
            "initial_equity": 0.0,
            "current_equity": 0.0,
            "daily_starting_equity": 0.0,
            "daily_high_equity": 0.0,
            "daily_low_equity": 0.0,
            "max_equity": 0.0,
            "drawdown_pct": 0.0,
            "daily_loss_pct": 0.0,
            "daily_trades_count": 0,
            "circuit_breaker_triggered": False,
            "last_equity_check": datetime.now().isoformat(),
            "positions_value_by_symbol": {},
            "trailing_stops": {},
        }
        
        # Inicializar estado
        self._initialize_risk_state()
        
        logger.info("Gestor de riesgo inicializado")

    def _initialize_risk_state(self):
        """Inicializa el estado del gestor de riesgo."""
        try:
            # Obtener información de la cuenta
            account = self.alpaca_client.get_account()
            
            if account and "equity" in account:
                equity = float(account["equity"])
                self.risk_state["initial_equity"] = equity
                self.risk_state["current_equity"] = equity
                self.risk_state["daily_starting_equity"] = equity
                self.risk_state["daily_high_equity"] = equity
                self.risk_state["daily_low_equity"] = equity
                self.risk_state["max_equity"] = equity
                
                logger.info(f"Estado de riesgo inicializado con equity: {equity}")
                
                # Obtener posiciones actuales
                positions = self.alpaca_client.get_positions()
                
                for position in positions:
                    symbol = position["symbol"]
                    market_value = float(position["market_value"])
                    self.risk_state["positions_value_by_symbol"][symbol] = market_value
                
                logger.info(f"Posiciones iniciales cargadas: {len(positions)}")
            else:
                logger.error("No se pudo obtener información de la cuenta")
                
        except Exception as e:
            logger.error(f"Error al inicializar estado de riesgo: {e}", exc_info=True)

    def update_risk_state(self):
        """Actualiza el estado del gestor de riesgo."""
        try:
            # Obtener información de la cuenta
            account = self.alpaca_client.get_account()
            
            if account and "equity" in account:
                current_equity = float(account["equity"])
                previous_equity = self.risk_state["current_equity"]
                
                self.risk_state["current_equity"] = current_equity
                self.risk_state["last_equity_check"] = datetime.now().isoformat()
                
                # Actualizar máximos y mínimos
                if current_equity > self.risk_state["max_equity"]:
                    self.risk_state["max_equity"] = current_equity
                
                if current_equity > self.risk_state["daily_high_equity"]:
                    self.risk_state["daily_high_equity"] = current_equity
                
                if current_equity < self.risk_state["daily_low_equity"]:
                    self.risk_state["daily_low_equity"] = current_equity
                
                # Calcular drawdown
                if self.risk_state["max_equity"] > 0:
                    self.risk_state["drawdown_pct"] = (self.risk_state["max_equity"] - current_equity) / self.risk_state["max_equity"] * 100
                
                # Calcular pérdida diaria
                if self.risk_state["daily_starting_equity"] > 0:
                    self.risk_state["daily_loss_pct"] = (self.risk_state["daily_starting_equity"] - current_equity) / self.risk_state["daily_starting_equity"] * 100
                    if current_equity > self.risk_state["daily_starting_equity"]:
                        self.risk_state["daily_loss_pct"] = 0  # No hay pérdida si hay ganancia
                
                # Actualizar posiciones
                positions = self.alpaca_client.get_positions()
                self.risk_state["positions_value_by_symbol"] = {}
                
                for position in positions:
                    symbol = position["symbol"]
                    market_value = float(position["market_value"])
                    self.risk_state["positions_value_by_symbol"][symbol] = market_value
                    
                    # Actualizar trailing stops si existen
                    if symbol in self.risk_state["trailing_stops"]:
                        self._update_trailing_stop(symbol, position)
                
                # Verificar circuit breaker
                self._check_circuit_breaker()
                
                logger.debug(f"Estado de riesgo actualizado: equity={current_equity}, drawdown={self.risk_state['drawdown_pct']:.2f}%, "
                          f"pérdida diaria={self.risk_state['daily_loss_pct']:.2f}%")
                
                return True
            else:
                logger.error("No se pudo obtener información de la cuenta")
                return False
                
        except Exception as e:
            logger.error(f"Error al actualizar estado de riesgo: {e}", exc_info=True)
            return False

    def _check_circuit_breaker(self):
        """Verifica si se debe activar el circuit breaker."""
        if not self.risk_config["circuit_breaker_enabled"]:
            return False
        
        # Verificar drawdown máximo
        if self.risk_state["drawdown_pct"] > self.risk_config["max_drawdown_pct"]:
            logger.warning(f"Circuit breaker activado: drawdown {self.risk_state['drawdown_pct']:.2f}% excede máximo "
                         f"permitido {self.risk_config['max_drawdown_pct']}%")
            self.risk_state["circuit_breaker_triggered"] = True
            self._execute_circuit_breaker()
            return True
        
        # Verificar pérdida diaria máxima
        if self.risk_state["daily_loss_pct"] > self.risk_config["max_daily_loss_pct"]:
            logger.warning(f"Circuit breaker activado: pérdida diaria {self.risk_state['daily_loss_pct']:.2f}% excede máximo "
                         f"permitido {self.risk_config['max_daily_loss_pct']}%")
            self.risk_state["circuit_breaker_triggered"] = True
            self._execute_circuit_breaker()
            return True
        
        return False

    def _execute_circuit_breaker(self):
        """Ejecuta el circuit breaker (cierra todas las posiciones)."""
        try:
            logger.warning("Ejecutando circuit breaker: cerrando todas las posiciones")
            
            # Cancelar todas las órdenes pendientes
            self.order_manager.cancel_all_orders()
            
            # Cerrar todas las posiciones
            self.alpaca_client.close_all_positions()
            
            # Notificar
            logger.critical("CIRCUIT BREAKER ACTIVADO: Todas las posiciones cerradas y órdenes canceladas")
            
            return True
            
        except Exception as e:
            logger.error(f"Error al ejecutar circuit breaker: {e}", exc_info=True)
            return False

    def reset_daily_metrics(self):
        """Reinicia las métricas diarias."""
        try:
            # Obtener equity actual
            account = self.alpaca_client.get_account()
            
            if account and "equity" in account:
                current_equity = float(account["equity"])
                
                self.risk_state["daily_starting_equity"] = current_equity
                self.risk_state["daily_high_equity"] = current_equity
                self.risk_state["daily_low_equity"] = current_equity
                self.risk_state["daily_loss_pct"] = 0.0
                self.risk_state["daily_trades_count"] = 0
                
                logger.info(f"Métricas diarias reiniciadas: equity={current_equity}")
                return True
            else:
                logger.error("No se pudo obtener información de la cuenta")
                return False
                
        except Exception as e:
            logger.error(f"Error al reiniciar métricas diarias: {e}", exc_info=True)
            return False

    def check_order_risk(self, order_params: Dict[str, Any]) -> Tuple[bool, str]:
        """Verifica si una orden cumple con los criterios de riesgo.

        Args:
            order_params: Parámetros de la orden

        Returns:
            Tuple[bool, str]: (orden aprobada, mensaje)
        """
        # Verificar si el circuit breaker está activado
        if self.risk_state["circuit_breaker_triggered"]:
            return False, "Circuit breaker activado, no se permiten nuevas órdenes"
        
        # Verificar número máximo de operaciones diarias
        if self.risk_state["daily_trades_count"] >= self.risk_config["max_daily_trades"]:
            return False, f"Número máximo de operaciones diarias alcanzado: {self.risk_config['max_daily_trades']}"
        
        # Verificar tamaño máximo de orden
        try:
            symbol = order_params.get("symbol")
            qty = float(order_params.get("qty", 0))
            side = order_params.get("side")
            
            if not symbol or qty <= 0 or not side:
                return False, "Parámetros de orden incompletos"
            
            # Obtener último precio
            latest_trade = self.alpaca_client.get_latest_trade(symbol)
            if not latest_trade or "price" not in latest_trade:
                return False, f"No se pudo obtener precio actual para {symbol}"
            
            price = latest_trade["price"]
            order_value = price * qty
            
            # Verificar tamaño máximo de orden como porcentaje del equity
            account = self.alpaca_client.get_account()
            if not account or "equity" not in account:
                return False, "No se pudo obtener información de la cuenta"
            
            equity = float(account["equity"])
            order_size_pct = (order_value / equity) * 100
            
            if order_size_pct > self.risk_config["max_single_order_size_pct"]:
                return False, f"Tamaño de orden ({order_size_pct:.2f}%) excede máximo permitido ({self.risk_config['max_single_order_size_pct']}%)"
            
            # Verificar número máximo de posiciones
            positions = self.alpaca_client.get_positions()
            current_positions = len(positions)
            
            # Si es una nueva posición (no tenemos el símbolo actualmente)
            has_position = any(p["symbol"] == symbol for p in positions)
            if not has_position and side == "buy" and current_positions >= self.risk_config["max_total_positions"]:
                return False, f"Número máximo de posiciones alcanzado: {self.risk_config['max_total_positions']}"
            
            # Verificar concentración máxima por posición
            if side == "buy":
                # Calcular valor total de la posición después de la orden
                current_position_value = 0
                for p in positions:
                    if p["symbol"] == symbol:
                        current_position_value = float(p["market_value"])
                        break
                
                new_position_value = current_position_value + order_value
                position_concentration_pct = (new_position_value / equity) * 100
                
                if position_concentration_pct > self.risk_config["max_position_concentration_pct"]:
                    return False, f"Concentración de posición ({position_concentration_pct:.2f}%) excede máximo permitido ({self.risk_config['max_position_concentration_pct']}%)"
            
            # Incrementar contador de operaciones diarias
            self.risk_state["daily_trades_count"] += 1
            
            return True, "Orden aprobada"
            
        except Exception as e:
            logger.error(f"Error al verificar riesgo de orden: {e}", exc_info=True)
            return False, f"Error al verificar riesgo: {str(e)}"

    def set_stop_loss(self, symbol: str, position_qty: float, entry_price: float, stop_loss_pct: Optional[float] = None) -> Dict[str, Any]:
        """Establece un stop loss para una posición.

        Args:
            symbol: Símbolo de la posición
            position_qty: Cantidad de la posición
            entry_price: Precio de entrada
            stop_loss_pct: Porcentaje de stop loss (opcional)

        Returns:
            Dict[str, Any]: Resultado de la operación
        """
        try:
            if stop_loss_pct is None:
                stop_loss_pct = self.risk_config["stop_loss_pct"]
            
            # Determinar side (opuesto a la posición)
            side = "sell" if position_qty > 0 else "buy"
            qty = abs(position_qty)
            
            # Calcular precio de stop loss
            if side == "sell":  # Posición larga, stop loss por debajo
                stop_price = round(entry_price * (1 - stop_loss_pct / 100), 2)
                limit_price = round(stop_price * 0.99, 2)  # 1% por debajo para asegurar ejecución
            else:  # Posición corta, stop loss por encima
                stop_price = round(entry_price * (1 + stop_loss_pct / 100), 2)
                limit_price = round(stop_price * 1.01, 2)  # 1% por encima para asegurar ejecución
            
            # Crear orden de stop loss
            order_params = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "type": "stop_limit",
                "time_in_force": "gtc",  # Good till cancelled
                "stop_price": stop_price,
                "limit_price": limit_price,
                "client_order_id": f"sl_{symbol}_{int(time.time())}"
            }
            
            result = self.order_manager.submit_order(order_params)
            
            if result.get("status") in ["submitted", "accepted"]:
                logger.info(f"Stop loss establecido para {symbol}: precio={stop_price}, qty={qty}")
                return {
                    "status": "success",
                    "message": "Stop loss establecido",
                    "symbol": symbol,
                    "stop_price": stop_price,
                    "order_id": result.get("order_id"),
                }
            else:
                logger.error(f"Error al establecer stop loss: {result}")
                return {
                    "status": "error",
                    "message": f"Error al establecer stop loss: {result.get('message', 'Error desconocido')}",
                }
            
        except Exception as e:
            logger.error(f"Error al establecer stop loss: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def set_trailing_stop(self, symbol: str, position_qty: float, activation_pct: Optional[float] = None, 
                         trail_pct: Optional[float] = None) -> Dict[str, Any]:
        """Establece un trailing stop para una posición.

        Args:
            symbol: Símbolo de la posición
            position_qty: Cantidad de la posición
            activation_pct: Porcentaje de activación (opcional)
            trail_pct: Porcentaje de trailing (opcional)

        Returns:
            Dict[str, Any]: Resultado de la operación
        """
        try:
            if activation_pct is None:
                activation_pct = self.risk_config["trailing_stop_activation_pct"]
                
            if trail_pct is None:
                trail_pct = self.risk_config["trailing_stop_distance_pct"]
            
            # Obtener último precio
            latest_trade = self.alpaca_client.get_latest_trade(symbol)
            if not latest_trade or "price" not in latest_trade:
                return {"status": "error", "message": f"No se pudo obtener precio actual para {symbol}"}
            
            current_price = latest_trade["price"]
            
            # Determinar side (opuesto a la posición)
            side = "sell" if position_qty > 0 else "buy"
            qty = abs(position_qty)
            
            # Calcular precio de activación
            if side == "sell":  # Posición larga, trailing stop por debajo
                activation_price = round(current_price * (1 + activation_pct / 100), 2)
                trail_value = round(current_price * (trail_pct / 100), 2)
            else:  # Posición corta, trailing stop por encima
                activation_price = round(current_price * (1 - activation_pct / 100), 2)
                trail_value = round(current_price * (trail_pct / 100), 2)
            
            # Guardar información del trailing stop
            self.risk_state["trailing_stops"][symbol] = {
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "activation_price": activation_price,
                "trail_value": trail_value,
                "trail_percent": trail_pct,
                "current_price": current_price,
                "highest_price": current_price if side == "sell" else 0,
                "lowest_price": current_price if side == "buy" else float('inf'),
                "activated": False,
                "stop_price": None,
                "order_id": None,
                "created_at": datetime.now().isoformat(),
            }
            
            logger.info(f"Trailing stop configurado para {symbol}: activación={activation_price}, trail={trail_pct}%")
            
            return {
                "status": "success",
                "message": "Trailing stop configurado",
                "symbol": symbol,
                "activation_price": activation_price,
                "trail_percent": trail_pct,
            }
            
        except Exception as e:
            logger.error(f"Error al establecer trailing stop: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}

    def _update_trailing_stop(self, symbol: str, position: Dict[str, Any]):
        """Actualiza un trailing stop existente.

        Args:
            symbol: Símbolo de la posición
            position: Información de la posición
        """
        if symbol not in self.risk_state["trailing_stops"]:
            return
        
        try:
            ts = self.risk_state["trailing_stops"][symbol]
            
            # Obtener último precio
            latest_trade = self.alpaca_client.get_latest_trade(symbol)
            if not latest_trade or "price" not in latest_trade:
                logger.warning(f"No se pudo obtener precio actual para trailing stop de {symbol}")
                return
            
            current_price = latest_trade["price"]
            ts["current_price"] = current_price
            
            # Actualizar máximos/mínimos
            if ts["side"] == "sell":  # Posición larga
                if current_price > ts["highest_price"]:
                    ts["highest_price"] = current_price
            else:  # Posición corta
                if current_price < ts["lowest_price"]:
                    ts["lowest_price"] = current_price
            
            # Verificar activación
            if not ts["activated"]:
                if (ts["side"] == "sell" and current_price >= ts["activation_price"]) or \
                   (ts["side"] == "buy" and current_price <= ts["activation_price"]):
                    ts["activated"] = True
                    logger.info(f"Trailing stop para {symbol} activado a precio {current_price}")
            
            # Si está activado, verificar si debe ejecutarse
            if ts["activated"]:
                if ts["side"] == "sell":  # Posición larga
                    stop_price = ts["highest_price"] * (1 - ts["trail_percent"] / 100)
                    
                    # Si el precio ha caído por debajo del stop, ejecutar
                    if current_price <= stop_price and ts["order_id"] is None:
                        self._execute_trailing_stop(symbol)
                        
                else:  # Posición corta
                    stop_price = ts["lowest_price"] * (1 + ts["trail_percent"] / 100)
                    
                    # Si el precio ha subido por encima del stop, ejecutar
                    if current_price >= stop_price and ts["order_id"] is None:
                        self._execute_trailing_stop(symbol)
            
        except Exception as e:
            logger.error(f"Error al actualizar trailing stop para {symbol}: {e}", exc_info=True)

    def _execute_trailing_stop(self, symbol: str):
        """Ejecuta un trailing stop.

        Args:
            symbol: Símbolo de la posición
        """
        if symbol not in self.risk_state["trailing_stops"]:
            return
        
        try:
            ts = self.risk_state["trailing_stops"][symbol]
            
            # Crear orden de mercado para cerrar la posición
            order_params = {
                "symbol": symbol,
                "qty": ts["qty"],
                "side": ts["side"],
                "type": "market",
                "time_in_force": "day",
                "client_order_id": f"ts_{symbol}_{int(time.time())}"
            }
            
            result = self.order_manager.submit_order(order_params)
            
            if result.get("status") in ["submitted", "accepted"]:
                ts["order_id"] = result.get("order_id")
                ts["executed_at"] = datetime.now().isoformat()
                ts["stop_price"] = ts["current_price"]
                
                logger.info(f"Trailing stop ejecutado para {symbol} a precio {ts['current_price']}")
                
                # Eliminar el trailing stop después de ejecutarlo
                # self.risk_state["trailing_stops"].pop(symbol, None)
            else:
                logger.error(f"Error al ejecutar trailing stop para {symbol}: {result}")
            
        except Exception as e:
            logger.error(f"Error al ejecutar trailing stop para {symbol}: {e}", exc_info=True)

    def get_risk_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas de riesgo actuales.

        Returns:
            Dict[str, Any]: Métricas de riesgo
        """
        # Actualizar estado primero
        self.update_risk_state()
        
        # Calcular métricas adicionales
        try:
            account = self.alpaca_client.get_account()
            positions = self.alpaca_client.get_positions()
            
            # Calcular exposición total
            total_position_value = sum(float(p["market_value"]) for p in positions)
            total_exposure_pct = 0
            
            if account and "equity" in account:
                equity = float(account["equity"])
                total_exposure_pct = (total_position_value / equity) * 100 if equity > 0 else 0
            
            # Calcular exposición por posición
            position_exposures = []
            for p in positions:
                symbol = p["symbol"]
                market_value = float(p["market_value"])
                exposure_pct = (market_value / equity) * 100 if equity > 0 else 0
                
                position_exposures.append({
                    "symbol": symbol,
                    "market_value": market_value,
                    "exposure_pct": exposure_pct,
                    "side": "long" if float(p["qty"]) > 0 else "short",
                    "unrealized_pl_pct": float(p["unrealized_pl_pct"]) if "unrealized_pl_pct" in p else 0,
                })
            
            # Ordenar por exposición
            position_exposures.sort(key=lambda x: x["exposure_pct"], reverse=True)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "equity": self.risk_state["current_equity"],
                "initial_equity": self.risk_state["initial_equity"],
                "daily_starting_equity": self.risk_state["daily_starting_equity"],
                "drawdown_pct": self.risk_state["drawdown_pct"],
                "daily_loss_pct": self.risk_state["daily_loss_pct"],
                "daily_trades_count": self.risk_state["daily_trades_count"],
                "circuit_breaker_triggered": self.risk_state["circuit_breaker_triggered"],
                "total_positions": len(positions),
                "total_position_value": total_position_value,
                "total_exposure_pct": total_exposure_pct,
                "position_exposures": position_exposures,
                "trailing_stops_active": len([ts for ts in self.risk_state["trailing_stops"].values() if ts["activated"] and ts["order_id"] is None]),
            }
            
        except Exception as e:
            logger.error(f"Error al obtener métricas de riesgo: {e}", exc_info=True)
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                **{k: v for k, v in self.risk_state.items() if k not in ["positions_value_by_symbol", "trailing_stops"]},
            }

    def validate_signals(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Filtra las señales basándose en las reglas de riesgo pre-trade.

        Args:
            signals: Diccionario de señales a validar.

        Returns:
            Diccionario de señales aprobadas.
        """
        if self.risk_state["circuit_breaker_triggered"]:
            logger.warning("Circuit breaker activado, se rechazan todas las nuevas señales.")
            return {}

        if self.risk_state["daily_trades_count"] >= self.risk_config["max_daily_trades"]:
            logger.warning(f"Se alcanzó el máximo de operaciones diarias ({self.risk_config['max_daily_trades']}). Se rechazan nuevas señales.")
            return {}

        # Por ahora, se aprueban todas las señales si no se activan las reglas anteriores.
        # Se pueden añadir más validaciones aquí.
        approved_signals = signals
        logger.info(f"Señales validadas. Aprobadas: {len(approved_signals)} de {len(signals)}.")

        return approved_signals