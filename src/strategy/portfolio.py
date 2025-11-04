#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para la gestión de cartera y determinación de tamaños de posición.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Clase para gestionar la cartera y determinar tamaños de posición."""

    def __init__(self, config: Dict[str, Any]):
        """Inicializa el gestor de cartera.

        Args:
            config: Configuración del sistema
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Configuración de cartera
        self.portfolio_config = {
            "max_positions": int(config.get("MAX_POSITIONS", "5")),
            "position_sizing_type": config.get("POSITION_SIZING_TYPE", "equal"),
            "max_position_size_pct": float(config.get("MAX_POSITION_SIZE_PCT", "20")),
            "risk_per_trade_pct": float(config.get("RISK_PER_TRADE_PCT", "1")),
            "max_leverage": float(config.get("MAX_LEVERAGE", "1")),
            "max_drawdown_pct": float(config.get("MAX_DRAWDOWN_PCT", "10")),
            "allow_shorts": config.get("ALLOW_SHORTS", "false").lower() == "true",
        }

        self.logger.info("Gestor de cartera inicializado")

    def calculate_position_sizes(self, signals: Dict[str, Dict[str, Any]],
                               account_info: Dict[str, Any],
                               current_positions: Dict[str, float],
                               market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Calcula tamaños de posición para las señales generadas.

        Args:
            signals: Diccionario con señales por símbolo
            account_info: Información de la cuenta (equity, cash, etc.)
            current_positions: Posiciones actuales
            market_data: Datos de mercado recientes por símbolo

        Returns:
            Dict[str, Dict[str, Any]]: Órdenes con tamaños calculados
        """
        # Verificar que tenemos información de cuenta
        if not account_info or "equity" not in account_info:
            logger.error("No se proporcionó información de cuenta válida")
            return {}
        
        equity = float(account_info["equity"])
        cash = float(account_info.get("cash", equity))
        
        # Filtrar señales por confianza y priorizar
        filtered_signals = self._filter_and_prioritize_signals(signals)
        
        # Limitar número de posiciones
        max_new_positions = self.portfolio_config["max_positions"] - len([p for p in current_positions.values() if p > 0])
        if max_new_positions <= 0:
            logger.info("Ya se alcanzó el número máximo de posiciones")
            return {}

        # Limitar señales al número máximo de nuevas posiciones
        top_signals = dict(list(filtered_signals.items())[:max_new_positions])
        
        # Calcular tamaños de posición según el método configurado
        if self.portfolio_config["position_sizing_type"] == "equal":
            orders = self._calculate_equal_position_sizes(top_signals, equity, cash, current_positions, market_data)
        elif self.portfolio_config["position_sizing_type"] == "risk_parity":
            orders = self._calculate_risk_parity_position_sizes(top_signals, equity, cash, current_positions, market_data)
        elif self.portfolio_config["position_sizing_type"] == "kelly":
            orders = self._calculate_kelly_position_sizes(top_signals, equity, cash, current_positions, market_data)
        else:
            logger.warning(f"Tipo de sizing desconocido: {self.portfolio_config['position_sizing_type']}")
            orders = self._calculate_equal_position_sizes(top_signals, equity, cash, current_positions, market_data)
        
        logger.info(f"Calculados tamaños para {len(orders)} órdenes")
        return orders

    def _filter_and_prioritize_signals(self, signals: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Filtra y prioriza señales según confianza.

        Args:
            signals: Diccionario con señales por símbolo

        Returns:
            Dict[str, Dict[str, Any]]: Señales filtradas y ordenadas
        """
        # Filtrar señales por confianza mínima
        min_confidence = float(self.config.get("SIGNAL_MIN_CONFIDENCE", "0.55"))
        filtered = {symbol: signal for symbol, signal in signals.items() 
                   if signal["confidence"] >= min_confidence}
        
        # No permitir señales de venta si no se permiten posiciones cortas
        if not self.portfolio_config["allow_shorts"]:
            filtered = {symbol: signal for symbol, signal in filtered.items() 
                       if signal["direction"] != "sell"}
        
        # Ordenar por confianza (descendente)
        sorted_signals = dict(sorted(filtered.items(), 
                                    key=lambda x: x[1]["confidence"], 
                                    reverse=True))
        
        return sorted_signals

    def _calculate_equal_position_sizes(self, signals: Dict[str, Dict[str, Any]],
                                      equity: float, cash: float,
                                      current_positions: Dict[str, float],
                                      market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Calcula tamaños de posición iguales para todas las señales.

        Args:
            signals: Diccionario con señales por símbolo
            equity: Capital total de la cuenta
            cash: Efectivo disponible
            current_positions: Posiciones actuales
            market_data: Datos de mercado recientes por símbolo

        Returns:
            Dict[str, Dict[str, Any]]: Órdenes con tamaños calculados
        """
        orders = {}
        
        if not signals:
            return orders
        
        # Calcular capital disponible por posición
        max_position_size = equity * (self.portfolio_config["max_position_size_pct"] / 100)
        position_equity = min(equity / len(signals), max_position_size)
        
        for symbol, signal in signals.items():
            try:
                # Verificar que tenemos datos de mercado
                if symbol not in market_data:
                    logger.warning(f"No hay datos de mercado para {symbol}")
                    continue
                
                # Obtener último precio
                last_price = market_data[symbol]["close"].iloc[-1]
                
                # Calcular cantidad de acciones
                qty = int(position_equity / last_price)
                
                # Ajustar dirección
                if signal["direction"] == "sell":
                    qty = -qty
                
                # Verificar cantidad mínima
                if abs(qty) < 1:
                    logger.warning(f"Cantidad calculada para {symbol} es menor que 1: {qty}")
                    continue
                
                # Crear orden
                orders[symbol] = {
                    "symbol": symbol,
                    "qty": qty,
                    "side": "buy" if qty > 0 else "sell",
                    "type": "market",
                    "time_in_force": "day",
                    "position_value": qty * last_price,
                    "signal_confidence": signal["confidence"],
                }
                
            except Exception as e:
                logger.error(f"Error al calcular tamaño para {symbol}: {e}", exc_info=True)
        
        return orders

    def _calculate_risk_parity_position_sizes(self, signals: Dict[str, Dict[str, Any]],
                                            equity: float, cash: float,
                                            current_positions: Dict[str, float],
                                            market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Calcula tamaños de posición basados en paridad de riesgo.

        Args:
            signals: Diccionario con señales por símbolo
            equity: Capital total de la cuenta
            cash: Efectivo disponible
            current_positions: Posiciones actuales
            market_data: Datos de mercado recientes por símbolo

        Returns:
            Dict[str, Dict[str, Any]]: Órdenes con tamaños calculados
        """
        orders = {}
        
        if not signals:
            return orders
        
        # Calcular volatilidad para cada símbolo
        volatilities = {}
        for symbol in signals.keys():
            if symbol not in market_data:
                continue
            
            # Calcular volatilidad como desviación estándar de retornos diarios
            returns = market_data[symbol]["close"].pct_change().dropna()
            if len(returns) > 5:  # Necesitamos al menos 5 días de datos
                volatilities[symbol] = returns.std()
            else:
                volatilities[symbol] = 0.02  # Valor por defecto si no hay suficientes datos
        
        # Calcular pesos inversos a la volatilidad
        if not volatilities:
            return self._calculate_equal_position_sizes(signals, equity, cash, current_positions, market_data)
        
        inverse_vol = {symbol: 1/vol if vol > 0 else 0 for symbol, vol in volatilities.items()}
        total_inverse_vol = sum(inverse_vol.values())
        
        if total_inverse_vol == 0:
            return self._calculate_equal_position_sizes(signals, equity, cash, current_positions, market_data)
        
        # Normalizar pesos
        weights = {symbol: inv_vol/total_inverse_vol for symbol, inv_vol in inverse_vol.items()}
        
        # Calcular capital disponible por posición según pesos
        risk_per_trade = equity * (self.portfolio_config["risk_per_trade_pct"] / 100)
        max_position_size = equity * (self.portfolio_config["max_position_size_pct"] / 100)
        
        for symbol, signal in signals.items():
            try:
                if symbol not in weights or symbol not in market_data:
                    continue
                
                # Obtener último precio
                last_price = market_data[symbol]["close"].iloc[-1]
                
                # Calcular stop loss (simplificado)
                atr = market_data[symbol]["high"].rolling(14).max() - market_data[symbol]["low"].rolling(14).min()
                atr = atr.iloc[-1] / 14  # ATR simplificado
                stop_distance = atr * 2  # Stop a 2 ATR
                
                # Calcular cantidad basada en riesgo
                position_equity = min(risk_per_trade / (stop_distance / last_price), 
                                    max_position_size * weights[symbol])
                qty = int(position_equity / last_price)
                
                # Ajustar dirección
                if signal["direction"] == "sell":
                    qty = -qty
                
                # Verificar cantidad mínima
                if abs(qty) < 1:
                    logger.warning(f"Cantidad calculada para {symbol} es menor que 1: {qty}")
                    continue
                
                # Crear orden
                orders[symbol] = {
                    "symbol": symbol,
                    "qty": qty,
                    "side": "buy" if qty > 0 else "sell",
                    "type": "market",
                    "time_in_force": "day",
                    "position_value": qty * last_price,
                    "signal_confidence": signal["confidence"],
                    "weight": weights[symbol],
                    "volatility": volatilities[symbol],
                }
                
            except Exception as e:
                logger.error(f"Error al calcular tamaño para {symbol}: {e}", exc_info=True)
        
        return orders

    def _calculate_kelly_position_sizes(self, signals: Dict[str, Dict[str, Any]],
                                      equity: float, cash: float,
                                      current_positions: Dict[str, float],
                                      market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Calcula tamaños de posición basados en el criterio de Kelly.

        Args:
            signals: Diccionario con señales por símbolo
            equity: Capital total de la cuenta
            cash: Efectivo disponible
            current_positions: Posiciones actuales
            market_data: Datos de mercado recientes por símbolo

        Returns:
            Dict[str, Dict[str, Any]]: Órdenes con tamaños calculados
        """
        orders = {}
        
        if not signals:
            return orders
        
        # Calcular fracciones de Kelly para cada símbolo
        kelly_fractions = {}
        for symbol, signal in signals.items():
            # Usar confianza de la señal como probabilidad de éxito
            win_prob = signal["confidence"]
            
            # Calcular ratio de ganancia/pérdida (simplificado)
            # Asumimos ratio de 1:1 por defecto
            win_loss_ratio = 1.0
            
            # Si hay metadatos con información de modelo, usar esa información
            if "metadata" in signal and "model_prediction" in signal["metadata"]:
                # Si es un modelo de regresión, usar la predicción como expectativa de retorno
                expected_return = signal["metadata"].get("model_prediction", 0)
                if expected_return > 0:
                    win_loss_ratio = expected_return / 0.01  # Asumimos pérdida de 1%
            
            # Calcular fracción de Kelly
            # f* = (p * b - q) / b donde p = prob de ganar, q = prob de perder, b = ratio ganancia/pérdida
            kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
            
            # Limitar Kelly a un máximo de 25% (Kelly fraccional)
            kelly = min(max(kelly, 0), 0.25)
            
            kelly_fractions[symbol] = kelly
        
        # Calcular capital disponible por posición según Kelly
        max_position_size = equity * (self.portfolio_config["max_position_size_pct"] / 100)
        
        for symbol, signal in signals.items():
            try:
                if symbol not in kelly_fractions or symbol not in market_data:
                    continue
                
                # Obtener último precio
                last_price = market_data[symbol]["close"].iloc[-1]
                
                # Calcular posición según Kelly
                position_equity = min(equity * kelly_fractions[symbol], max_position_size)
                qty = int(position_equity / last_price)
                
                # Ajustar dirección
                if signal["direction"] == "sell":
                    qty = -qty
                
                # Verificar cantidad mínima
                if abs(qty) < 1:
                    logger.warning(f"Cantidad calculada para {symbol} es menor que 1: {qty}")
                    continue
                
                # Crear orden
                orders[symbol] = {
                    "symbol": symbol,
                    "qty": qty,
                    "side": "buy" if qty > 0 else "sell",
                    "type": "market",
                    "time_in_force": "day",
                    "position_value": qty * last_price,
                    "signal_confidence": signal["confidence"],
                    "kelly_fraction": kelly_fractions[symbol],
                }
                
            except Exception as e:
                logger.error(f"Error al calcular tamaño para {symbol}: {e}", exc_info=True)
        
        return orders

    def rebalance_portfolio(self, account_info: Dict[str, Any],
                          current_positions: Dict[str, float],
                          market_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
        """Rebalancea la cartera para mantener pesos objetivo.

        Args:
            account_info: Información de la cuenta (equity, cash, etc.)
            current_positions: Posiciones actuales
            market_data: Datos de mercado recientes por símbolo

        Returns:
            Dict[str, Dict[str, Any]]: Órdenes de rebalanceo
        """
        # Esta función se puede implementar para rebalancear la cartera periódicamente
        # Por ahora, devolvemos un diccionario vacío
        return {}

    def calculate_portfolio_metrics(self, account_info: Dict[str, Any],
                                  current_positions: Dict[str, float],
                                  market_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Calcula métricas de la cartera.

        Args:
            account_info: Información de la cuenta (equity, cash, etc.)
            current_positions: Posiciones actuales
            market_data: Datos de mercado recientes por símbolo

        Returns:
            Dict[str, Any]: Métricas de la cartera
        """
        metrics = {
            "total_equity": account_info.get("equity", 0),
            "cash": account_info.get("cash", 0),
            "num_positions": len(current_positions),
            "long_exposure": 0,
            "short_exposure": 0,
            "net_exposure": 0,
            "gross_exposure": 0,
            "leverage": 0,
        }
        
        # Calcular exposición
        total_long = 0
        total_short = 0
        
        for symbol, position in current_positions.items():
            market_value = position.get("market_value", 0)
            if market_value > 0:
                total_long += market_value
            else:
                total_short += abs(market_value)
        
        metrics["long_exposure"] = total_long
        metrics["short_exposure"] = total_short
        metrics["net_exposure"] = total_long - total_short
        metrics["gross_exposure"] = total_long + total_short
        
        # Calcular apalancamiento
        if metrics["total_equity"] > 0:
            metrics["leverage"] = metrics["gross_exposure"] / metrics["total_equity"]
        
        return metrics

    def simulate_portfolio(self, signals, historical_data):
        """
        Simula la evolución del portafolio a partir de señales y datos históricos.
        """
        self.logger.info("Iniciando simulación de portafolio...")

        if not signals or len(signals) == 0:
            self.logger.warning("No hay señales disponibles para simular el portafolio.")
            return pd.DataFrame()

        portfolio_value = self.config.get("INITIAL_EQUITY", 100000)
        history = []
        equity_history = []

        for symbol, df in historical_data.items():
            if symbol not in signals:
                continue

            # signals es un dict con señales por símbolo, no una serie
            # Necesitamos procesar las señales correctamente
            signal_data = signals.get(symbol, {})
            if not signal_data:
                continue

            # Para simplificar, asumimos que las señales están en un formato que podemos usar
            # Por ahora, creamos una simulación básica
            prices = df["close"].values
            if len(prices) < 2:
                continue

            # Simulación simplificada: asumir posición constante
            returns = np.diff(prices) / prices[:-1]
            # Simular una posición básica (por ejemplo, comprar al inicio y mantener)
            position = 1  # posición larga
            pnl = np.sum(returns) * position * (portfolio_value / len(historical_data))
            portfolio_value += pnl

            # Crear historial de equity por día
            equity_values = [portfolio_value]
            for ret in returns:
                equity_values.append(equity_values[-1] * (1 + ret))

            # Añadir al historial general
            for i, equity in enumerate(equity_values):
                equity_history.append({
                    "date": df.index[i] if i < len(df.index) else df.index[-1],
                    "equity": equity,
                    "symbol": symbol
                })

            history.append({
                "symbol": symbol,
                "final_value": portfolio_value,
                "total_pnl": pnl
            })

        # Crear DataFrame con historial de equity
        equity_df = pd.DataFrame(equity_history)
        if not equity_df.empty:
            # Agrupar por fecha y sumar equity
            equity_df = equity_df.groupby('date')['equity'].sum().reset_index()

        result_df = pd.DataFrame(history)
        self.logger.info("Simulación de portafolio completada.")
        return equity_df

    def generate_orders(self, approved_signals: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Genera órdenes basadas en señales aprobadas.

        Args:
            approved_signals: Señales aprobadas por el gestor de riesgo

        Returns:
            Dict[str, Dict[str, Any]]: Órdenes a ejecutar
        """
        # Para el modo live/paper, necesitamos información de cuenta y posiciones actuales
        # Por ahora, devolver un diccionario vacío ya que el sistema no está ejecutando operaciones
        self.logger.info(f"Señales aprobadas recibidas: {len(approved_signals)}, pero no se generan órdenes porque el ML no está ejecutando operaciones.")
        return {}
