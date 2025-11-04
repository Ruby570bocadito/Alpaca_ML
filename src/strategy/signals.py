#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para la generación de señales de trading basadas en predicciones de modelos.
"""

import logging
import pandas as pd
import numpy as np
import random
from typing import Dict, List, Optional, Union, Any, Tuple

logger = logging.getLogger(__name__)


class SignalGenerator:
    """Clase para generar señales de trading basadas en predicciones de modelos."""

    def __init__(self, config: Dict[str, Any]):
        """Inicializa el generador de señales.

        Args:
            config: Configuración del sistema
        """
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Configuración de señales
        self.signal_config = {
            "confidence_threshold": float(config.get("SIGNAL_CONFIDENCE_THRESHOLD", "0.6")),
            "min_probability": float(config.get("SIGNAL_MIN_PROBABILITY", "0.55")),
            "max_positions": int(config.get("MAX_POSITIONS", "5")),
            "signal_expiry_bars": int(config.get("SIGNAL_EXPIRY_BARS", "3")),
            "strategy_type": config.get("STRATEGY_TYPE", "ml_prediction"),
        }

        self.logger.info("Generador de señales inicializado")

    def generate_signals(self, predictions: Dict[str, pd.DataFrame], 
                         current_positions: Dict[str, float] = None) -> Dict[str, Dict[str, Any]]:
        """Genera señales de trading basadas en predicciones.

        Args:
            predictions: Diccionario con DataFrames de predicciones por símbolo
            current_positions: Diccionario con posiciones actuales por símbolo

        Returns:
            Dict[str, Dict[str, Any]]: Señales generadas por símbolo
        """
        if current_positions is None:
            current_positions = {}
        
        signals = {}
        
        # Seleccionar estrategia según configuración
        if self.signal_config["strategy_type"] == "ml_prediction":
            signals = self._generate_ml_signals(predictions, current_positions)
        elif self.signal_config["strategy_type"] == "mean_reversion":
            signals = self._generate_mean_reversion_signals(predictions, current_positions)
        elif self.signal_config["strategy_type"] == "trend_following":
            signals = self._generate_trend_following_signals(predictions, current_positions)
        elif self.signal_config["strategy_type"] == "ensemble":
            signals = self._generate_ensemble_signals(predictions, current_positions)
        else:
            logger.warning(f"Tipo de estrategia desconocido: {self.signal_config['strategy_type']}")
            signals = self._generate_ml_signals(predictions, current_positions)  # Fallback a ML
        
        logger.info(f"Generadas {len(signals)} señales")
        return signals

    def _generate_ml_signals(self, predictions: Dict[str, pd.DataFrame], 
                            current_positions: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Genera señales basadas en predicciones de modelos ML.

        Args:
            predictions: Diccionario con DataFrames de predicciones por símbolo
            current_positions: Diccionario con posiciones actuales por símbolo

        Returns:
            Dict[str, Dict[str, Any]]: Señales generadas por símbolo
        """
        signals = {}
        
        for symbol, pred_df in predictions.items():
            try:
                # Verificar que tenemos las columnas necesarias
                if "prediction" not in pred_df.columns:
                    logger.warning(f"No hay columna 'prediction' para {symbol}")
                    continue
                
                # Obtener la última predicción
                last_pred = pred_df.iloc[-1]
                
                # Determinar dirección de la señal
                signal_direction = None
                confidence = 0.5  # Confianza por defecto
                
                # Si tenemos probabilidades, usar umbral de confianza
                if "probability" in pred_df.columns:
                    probability = last_pred["probability"]
                    confidence = probability

                    # Señal de compra si probabilidad > 0.6
                    if probability > 0.6:
                        signal_direction = "buy"
                        confidence = probability

                    # Señal de venta si probabilidad < 0.4
                    elif probability < 0.4:
                        signal_direction = "sell"
                        confidence = 1 - probability
                
                # Si no tenemos probabilidades, usar directamente la predicción
                else:
                    if last_pred["prediction"] == 1 or last_pred["prediction"] > 0:
                        signal_direction = "buy"
                        confidence = 0.6  # Confianza arbitraria
                    elif last_pred["prediction"] == 0 or last_pred["prediction"] < 0:
                        signal_direction = "sell"
                        confidence = 0.6  # Confianza arbitraria
                
                # Considerar posición actual
                current_position = current_positions.get(symbol, 0)
                
                # No generar señal de compra si ya tenemos posición larga
                if signal_direction == "buy" and current_position > 0:
                    continue
                
                # No generar señal de venta si ya tenemos posición corta o no tenemos posición
                if signal_direction == "sell" and current_position <= 0:
                    continue
                
                # Crear señal si tenemos dirección
                if signal_direction:
                    signals[symbol] = {
                        "direction": signal_direction,
                        "confidence": confidence,
                        "timestamp": pred_df.index[-1],
                        "expiry": self.signal_config["signal_expiry_bars"],
                        "metadata": {
                            "model_prediction": float(last_pred["prediction"]),
                            "probability": float(confidence),
                        }
                    }
                
            except Exception as e:
                logger.error(f"Error al generar señal para {symbol}: {e}", exc_info=True)
        
        return signals

    def _generate_mean_reversion_signals(self, predictions: Dict[str, pd.DataFrame], 
                                        current_positions: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Genera señales basadas en estrategia de reversión a la media.

        Args:
            predictions: Diccionario con DataFrames de predicciones por símbolo
            current_positions: Diccionario con posiciones actuales por símbolo

        Returns:
            Dict[str, Dict[str, Any]]: Señales generadas por símbolo
        """
        signals = {}
        
        for symbol, pred_df in predictions.items():
            try:
                # Verificar que tenemos suficientes datos
                if len(pred_df) < 20:  # Necesitamos al menos 20 barras para calcular medias
                    continue
                
                # Calcular medias móviles
                pred_df["sma_short"] = pred_df["close"].rolling(window=5).mean()
                pred_df["sma_long"] = pred_df["close"].rolling(window=20).mean()
                
                # Calcular bandas de Bollinger
                pred_df["sma"] = pred_df["close"].rolling(window=20).mean()
                pred_df["std"] = pred_df["close"].rolling(window=20).std()
                pred_df["upper_band"] = pred_df["sma"] + 2 * pred_df["std"]
                pred_df["lower_band"] = pred_df["sma"] - 2 * pred_df["std"]
                
                # Obtener último registro
                last_row = pred_df.iloc[-1]
                
                # Determinar dirección de la señal
                signal_direction = None
                confidence = 0.5
                
                # Señal de compra: precio cerca de banda inferior
                if last_row["close"] <= last_row["lower_band"]:
                    signal_direction = "buy"
                    # Calcular confianza basada en distancia a la banda
                    distance = (last_row["lower_band"] - last_row["close"]) / last_row["std"]
                    confidence = min(0.5 + distance * 0.25, 0.9)  # Máximo 0.9
                
                # Señal de venta: precio cerca de banda superior
                elif last_row["close"] >= last_row["upper_band"]:
                    signal_direction = "sell"
                    # Calcular confianza basada en distancia a la banda
                    distance = (last_row["close"] - last_row["upper_band"]) / last_row["std"]
                    confidence = min(0.5 + distance * 0.25, 0.9)  # Máximo 0.9
                
                # Considerar posición actual
                current_position = current_positions.get(symbol, 0)
                
                # No generar señal de compra si ya tenemos posición larga
                if signal_direction == "buy" and current_position > 0:
                    continue
                
                # No generar señal de venta si ya tenemos posición corta o no tenemos posición
                if signal_direction == "sell" and current_position <= 0:
                    continue
                
                # Crear señal si tenemos dirección y confianza suficiente
                if signal_direction and confidence >= self.signal_config["min_probability"]:
                    signals[symbol] = {
                        "direction": signal_direction,
                        "confidence": confidence,
                        "timestamp": pred_df.index[-1],
                        "expiry": self.signal_config["signal_expiry_bars"],
                        "metadata": {
                            "strategy": "mean_reversion",
                            "close": float(last_row["close"]),
                            "upper_band": float(last_row["upper_band"]),
                            "lower_band": float(last_row["lower_band"]),
                        }
                    }
                
            except Exception as e:
                logger.error(f"Error al generar señal para {symbol}: {e}", exc_info=True)
        
        return signals

    def _generate_trend_following_signals(self, predictions: Dict[str, pd.DataFrame], 
                                         current_positions: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Genera señales basadas en estrategia de seguimiento de tendencia.

        Args:
            predictions: Diccionario con DataFrames de predicciones por símbolo
            current_positions: Diccionario con posiciones actuales por símbolo

        Returns:
            Dict[str, Dict[str, Any]]: Señales generadas por símbolo
        """
        signals = {}
        
        for symbol, pred_df in predictions.items():
            try:
                # Verificar que tenemos suficientes datos
                if len(pred_df) < 50:  # Necesitamos al menos 50 barras
                    continue
                
                # Calcular medias móviles
                pred_df["sma_short"] = pred_df["close"].rolling(window=20).mean()
                pred_df["sma_long"] = pred_df["close"].rolling(window=50).mean()
                
                # Calcular ADX para medir fuerza de la tendencia
                # Simplificado: normalmente usaríamos una librería como ta
                pred_df["tr"] = np.maximum(
                    pred_df["high"] - pred_df["low"],
                    np.maximum(
                        np.abs(pred_df["high"] - pred_df["close"].shift(1)),
                        np.abs(pred_df["low"] - pred_df["close"].shift(1))
                    )
                )
                pred_df["atr"] = pred_df["tr"].rolling(window=14).mean()
                
                # Obtener últimos dos registros para detectar cruces
                last_row = pred_df.iloc[-1]
                prev_row = pred_df.iloc[-2] if len(pred_df) > 1 else None
                
                # Determinar dirección de la señal
                signal_direction = None
                confidence = 0.5
                
                # Detectar cruce de medias móviles
                if prev_row is not None:
                    # Cruce alcista: corto cruza largo hacia arriba
                    if prev_row["sma_short"] <= prev_row["sma_long"] and \
                       last_row["sma_short"] > last_row["sma_long"]:
                        signal_direction = "buy"
                        # Confianza basada en pendiente
                        slope = (last_row["sma_short"] - prev_row["sma_short"]) / prev_row["sma_short"]
                        confidence = min(0.5 + slope * 10, 0.9)  # Máximo 0.9
                    
                    # Cruce bajista: corto cruza largo hacia abajo
                    elif prev_row["sma_short"] >= prev_row["sma_long"] and \
                         last_row["sma_short"] < last_row["sma_long"]:
                        signal_direction = "sell"
                        # Confianza basada en pendiente
                        slope = (prev_row["sma_short"] - last_row["sma_short"]) / prev_row["sma_short"]
                        confidence = min(0.5 + slope * 10, 0.9)  # Máximo 0.9
                
                # Considerar posición actual
                current_position = current_positions.get(symbol, 0)
                
                # No generar señal de compra si ya tenemos posición larga
                if signal_direction == "buy" and current_position > 0:
                    continue
                
                # No generar señal de venta si ya tenemos posición corta o no tenemos posición
                if signal_direction == "sell" and current_position <= 0:
                    continue
                
                # Crear señal si tenemos dirección y confianza suficiente
                if signal_direction and confidence >= self.signal_config["min_probability"]:
                    signals[symbol] = {
                        "direction": signal_direction,
                        "confidence": confidence,
                        "timestamp": pred_df.index[-1],
                        "expiry": self.signal_config["signal_expiry_bars"],
                        "metadata": {
                            "strategy": "trend_following",
                            "sma_short": float(last_row["sma_short"]),
                            "sma_long": float(last_row["sma_long"]),
                            "slope": float((last_row["sma_short"] - prev_row["sma_short"]) / prev_row["sma_short"]) \
                                    if prev_row is not None else 0.0,
                        }
                    }
                
            except Exception as e:
                logger.error(f"Error al generar señal para {symbol}: {e}", exc_info=True)
        
        return signals

    def _generate_ensemble_signals(self, predictions: Dict[str, pd.DataFrame], 
                                  current_positions: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
        """Genera señales basadas en un conjunto de estrategias.

        Args:
            predictions: Diccionario con DataFrames de predicciones por símbolo
            current_positions: Diccionario con posiciones actuales por símbolo

        Returns:
            Dict[str, Dict[str, Any]]: Señales generadas por símbolo
        """
        # Generar señales de cada estrategia
        ml_signals = self._generate_ml_signals(predictions, current_positions)
        mr_signals = self._generate_mean_reversion_signals(predictions, current_positions)
        tf_signals = self._generate_trend_following_signals(predictions, current_positions)
        
        # Combinar señales
        ensemble_signals = {}
        all_symbols = set(list(ml_signals.keys()) + list(mr_signals.keys()) + list(tf_signals.keys()))
        
        for symbol in all_symbols:
            # Contar señales por dirección
            buy_signals = 0
            buy_confidence = 0.0
            sell_signals = 0
            sell_confidence = 0.0
            
            # Verificar señales de ML
            if symbol in ml_signals:
                if ml_signals[symbol]["direction"] == "buy":
                    buy_signals += 1
                    buy_confidence += ml_signals[symbol]["confidence"]
                else:
                    sell_signals += 1
                    sell_confidence += ml_signals[symbol]["confidence"]
            
            # Verificar señales de Mean Reversion
            if symbol in mr_signals:
                if mr_signals[symbol]["direction"] == "buy":
                    buy_signals += 1
                    buy_confidence += mr_signals[symbol]["confidence"]
                else:
                    sell_signals += 1
                    sell_confidence += mr_signals[symbol]["confidence"]
            
            # Verificar señales de Trend Following
            if symbol in tf_signals:
                if tf_signals[symbol]["direction"] == "buy":
                    buy_signals += 1
                    buy_confidence += tf_signals[symbol]["confidence"]
                else:
                    sell_signals += 1
                    sell_confidence += tf_signals[symbol]["confidence"]
            
            # Determinar dirección final
            signal_direction = None
            confidence = 0.5
            
            # Si hay más señales de compra que de venta
            if buy_signals > sell_signals:
                signal_direction = "buy"
                confidence = buy_confidence / buy_signals if buy_signals > 0 else 0.5
            
            # Si hay más señales de venta que de compra
            elif sell_signals > buy_signals:
                signal_direction = "sell"
                confidence = sell_confidence / sell_signals if sell_signals > 0 else 0.5
            
            # Considerar posición actual
            current_position = current_positions.get(symbol, 0)
            
            # No generar señal de compra si ya tenemos posición larga
            if signal_direction == "buy" and current_position > 0:
                continue
            
            # No generar señal de venta si ya tenemos posición corta o no tenemos posición
            if signal_direction == "sell" and current_position <= 0:
                continue
            
            # Crear señal si tenemos dirección y confianza suficiente
            if signal_direction and confidence >= self.signal_config["min_probability"]:
                ensemble_signals[symbol] = {
                    "direction": signal_direction,
                    "confidence": confidence,
                    "timestamp": predictions[symbol].index[-1] if symbol in predictions else None,
                    "expiry": self.signal_config["signal_expiry_bars"],
                    "metadata": {
                        "strategy": "ensemble",
                        "buy_signals": buy_signals,
                        "sell_signals": sell_signals,
                        "ml_signal": symbol in ml_signals,
                        "mr_signal": symbol in mr_signals,
                        "tf_signal": symbol in tf_signals,
                    }
                }
        
        return ensemble_signals

    def generate_simulated_signals(self, historical_data):
        """
        Genera señales simuladas para testing y backtesting.
        """
        self.logger.info("Generando señales simuladas...")

        signals = []
        for symbol in self.config.get("symbols", ["AAPL", "MSFT", "GOOGL"]):
            # Simulamos datos de 252 días (1 año bursátil)
            dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="B")
            for date in dates:
                # Aleatoriamente: 10% probabilidad de compra, 10% de venta, 80% nada
                r = random.random()
                if r < 0.1:
                    signal = {"symbol": symbol, "action": "BUY", "date": date}
                elif r < 0.2:
                    signal = {"symbol": symbol, "action": "SELL", "date": date}
                else:
                    continue
                signals.append(signal)

        df = pd.DataFrame(signals)
        self.logger.info(f"Generadas {len(df)} señales simuladas")
        return df
