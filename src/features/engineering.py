#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para el procesamiento y generación de features para los modelos de ML.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any

# Importar librería de indicadores técnicos
import ta

# Importar módulos de noticias y sentimiento
from src.data.news_ingestion import NewsIngestionManager
from src.features.sentiment_analysis import SentimentAnalyzer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Clase para la generación y procesamiento de features."""

    def __init__(self, config: Dict[str, Any]):
        """Inicializa el ingeniero de features.

        Args:
            config: Configuración del sistema
        """
        self.config = config
        self.feature_config = {
            "window_sizes": [5, 10, 20, 50, 200],  # Ventanas para medias móviles, etc.
            "use_ta_lib": True,  # Usar librería TA para indicadores técnicos
            "normalize": True,  # Normalizar features
            "use_news_sentiment": config.get("USE_NEWS_SENTIMENT", True),  # Usar análisis de sentimiento
        }

        # Inicializar módulos de noticias y sentimiento
        self.news_manager = NewsIngestionManager(config) if self.feature_config["use_news_sentiment"] else None
        self.sentiment_analyzer = SentimentAnalyzer(config) if self.feature_config["use_news_sentiment"] else None

        logger.info("Feature Engineer inicializado")

    def process_data(self, market_data: Dict[str, pd.DataFrame], drop_na: bool = True) -> Dict[str, pd.DataFrame]:
        """Procesa datos de mercado y genera features.

        Args:
            market_data: Diccionario con DataFrames por símbolo
            drop_na: Si eliminar filas con NaN (para training) o rellenar (para prediction)

        Returns:
            Dict[str, pd.DataFrame]: Diccionario con DataFrames de features por símbolo
        """
        result = {}

        for symbol, df in market_data.items():
            try:
                # Verificar que el DataFrame no esté vacío
                if df.empty:
                    logger.warning(f"DataFrame vacío para {symbol}, omitiendo")
                    continue

                # Procesar datos y generar features
                features_df = self._generate_features(df)

                # Agregar features de sentimiento si está habilitado
                if self.feature_config["use_news_sentiment"] and self.news_manager and self.sentiment_analyzer:
                    try:
                        features_df = self._add_sentiment_features(features_df, symbol)
                    except Exception as e:
                        logger.warning(f"Error agregando features de sentimiento para {symbol}: {e}")

                # Crear variable objetivo
                features_df = self.create_target_variable(features_df)

                # Normalizar si está configurado
                if self.feature_config["normalize"]:
                    features_df = self._normalize_features(features_df)

                # Asegurar que target_direction sea int después de normalización
                if 'target_direction' in features_df.columns:
                    features_df['target_direction'] = features_df['target_direction'].astype(int)

                if drop_na:
                    # Eliminar filas con NaN (para training) - pero mantener algunas filas para backtest
                    initial_len = len(features_df)
                    features_df = features_df.dropna()
                    logger.debug(f"Eliminadas {initial_len - len(features_df)} filas con NaN para {symbol}")
                else:
                    # Rellenar NaN con 0 para prediction (última fila disponible)
                    features_df = features_df.fillna(0)

                result[symbol] = features_df
                logger.debug(f"Features generadas para {symbol}: {features_df.shape[1]} características")

            except Exception as e:
                logger.error(f"Error al generar features para {symbol}: {e}", exc_info=True)

        return result

    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Genera features a partir de datos de mercado.

        Args:
            df: DataFrame con datos de mercado (OHLCV)

        Returns:
            pd.DataFrame: DataFrame con features
        """
        # Crear copia para no modificar el original
        result = df.copy()
        
        # Asegurarse de que tenemos las columnas necesarias
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in result.columns]
        
        if missing_cols:
            logger.warning(f"Faltan columnas requeridas: {missing_cols}")
            # Intentar mapear nombres de columnas si es necesario
            for col in missing_cols:
                if col.upper() in result.columns:
                    result[col] = result[col.upper()]
        
        # 1. Features básicas de retornos
        result['returns'] = result['close'].pct_change()
        result['log_returns'] = np.log(result['close'] / result['close'].shift(1))
        
        # 2. Volatilidad
        for window in self.feature_config["window_sizes"]:
            # Usar ventana adaptativa si hay pocos datos
            effective_window = min(window, len(result) - 1)
            if effective_window >= 5:  # Mínimo 5 períodos para volatilidad
                result[f'volatility_{window}'] = result['returns'].rolling(window=effective_window).std()
            else:
                result[f'volatility_{window}'] = np.nan
        
        # 3. Medias móviles y cruces
        for window in self.feature_config["window_sizes"]:
            # Usar ventana adaptativa si hay pocos datos
            effective_window = min(window, len(result) - 1)
            if effective_window >= 2:  # Mínimo 2 períodos para medias móviles
                result[f'sma_{window}'] = result['close'].rolling(window=effective_window).mean()
                result[f'ema_{window}'] = result['close'].ewm(span=effective_window, adjust=False).mean()
            else:
                result[f'sma_{window}'] = np.nan
                result[f'ema_{window}'] = np.nan
        
        # Señales de cruce de medias móviles
        result['sma_cross_5_20'] = np.where(result['sma_5'] > result['sma_20'], 1, -1)
        result['sma_cross_10_50'] = np.where(result['sma_10'] > result['sma_50'], 1, -1)
        
        # 4. Momentum
        for window in self.feature_config["window_sizes"]:
            # Usar ventana adaptativa si hay pocos datos
            effective_window = min(window, len(result) - 1)
            if effective_window >= 1:  # Mínimo 1 período para momentum
                result[f'momentum_{window}'] = result['close'] / result['close'].shift(effective_window) - 1
            else:
                result[f'momentum_{window}'] = np.nan
        
        # 5. Indicadores técnicos usando TA-Lib
        if self.feature_config["use_ta_lib"]:
            # RSI
            result['rsi_14'] = ta.momentum.RSIIndicator(result['close'], window=14).rsi()
            
            # MACD
            macd = ta.trend.MACD(result['close'])
            result['macd'] = macd.macd()
            result['macd_signal'] = macd.macd_signal()
            result['macd_diff'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(result['close'])
            result['bb_high'] = bollinger.bollinger_hband()
            result['bb_low'] = bollinger.bollinger_lband()
            result['bb_width'] = (result['bb_high'] - result['bb_low']) / result['close']
            
            # ATR - Average True Range (solo si hay suficientes datos)
            if len(result) >= 14:  # ATR necesita al menos 14 períodos
                result['atr'] = ta.volatility.AverageTrueRange(
                    result['high'], result['low'], result['close']
                ).average_true_range()
            else:
                result['atr'] = np.nan
            
            # Volumen
            result['volume_sma_20'] = result['volume'].rolling(window=20).mean()
            result['volume_ratio'] = result['volume'] / result['volume_sma_20']
        
        # 6. Price patterns
        result['higher_high'] = (result['high'] > result['high'].shift(1)) & \
                              (result['high'].shift(1) > result['high'].shift(2))
        result['lower_low'] = (result['low'] < result['low'].shift(1)) & \
                            (result['low'].shift(1) < result['low'].shift(2))
        
        # 7. Gaps
        result['gap_up'] = result['open'] > result['close'].shift(1)
        result['gap_down'] = result['open'] < result['close'].shift(1)
        
        # 8. Day of week (para patrones semanales)
        if isinstance(result.index, pd.DatetimeIndex):
            result['day_of_week'] = result.index.dayofweek
        else:
            result['day_of_week'] = 0  # Valor por defecto si no hay índice datetime

        # 10. Features adicionales para mejor predicción
        # Ratio precio/volumen
        result['price_volume_ratio'] = result['close'] / (result['volume'] + 1)  # +1 para evitar división por cero

        # Momentum relativo
        result['relative_strength'] = result['close'] / result['close'].shift(20) - 1

        # Volumen relativo
        result['volume_relative'] = result['volume'] / result['volume'].shift(1)

        # Tendencia de precio (slope)
        result['price_trend_5'] = (result['close'] - result['close'].shift(5)) / result['close'].shift(5)
        result['price_trend_10'] = (result['close'] - result['close'].shift(10)) / result['close'].shift(10)
        result['price_trend_20'] = (result['close'] - result['close'].shift(20)) / result['close'].shift(20)
        
        # 9. Distancia a máximos/mínimos
        for window in [20, 50, 200]:
            # Usar ventana adaptativa si hay pocos datos
            effective_window = min(window, len(result) - 1)
            if effective_window >= 2:  # Mínimo 2 períodos para rolling max/min
                result[f'dist_to_high_{window}'] = result['close'] / result['high'].rolling(effective_window).max() - 1
                result[f'dist_to_low_{window}'] = result['close'] / result['low'].rolling(effective_window).min() - 1
            else:
                result[f'dist_to_high_{window}'] = np.nan
                result[f'dist_to_low_{window}'] = np.nan
        
        return result

    def _add_sentiment_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Agrega features de sentimiento a los datos de mercado.

        Args:
            df: DataFrame con datos de mercado
            symbol: Símbolo del instrumento

        Returns:
            pd.DataFrame: DataFrame con features de sentimiento agregadas
        """
        try:
            # Obtener noticias para el símbolo
            news_df = self.news_manager.get_news_for_symbol(symbol, days_back=7)

            if news_df.empty:
                logger.debug(f"No hay noticias disponibles para {symbol}")
                # Agregar columnas vacías de sentimiento
                df['news_count_24h'] = 0.0
                df['sentiment_mean_24h'] = 0.0
                df['sentiment_std_24h'] = 0.0
                df['sentiment_trend_24h'] = 0.0
                df['pos_sentiment_ratio_24h'] = 0.0
                df['neg_sentiment_ratio_24h'] = 0.0
                return df

            # Crear features de sentimiento
            df_with_sentiment = self.sentiment_analyzer.create_sentiment_features(news_df, df)

            logger.debug(f"Features de sentimiento agregadas para {symbol}: {len(news_df)} noticias procesadas")
            return df_with_sentiment

        except Exception as e:
            logger.error(f"Error agregando features de sentimiento para {symbol}: {e}", exc_info=True)
            # En caso de error, devolver DataFrame original con columnas vacías
            df['news_count_24h'] = 0.0
            df['sentiment_mean_24h'] = 0.0
            df['sentiment_std_24h'] = 0.0
            df['sentiment_trend_24h'] = 0.0
            df['pos_sentiment_ratio_24h'] = 0.0
            df['neg_sentiment_ratio_24h'] = 0.0
            return df

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza las features.

        Args:
            df: DataFrame con features

        Returns:
            pd.DataFrame: DataFrame con features normalizadas
        """
        result = df.copy()
        
        # Columnas a excluir de la normalización
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'day_of_week', 
                       'gap_up', 'gap_down', 'higher_high', 'lower_low']
        
        # Normalizar solo columnas numéricas que no estén en la lista de exclusión
        for col in result.columns:
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(result[col]):
                # Z-score normalization
                mean = result[col].mean()
                std = result[col].std()
                if std > 0:  # Evitar división por cero
                    result[col] = (result[col] - mean) / std
        
        return result

    def create_lagged_features(self, df: pd.DataFrame, lag_periods: List[int]) -> pd.DataFrame:
        """Crea features con retardos temporales.

        Args:
            df: DataFrame con features
            lag_periods: Lista de períodos de retardo

        Returns:
            pd.DataFrame: DataFrame con features retardadas
        """
        result = df.copy()
        
        # Columnas a excluir
        exclude_cols = ['open', 'high', 'low', 'close', 'volume']
        
        # Crear versiones retardadas de las features seleccionadas
        for col in result.columns:
            if col not in exclude_cols:
                for lag in lag_periods:
                    result[f'{col}_lag_{lag}'] = result[col].shift(lag)
        
        return result

    def prepare_prediction_data(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepara datos para predicción generando features sin variables objetivo.

        Args:
            market_data: Diccionario con DataFrames por símbolo

        Returns:
            Dict[str, pd.DataFrame]: Diccionario con DataFrames de features por símbolo
        """
        result = {}

        for symbol, df in market_data.items():
            try:
                # Verificar que el DataFrame no esté vacío
                if df.empty:
                    logger.warning(f"DataFrame vacío para {symbol}, omitiendo")
                    continue

                # Procesar datos y generar features
                features_df = self._generate_features(df)

                # Agregar features de sentimiento si está habilitado
                if self.feature_config["use_news_sentiment"] and self.news_manager and self.sentiment_analyzer:
                    try:
                        features_df = self._add_sentiment_features(features_df, symbol)
                    except Exception as e:
                        logger.warning(f"Error agregando features de sentimiento para {symbol}: {e}")

                # Normalizar si está configurado
                if self.feature_config["normalize"]:
                    features_df = self._normalize_features(features_df)

                # Rellenar NaN con 0 para prediction (última fila disponible)
                features_df = features_df.fillna(0)

                result[symbol] = features_df
                logger.debug(f"Features generadas para {symbol}: {features_df.shape[1]} características")

            except Exception as e:
                logger.error(f"Error al generar features para {symbol}: {e}", exc_info=True)

        return result

    def create_target_variable(self, df: pd.DataFrame, horizon: int = 1, threshold: float = 0.0) -> pd.DataFrame:
        """Crea la variable objetivo para entrenamiento supervisado.

        Args:
            df: DataFrame con datos
            horizon: Horizonte de predicción (períodos)
            threshold: Umbral para clasificación binaria

        Returns:
            pd.DataFrame: DataFrame con variable objetivo
        """
        result = df.copy()

        # Retorno futuro (variable objetivo para regresión)
        result['future_return'] = result['close'].pct_change(horizon).shift(-horizon)

        # Dirección del movimiento (variable objetivo para clasificación)
        result['target_direction'] = np.where(result['future_return'] > threshold, 1, -1).astype(int)

        # Retorno futuro categorizado (para clasificación multi-clase)
        bins = [-np.inf, -0.01, 0.01, np.inf]
        labels = [0, 1, 2]  # 0: bajada, 1: lateral, 2: subida
        result['target_class'] = pd.cut(result['future_return'], bins=bins, labels=labels)

        return result
