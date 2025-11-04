#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para la ingesta de datos históricos y en tiempo real de Alpaca.
"""

import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any

# Importaciones para Alpaca
# Para datos históricos
from alpaca.data import StockHistoricalDataClient
# Para streaming en tiempo real
try:
    from alpaca.data.live import StockDataStream
except ImportError:
    StockDataStream = None
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame

logger = logging.getLogger(__name__)


class DataIngestionManager:
    """Gestor de ingesta de datos para el sistema de trading."""

    def __init__(self, config: Dict[str, Any], alpaca_client=None):
        """Inicializa el gestor de ingesta de datos.

        Args:
            config: Configuración del sistema
            alpaca_client: Cliente de Alpaca (opcional)
        """
        self.config = config
        self.alpaca_client = alpaca_client
        self.data_dir = config.get("DATA_DIR")
        self.api_key = config.get("ALPACA_API_KEY")
        self.api_secret = config.get("ALPACA_API_SECRET")
        
        # Crear cliente de datos históricos
        self.historical_client = StockHistoricalDataClient(self.api_key, self.api_secret)
        
        # Inicializar stream de datos en tiempo real
        self.data_stream = None
        self.StockDataStream = None
        try:
            from alpaca.data.live import StockDataStream
            self.StockDataStream = StockDataStream
        except ImportError:
            logger.warning("No se pudo importar StockDataStream. El streaming no estará disponible.")

        self.stream_callbacks = {}
        
        logger.info("Gestor de ingesta de datos inicializado")

    def get_historical_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: str = "1D",
    ) -> Dict[str, pd.DataFrame]:
        """Obtiene datos históricos de Alpaca.

        Args:
            symbols: Lista de símbolos
            start_date: Fecha de inicio (formato: YYYY-MM-DD)
            end_date: Fecha de fin (formato: YYYY-MM-DD)
            timeframe: Intervalo de tiempo (1D, 1H, 15Min, etc.)

        Returns:
            Dict[str, pd.DataFrame]: Diccionario con DataFrames por símbolo
        """
        logger.info(f"Obteniendo datos históricos para {len(symbols)} símbolos")
        
        # Convertir timeframe a formato de Alpaca
        tf_mapping = {
            "1D": TimeFrame.Day,
            "1H": TimeFrame.Hour,
            "15Min": TimeFrame.Minute(15),
            "5Min": TimeFrame.Minute(5),
            "1Min": TimeFrame.Minute,
        }
        
        tf = tf_mapping.get(timeframe, TimeFrame.Day)
        
        # Crear request para barras
        bars_request = StockBarsRequest(
            symbol_or_symbols=symbols,
            start=start_date,
            end=end_date,
            timeframe=tf,
            adjustment="all",
        )
        
        try:
            # Obtener barras
            bars_data = self.historical_client.get_stock_bars(bars_request)
            
            # Convertir a diccionario de DataFrames
            result = {}
            for symbol in symbols:
                if symbol in bars_data:
                    df = bars_data[symbol].df
                    # Guardar en caché si está configurado
                    self._cache_data(symbol, df, f"historical_{timeframe}")
                    result[symbol] = df
                else:
                    logger.warning(f"No se encontraron datos para el símbolo {symbol}")
            
            logger.info(f"Datos históricos obtenidos correctamente para {len(result)} símbolos")
            return result
            
        except Exception as e:
            logger.error(f"Error al obtener datos históricos: {e}", exc_info=True)
            # Intentar cargar desde caché si está disponible
            return self._load_from_cache(symbols, f"historical_{timeframe}")

    def get_realtime_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Obtiene datos en tiempo real de Alpaca.

        Args:
            symbols: Lista de símbolos

        Returns:
            Dict[str, pd.DataFrame]: Diccionario con DataFrames por símbolo
        """
        logger.info(f"Obteniendo datos en tiempo real para {len(symbols)} símbolos")
        
        # Si estamos en modo backtest, usar datos históricos recientes
        if self.config.get("TRADING_MODE") == "backtest":
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
            return self.get_historical_data(symbols, start_date, end_date, "1Min")
        
        # Para paper y live trading, obtener datos en tiempo real
        result = {}
        
        try:
            # Obtener últimas barras (datos de 1 minuto más recientes)
            for symbol in symbols:
                bars_request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    start=(datetime.now() - timedelta(minutes=10)).isoformat(),
                    end=datetime.now().isoformat(),
                    timeframe=TimeFrame.Minute,
                    limit=10,
                )
                
                bars_data = self.historical_client.get_stock_bars(bars_request)
                if symbol in bars_data:
                    result[symbol] = bars_data[symbol].df
                else:
                    logger.warning(f"No se encontraron datos en tiempo real para {symbol}")
            
            logger.info(f"Datos en tiempo real obtenidos para {len(result)} símbolos")
            return result
            
        except Exception as e:
            logger.error(f"Error al obtener datos en tiempo real: {e}", exc_info=True)
            return {}

    def start_streaming(self, symbols: List[str], callback=None):
        """Inicia el streaming de datos en tiempo real.

        Args:
            symbols: Lista de símbolos
            callback: Función de callback para procesar datos
        """
        if self.config.get("TRADING_MODE") == "backtest":
            logger.info("Streaming no disponible en modo backtest")
            return
        
        try:
            # Inicializar stream si no existe
            if self.data_stream is None:
                if not self.StockDataStream:
                    logger.error("StockDataStream no está disponible. No se puede iniciar el streaming.")
                    return
                self.data_stream = self.StockDataStream(self.api_key, self.api_secret)
            
            # Registrar callback
            if callback:
                self.stream_callbacks["data"] = callback
            
            # Configurar handlers
            self.data_stream.subscribe_bars(self._process_bar_data, *symbols)
            self.data_stream.subscribe_quotes(self._process_quote_data, *symbols)
            self.data_stream.subscribe_trades(self._process_trade_data, *symbols)
            
            # Iniciar stream
            self.data_stream.run()
            logger.info(f"Streaming iniciado para {len(symbols)} símbolos")
            
        except Exception as e:
            logger.error(f"Error al iniciar streaming: {e}", exc_info=True)

    def stop_streaming(self):
        """Detiene el streaming de datos en tiempo real."""
        if self.data_stream:
            try:
                self.data_stream.stop()
                self.data_stream = None
                logger.info("Streaming detenido")
            except Exception as e:
                logger.error(f"Error al detener streaming: {e}", exc_info=True)

    def _process_bar_data(self, bar):
        """Procesa datos de barras recibidos por streaming."""
        if "data" in self.stream_callbacks:
            self.stream_callbacks["data"]("bar", bar)

    def _process_quote_data(self, quote):
        """Procesa datos de cotizaciones recibidos por streaming."""
        if "data" in self.stream_callbacks:
            self.stream_callbacks["data"]("quote", quote)

    def _process_trade_data(self, trade):
        """Procesa datos de operaciones recibidos por streaming."""
        if "data" in self.stream_callbacks:
            self.stream_callbacks["data"]("trade", trade)

    def _cache_data(self, symbol: str, data: pd.DataFrame, cache_type: str):
        """Guarda datos en caché."""
        if not self.data_dir:
            return
        
        try:
            cache_dir = os.path.join(self.data_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            
            file_path = os.path.join(cache_dir, f"{symbol}_{cache_type}.parquet")
            data.to_parquet(file_path)
            logger.debug(f"Datos guardados en caché: {file_path}")
            
        except Exception as e:
            logger.error(f"Error al guardar datos en caché: {e}")

    def _load_from_cache(self, symbols: List[str], cache_type: str) -> Dict[str, pd.DataFrame]:
        """Carga datos desde caché."""
        if not self.data_dir:
            return {}
        
        result = {}
        cache_dir = os.path.join(self.data_dir, "cache")
        
        for symbol in symbols:
            try:
                file_path = os.path.join(cache_dir, f"{symbol}_{cache_type}.parquet")
                if os.path.exists(file_path):
                    result[symbol] = pd.read_parquet(file_path)
                    logger.debug(f"Datos cargados desde caché: {file_path}")
            except Exception as e:
                logger.error(f"Error al cargar datos desde caché para {symbol}: {e}")
        
        return result