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
from alpaca.data import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame

# Importar módulo de noticias
from src.data.news_ingestion import NewsIngestionManager

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

        # Crear cliente de datos históricos solo si hay cliente Alpaca
        if alpaca_client is not None:
            self.historical_client = StockHistoricalDataClient(
                self.api_key,
                self.api_secret
            )
        else:
            self.historical_client = None

        # Inicializar stream de datos en tiempo real
        self.data_stream = None
        self.stream_callbacks = {}

        # Inicializar gestor de noticias
        self.news_manager = NewsIngestionManager(config)

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

        # Si no hay cliente histórico, generar datos simulados
        if self.historical_client is None:
            logger.info("Usando datos simulados para modo sin API")
            return self._generate_simulated_data(symbols, start_date, end_date, timeframe)

        # Mapear string de timeframe a TimeFrame de Alpaca
        timeframe_map = {
            "1Min": TimeFrame.Minute,
            "5Min": TimeFrame(5, TimeFrame.Minute),
            "15Min": TimeFrame(15, TimeFrame.Minute),
            "1H": TimeFrame.Hour,
            "1D": TimeFrame.Day,
        }

        tf = timeframe_map.get(timeframe, TimeFrame.Day)

        # Crear request para barras
        bars_request = StockBarsRequest(
            symbol_or_symbols=symbols,
            start=start_date,
            end=end_date,
            timeframe=tf,
            adjustment="all",
            feed=DataFeed.SIP
        )

        try:
            # Obtener barras
            bars_data = self.historical_client.get_stock_bars(bars_request)

            # Convertir a diccionario de DataFrames
            result = {}
            for symbol in symbols:
                if symbol in bars_data.data:
                    # Convertir la lista de barras a DataFrame
                    bars_list = bars_data.data[symbol]
                    if bars_list:
                        df = pd.DataFrame([bar.model_dump() for bar in bars_list])
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index(['timestamp'], inplace=True)
                        df = df[['open', 'high', 'low', 'close', 'volume']]
                        # Guardar en caché si está configurado
                        self._cache_data(symbol, df, f"historical_{timeframe}")
                        result[symbol] = df
                    else:
                        logger.warning(f"No se encontraron datos para el símbolo {symbol}")
                else:
                    logger.warning(f"No se encontraron datos para el símbolo {symbol}")

            logger.info(f"Datos históricos obtenidos correctamente para {len(result)} símbolos")
            return result

        except Exception as e:
            logger.error(f"Error al obtener datos históricos: {e}", exc_info=True)
            # Intentar cargar desde caché si está disponible
            return self._load_from_cache(symbols, f"historical_{timeframe}")

    def get_historical_data_with_news(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: str = "1D",
    ) -> Dict[str, pd.DataFrame]:
        """Obtiene datos históricos incluyendo noticias históricas.

        Args:
            symbols: Lista de símbolos
            start_date: Fecha de inicio (formato: YYYY-MM-DD)
            end_date: Fecha de fin (formato: YYYY-MM-DD)
            timeframe: Intervalo de tiempo (1D, 1H, 15Min, etc.)

        Returns:
            Dict[str, pd.DataFrame]: Diccionario con DataFrames por símbolo
        """
        # Obtener datos históricos de mercado
        market_data = self.get_historical_data(symbols, start_date, end_date, timeframe)

        # Calcular días de back para noticias (basado en el rango de fechas)
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        days_back = (end_dt - start_dt).days + 7  # +7 para margen

        # Obtener noticias históricas
        try:
            news_data = self.news_manager.get_market_news(symbols, days_back=days_back)

            # Loggear resultados de noticias
            if news_data:
                for symbol, news_df in news_data.items():
                    if not news_df.empty:
                        logger.info(f"Obtenidas {len(news_df)} noticias históricas para {symbol}")
                    else:
                        logger.debug(f"No hay noticias históricas para {symbol}")

        except Exception as e:
            logger.warning(f"Error obteniendo noticias históricas: {e}")

        return market_data

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

        # Si no hay cliente histórico, generar datos simulados recientes
        if self.historical_client is None:
            logger.info("Usando datos simulados recientes para modo sin API")
            end_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            start_date = (datetime.now() - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
            return self._generate_simulated_data(symbols, start_date, end_date, "1Min")

        # Para paper y live trading, obtener datos en tiempo real
        result = {}

        try:
            # Obtener últimas barras (datos de 1 minuto más recientes)
            bars_request = StockBarsRequest(
                symbol_or_symbols=symbols,
                start=(datetime.now() - timedelta(minutes=10)).isoformat(),
                end=datetime.now().isoformat(),
                timeframe=TimeFrame.Minute,
                limit=10,
                feed=DataFeed.IEX
            )

            bars_data = self.historical_client.get_stock_bars(bars_request)

            # Procesar datos por símbolo
            for symbol in symbols:
                if symbol in bars_data.data and bars_data.data[symbol]:
                    # Convertir la lista de barras a DataFrame
                    bars_list = bars_data.data[symbol]
                    df = pd.DataFrame([bar.model_dump() for bar in bars_list])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df.set_index(['timestamp'], inplace=True)
                    df = df[['open', 'high', 'low', 'close', 'volume']]
                    result[symbol] = df
                else:
                    logger.warning(f"No se encontraron datos en tiempo real para {symbol}")

            logger.info(f"Datos en tiempo real obtenidos para {len(result)} símbolos")
            return result

        except Exception as e:
            logger.error(f"Error al obtener datos en tiempo real: {e}", exc_info=True)
            return {}

    def get_realtime_data_with_news(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Obtiene datos en tiempo real incluyendo noticias.

        Args:
            symbols: Lista de símbolos

        Returns:
            Dict[str, pd.DataFrame]: Diccionario con DataFrames por símbolo
        """
        # Obtener datos de mercado en tiempo real
        market_data = self.get_realtime_data(symbols)

        # Integrar noticias si están disponibles
        if market_data:
            try:
                # Obtener noticias para los símbolos
                news_data = self.news_manager.get_market_news(symbols, days_back=1)

                # Aquí se podría integrar las noticias con los datos de mercado
                # Por ahora, solo loggeamos que se obtuvieron noticias
                if news_data:
                    for symbol, news_df in news_data.items():
                        if not news_df.empty:
                            logger.info(f"Obtenidas {len(news_df)} noticias recientes para {symbol}")
                        else:
                            logger.debug(f"No hay noticias recientes para {symbol}")

            except Exception as e:
                logger.warning(f"Error obteniendo noticias en tiempo real: {e}")

        return market_data

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
                self.data_stream = StockDataStream(self.api_key, self.api_secret)
            
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

    def _generate_simulated_data(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: str = "1D",
    ) -> Dict[str, pd.DataFrame]:
        """Genera datos simulados realistas para entrenamiento de ML."""
        logger.info(f"Generando datos simulados realistas para entrenamiento ML de {len(symbols)} símbolos")

        # Calcular número de períodos
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        if timeframe == "1D":
            periods = (end_dt - start_dt).days
            freq = "D"
        elif timeframe == "1H":
            periods = int((end_dt - start_dt).total_seconds() / 3600)
            freq = "H"
        elif timeframe == "1Min":
            periods = int((end_dt - start_dt).total_seconds() / 60)
            freq = "Min"
        else:
            periods = (end_dt - start_dt).days
            freq = "D"

        # Generar fechas
        dates = pd.date_range(start=start_dt, periods=periods, freq=freq)

        result = {}
        np.random.seed(42)  # Para reproducibilidad

        for symbol in symbols:
            # Precios base realistas basados en símbolos comunes
            symbol_prices = {
                "AAPL": 180.0, "MSFT": 380.0, "GOOGL": 140.0, "AMZN": 155.0,
                "TSLA": 250.0, "NVDA": 850.0, "META": 330.0, "NFLX": 480.0,
                "SPY": 450.0, "QQQ": 380.0
            }
            base_price = symbol_prices.get(symbol.upper(), 200.0)

            # Crear patrones de precio aprendibles para ML
            prices = self._generate_learnable_price_pattern(base_price, periods, timeframe, symbol)

            # Crear OHLC con spreads realistas
            opens = prices[:-1]
            closes = prices[1:]

            # High/Low basados en movimientos intradiarios realistas
            intraday_range = 0.005  # 0.5% rango intradiario típico
            highs = np.maximum(opens, closes) * (1 + np.random.uniform(0, intraday_range, len(opens)))
            lows = np.minimum(opens, closes) * (1 - np.random.uniform(0, intraday_range, len(opens)))

            # Volumen realista basado en el símbolo
            base_volume = {
                "AAPL": 50000000, "MSFT": 25000000, "GOOGL": 25000000, "AMZN": 35000000,
                "TSLA": 80000000, "NVDA": 40000000, "META": 20000000, "NFLX": 8000000,
                "SPY": 80000000, "QQQ": 40000000
            }.get(symbol.upper(), 20000000)

            # Volumen consistente con variación controlada
            volume_variation = np.random.normal(1.0, 0.3, len(opens))
            volume_variation = np.clip(volume_variation, 0.5, 2.0)  # Limitar variación
            volumes = (base_volume * volume_variation).astype(int)

            # Ajustar volumen para diferentes timeframes
            if timeframe == "1H":
                volumes = volumes // 24
            elif timeframe == "1Min":
                volumes = volumes // (24 * 60)

            # Crear DataFrame
            df = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            }, index=dates[:-1])

            result[symbol] = df

        logger.info(f"Datos simulados realistas generados para entrenamiento ML de {len(result)} símbolos")
        return result

    def _generate_learnable_price_pattern(
        self,
        base_price: float,
        periods: int,
        timeframe: str,
        symbol: str
    ) -> np.ndarray:
        """Genera patrones de precio aprendibles para ML con tendencias y ciclos."""
        # Parámetros base para diferentes timeframes
        if timeframe == "1D":
            # Ciclos semanales, mensuales, trimestrales
            cycle_lengths = [5, 20, 60]  # días
            volatilities = [0.01, 0.015, 0.02]  # volatilidades por ciclo
            trend_strength = 0.0002  # tendencia general
        elif timeframe == "1H":
            # Ciclos diarios, semanales
            cycle_lengths = [24, 168]  # horas
            volatilities = [0.005, 0.008]
            trend_strength = 0.00005
        else:  # 1Min
            # Ciclos intradiarios
            cycle_lengths = [60, 240, 1440]  # minutos
            volatilities = [0.002, 0.003, 0.004]
            trend_strength = 0.00001

        # Generar componente de tendencia general
        trend = trend_strength * np.arange(periods)

        # Generar componentes cíclicos
        cycles = np.zeros(periods)
        for cycle_len, vol in zip(cycle_lengths, volatilities):
            if cycle_len < periods:
                # Ciclo sinusoidal con ruido
                t = np.arange(periods) * (2 * np.pi / cycle_len)
                cycle = np.sin(t) * vol
                # Añadir ruido controlado
                noise = np.random.normal(0, vol * 0.3, periods)
                cycles += cycle + noise

        # Componente estocástico reducido para mayor predictibilidad
        stochastic_vol = 0.005 if timeframe == "1D" else 0.002 if timeframe == "1H" else 0.001
        stochastic = np.random.normal(0, stochastic_vol, periods)
        stochastic = np.cumsum(stochastic) * 0.1  # Integrar para mayor suavidad

        # Combinar componentes
        price_changes = trend + cycles + stochastic

        # Aplicar cambios de precio
        prices = base_price * np.exp(np.cumsum(price_changes))

        # Añadir reversiones a la media periódicas (patrón aprendible)
        if periods > 50:
            # Cada cierto período, revertir parcialmente hacia la media móvil
            reversion_periods = periods // 10
            for i in range(reversion_periods, periods, reversion_periods):
                if i < len(prices):
                    # Calcular media móvil reciente
                    window = min(20, i)
                    ma = np.mean(prices[i-window:i])
                    # Revertir parcialmente
                    reversion_strength = 0.1
                    prices[i:] = prices[i:] * (1 - reversion_strength) + ma * reversion_strength

        return prices
