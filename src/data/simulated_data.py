import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import yfinance as yf

logger = logging.getLogger(__name__)

class SimulatedDataManager:
    """Simula datos históricos para backtesting."""

    def __init__(self, config):
        self.logger = logging.getLogger(__name__)
        self.symbols = config.get("SYMBOLS", "AAPL,MSFT,GOOGL,AMZN,TSLA").split(",")
        self.lookback_days = config.get("lookback_days", 5)
        self.logger.info("Gestor de datos simulados inicializado")

    def get_historical_data(self, symbols, start_date, end_date, timeframe="1D"):
        """Genera datos históricos simulados."""
        self.logger.info(f"Generando datos simulados de {start_date} a {end_date} para: {symbols}")

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        # Generar un rango de fechas
        dates = pd.date_range(start=start, end=end, freq=timeframe)

        data = {}
        for symbol in symbols:
            # Simular precios con una caminata aleatoria
            prices = np.cumprod(1 + np.random.normal(0, 0.01, len(dates))) * 100
            df = pd.DataFrame({
                "timestamp": dates,
                "open": prices * np.random.uniform(0.99, 1.01, len(dates)),
                "high": prices * np.random.uniform(1.00, 1.02, len(dates)),
                "low": prices * np.random.uniform(0.98, 1.00, len(dates)),
                "close": prices,
                "volume": np.random.randint(100000, 1000000, len(dates)),
            })
            data[symbol] = df

        return data

    def get_realtime_data(self):
        """Simula datos en tiempo real usando datos históricos de Yahoo Finance"""
        data_dict = {}
        end = datetime.now()
        start = end - timedelta(days=self.lookback_days)

        for symbol in self.symbols:
            try:
                df = yf.download(symbol, start=start, end=end, interval="1m", progress=False)
                if df.empty:
                    logger.warning(f"No se pudieron descargar datos para {symbol}")
                    continue
                df = df.tail(100)  # simulamos últimos minutos
                df.reset_index(inplace=True)
                data_dict[symbol] = df
                logger.info(f"Datos simulados cargados para {symbol} ({len(df)} velas)")
            except Exception as e:
                logger.error(f"Error al descargar datos para {symbol}: {e}")
        return data_dict
