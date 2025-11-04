# src/data/historical_data.py

import yfinance as yf
import pandas as pd
import logging

class HistoricalDataManager:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Gestor de datos históricos inicializado")

    def get_historical_data(self, symbols, start_date, end_date, timeframe="1Day"):
        """
        Descarga datos históricos reales desde Yahoo Finance.
        Se ignora el timeframe en esta versión, pero se acepta para compatibilidad.
        """
        all_data = {}
        interval = "1d" if timeframe.lower().startswith("1") else "1h"  # básico

        for symbol in symbols:
            self.logger.info(f"Descargando datos históricos para {symbol} ({start_date} -> {end_date}, {interval})...")

            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=interval,
                progress=False,
                auto_adjust=False  # 👈 IMPORTANTE: mantiene las columnas originales
            )

            # Si no hay datos
            if data is None or data.empty:
                self.logger.error(f"No se descargaron datos para {symbol}.")
                continue

            # Normalizamos nombres de columnas
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = [col[0].lower() for col in data.columns]
            else:
                data.columns = [str(c).lower() for c in data.columns]

            # Si hay columnas duplicadas (por ejemplo 'close', 'close'), nos quedamos con la primera
            data = data.loc[:, ~data.columns.duplicated()]

            # Si existe 'adj close', la usamos en lugar de 'close' para precios ajustados
            if 'adj close' in data.columns:
                data['close'] = data['adj close']
                data = data.drop(columns=['adj close'])

            # Asegurar nombres estándar
            expected_cols = ['open', 'high', 'low', 'close', 'volume']
            print("COLUMNAS OBTENIDAS DE YFINANCE:", data.columns)
            print("PRIMERAS FILAS:", data.head())
            data = data[expected_cols].copy()
            data.reset_index(inplace=True)
            all_data[symbol] = data

        self.logger.info(f"Datos descargados para {len(all_data)} símbolos")
        return all_data
