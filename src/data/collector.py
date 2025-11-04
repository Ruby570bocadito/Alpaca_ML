#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.execution.alpaca_client import AlpacaClient
from src.data.store import FeatureStore


logger = logging.getLogger(__name__)


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))


class DataCollector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = AlpacaClient(config)
        self.store = FeatureStore(config)

    def fetch_bars(self, symbol: str, timeframe: str = "1Day", lookback_days: int = 120) -> Optional[pd.DataFrame]:
        end = datetime.utcnow()
        start = end - timedelta(days=lookback_days)
        try:
            bars = self.client.get_bars(symbol, timeframe, start.isoformat(), end.isoformat(), limit=1000)
            df = pd.DataFrame(bars)
            if df.empty:
                return None
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"]) 
                df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error obteniendo barras para {symbol}: {e}")
            return None

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if {"close", "high", "low", "open"}.issubset(out.columns):
            out["sma_10"] = out["close"].rolling(10).mean()
            out["sma_20"] = out["close"].rolling(20).mean()
            out["rsi_14"] = rsi(out["close"], 14)
            out["volatility_20"] = out["close"].pct_change().rolling(20).std()
            out["atr_14"] = (out["high"] - out["low"]).rolling(14).mean()
        out.dropna(inplace=True)
        return out

    def collect_and_store(self, symbol: str, timeframe: str = "1Day", lookback_days: int = 180, feature_set: str = "default") -> bool:
        df = self.fetch_bars(symbol, timeframe, lookback_days)
        if df is None or df.empty:
            return False
        feats = self.engineer_features(df)
        self.store.save_features(symbol, feats, feature_set)
        return True