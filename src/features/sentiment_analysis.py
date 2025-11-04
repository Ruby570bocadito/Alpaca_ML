#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para el análisis de sentimiento en textos de noticias financieras.
"""

import logging
import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Analizador de sentimiento para textos financieros."""

    def __init__(self, config: Dict[str, Any]):
        """Inicializa el analizador de sentimiento.

        Args:
            config: Configuración del sistema
        """
        self.config = config

        # Configuración de análisis
        self.sentiment_config = {
            "min_text_length": int(config.get("MIN_TEXT_LENGTH", "10")),
            "use_vader": config.get("USE_VADER", True),
            "compound_threshold": float(config.get("COMPOUND_THRESHOLD", "0.05")),
            "aggregation_window_hours": int(config.get("AGGREGATION_WINDOW_HOURS", "24")),
        }

        # Inicializar VADER
        try:
            nltk.data.find('vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')

        self.sia = SentimentIntensityAnalyzer()

        # Palabras clave financieras positivas y negativas
        self.financial_lexicon = self._load_financial_lexicon()

        logger.info("Sentiment Analyzer inicializado")

    def analyze_news_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Analiza el sentimiento de un DataFrame de noticias.

        Args:
            news_df: DataFrame con noticias

        Returns:
            pd.DataFrame: DataFrame con análisis de sentimiento
        """
        if news_df.empty:
            return pd.DataFrame()

        try:
            # Copia del DataFrame
            result_df = news_df.copy()

            # Analizar sentimiento de cada artículo
            sentiment_scores = []
            for _, row in result_df.iterrows():
                text = row.get("full_text", "")
                if pd.isna(text) or len(text.strip()) < self.sentiment_config["min_text_length"]:
                    sentiment_scores.append({
                        "sentiment_compound": 0.0,
                        "sentiment_pos": 0.0,
                        "sentiment_neg": 0.0,
                        "sentiment_neu": 0.0,
                        "sentiment_label": "neutral",
                        "confidence": 0.0
                    })
                else:
                    scores = self._analyze_text_sentiment(text)
                    sentiment_scores.append(scores)

            # Agregar scores al DataFrame
            sentiment_df = pd.DataFrame(sentiment_scores)
            result_df = pd.concat([result_df, sentiment_df], axis=1)

            logger.debug(f"Análisis de sentimiento completado para {len(result_df)} artículos")
            return result_df

        except Exception as e:
            logger.error(f"Error analizando sentimiento: {e}", exc_info=True)
            return news_df

    def aggregate_sentiment_by_time(self, news_df: pd.DataFrame,
                                   time_window: str = "H") -> pd.DataFrame:
        """Agrega sentimiento por ventanas temporales.

        Args:
            news_df: DataFrame con noticias y análisis de sentimiento
            time_window: Ventana temporal (ej: 'H' para hora, 'D' para día)

        Returns:
            pd.DataFrame: DataFrame con sentimiento agregado por tiempo
        """
        if news_df.empty or "sentiment_compound" not in news_df.columns:
            return pd.DataFrame()

        try:
            # Asegurar que published_at sea datetime
            news_df = news_df.copy()
            news_df["published_at"] = pd.to_datetime(news_df["published_at"])

            # Agrupar por tiempo y calcular estadísticas
            grouped = news_df.groupby(pd.Grouper(key="published_at", freq=time_window))

            aggregated = grouped.agg({
                "sentiment_compound": ["mean", "std", "count", "sum"],
                "sentiment_pos": "mean",
                "sentiment_neg": "mean",
                "sentiment_neu": "mean",
                "title": "count"  # Número de artículos
            }).fillna(0)

            # Aplanar columnas
            aggregated.columns = ["_".join(col).strip() for col in aggregated.columns]
            aggregated = aggregated.rename(columns={
                "sentiment_compound_mean": "sentiment_mean",
                "sentiment_compound_std": "sentiment_std",
                "sentiment_compound_count": "article_count",
                "sentiment_compound_sum": "sentiment_sum",
                "sentiment_pos_mean": "pos_mean",
                "sentiment_neg_mean": "neg_mean",
                "sentiment_neu_mean": "neu_mean",
                "title_count": "total_articles"
            })

            # Calcular métricas adicionales
            aggregated["sentiment_trend"] = aggregated["sentiment_mean"].pct_change()
            aggregated["sentiment_volatility"] = aggregated["sentiment_std"] / aggregated["sentiment_mean"].abs().replace(0, 1)
            aggregated["sentiment_intensity"] = aggregated["sentiment_mean"].abs()

            # Etiquetar sentimiento general
            aggregated["overall_sentiment"] = aggregated["sentiment_mean"].apply(self._classify_sentiment)

            return aggregated

        except Exception as e:
            logger.error(f"Error agregando sentimiento por tiempo: {e}", exc_info=True)
            return pd.DataFrame()

    def create_sentiment_features(self, news_df: pd.DataFrame,
                                 market_data: pd.DataFrame) -> pd.DataFrame:
        """Crea features de sentimiento para integrar con datos de mercado.

        Args:
            news_df: DataFrame con noticias
            market_data: DataFrame con datos de mercado (OHLCV)

        Returns:
            pd.DataFrame: DataFrame con features de sentimiento
        """
        if news_df.empty or market_data.empty:
            return market_data.copy()

        try:
            # Analizar sentimiento si no está hecho
            if "sentiment_compound" not in news_df.columns:
                news_df = self.analyze_news_sentiment(news_df)

            # Agregar features de sentimiento al market_data
            market_with_sentiment = market_data.copy()

            # Features básicos de sentimiento
            market_with_sentiment["news_count_24h"] = 0.0
            market_with_sentiment["sentiment_mean_24h"] = 0.0
            market_with_sentiment["sentiment_std_24h"] = 0.0
            market_with_sentiment["sentiment_trend_24h"] = 0.0
            market_with_sentiment["pos_sentiment_ratio_24h"] = 0.0
            market_with_sentiment["neg_sentiment_ratio_24h"] = 0.0

            # Para cada fila de market_data, calcular features de sentimiento
            for idx, row in market_with_sentiment.iterrows():
                current_time = idx if isinstance(idx, pd.Timestamp) else pd.Timestamp(idx)

                # Ventana de 24 horas hacia atrás
                window_start = current_time - pd.Timedelta(hours=24)

                # Filtrar noticias en la ventana
                window_news = news_df[
                    (news_df["published_at"] >= window_start) &
                    (news_df["published_at"] <= current_time)
                ]

                if not window_news.empty:
                    # Calcular métricas
                    market_with_sentiment.at[idx, "news_count_24h"] = len(window_news)
                    market_with_sentiment.at[idx, "sentiment_mean_24h"] = window_news["sentiment_compound"].mean()
                    market_with_sentiment.at[idx, "sentiment_std_24h"] = window_news["sentiment_compound"].std()
                    market_with_sentiment.at[idx, "sentiment_trend_24h"] = window_news["sentiment_compound"].tail(5).mean() - window_news["sentiment_compound"].head(5).mean() if len(window_news) >= 10 else 0.0

                    # Ratios de sentimiento positivo/negativo
                    pos_count = (window_news["sentiment_compound"] > 0.1).sum()
                    neg_count = (window_news["sentiment_compound"] < -0.1).sum()
                    total_count = len(window_news)

                    market_with_sentiment.at[idx, "pos_sentiment_ratio_24h"] = pos_count / total_count if total_count > 0 else 0.0
                    market_with_sentiment.at[idx, "neg_sentiment_ratio_24h"] = neg_count / total_count if total_count > 0 else 0.0

            # Rellenar NaN
            market_with_sentiment = market_with_sentiment.fillna(0)

            logger.info(f"Features de sentimiento agregadas a datos de mercado")
            return market_with_sentiment

        except Exception as e:
            logger.error(f"Error creando features de sentimiento: {e}", exc_info=True)
            return market_data.copy()

    def _analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analiza el sentimiento de un texto usando VADER.

        Args:
            text: Texto a analizar

        Returns:
            Dict[str, float]: Scores de sentimiento
        """
        try:
            # Limpiar texto
            text = self._clean_text(text)

            # Análisis con VADER
            scores = self.sia.polarity_scores(text)

            # Ajustar scores basado en léxico financiero
            adjusted_scores = self._adjust_financial_sentiment(text, scores)

            # Clasificar sentimiento
            compound = adjusted_scores["compound"]
            if compound >= self.sentiment_config["compound_threshold"]:
                label = "positive"
            elif compound <= -self.sentiment_config["compound_threshold"]:
                label = "negative"
            else:
                label = "neutral"

            # Calcular confianza
            confidence = abs(compound)

            return {
                "sentiment_compound": compound,
                "sentiment_pos": adjusted_scores["pos"],
                "sentiment_neg": adjusted_scores["neg"],
                "sentiment_neu": adjusted_scores["neu"],
                "sentiment_label": label,
                "confidence": confidence
            }

        except Exception as e:
            logger.warning(f"Error analizando texto: {e}")
            return {
                "sentiment_compound": 0.0,
                "sentiment_pos": 0.0,
                "sentiment_neg": 0.0,
                "sentiment_neu": 1.0,
                "sentiment_label": "neutral",
                "confidence": 0.0
            }

    def _clean_text(self, text: str) -> str:
        """Limpia el texto para análisis de sentimiento.

        Args:
            text: Texto original

        Returns:
            str: Texto limpio
        """
        if not text:
            return ""

        # Convertir a minúsculas
        text = text.lower()

        # Remover URLs
        text = re.sub(r'http\S+', '', text)

        # Remover caracteres especiales pero mantener puntuación importante
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)

        # Remover espacios extra
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _adjust_financial_sentiment(self, text: str, scores: Dict[str, float]) -> Dict[str, float]:
        """Ajusta scores de sentimiento basado en contexto financiero.

        Args:
            text: Texto analizado
            scores: Scores originales de VADER

        Returns:
            Dict[str, float]: Scores ajustados
        """
        adjusted_scores = scores.copy()

        # Buscar palabras clave financieras
        text_lower = text.lower()

        # Aumentar positividad para palabras positivas financieras
        for word in self.financial_lexicon["positive"]:
            if word in text_lower:
                adjusted_scores["compound"] += 0.1
                adjusted_scores["pos"] += 0.1
                adjusted_scores["neu"] -= 0.05

        # Aumentar negatividad para palabras negativas financieras
        for word in self.financial_lexicon["negative"]:
            if word in text_lower:
                adjusted_scores["compound"] -= 0.1
                adjusted_scores["neg"] += 0.1
                adjusted_scores["neu"] -= 0.05

        # Normalizar scores
        total = adjusted_scores["pos"] + adjusted_scores["neg"] + adjusted_scores["neu"]
        if total > 0:
            adjusted_scores["pos"] /= total
            adjusted_scores["neg"] /= total
            adjusted_scores["neu"] /= total

        # Limitar compound entre -1 y 1
        adjusted_scores["compound"] = np.clip(adjusted_scores["compound"], -1, 1)

        return adjusted_scores

    def _classify_sentiment(self, compound_score: float) -> str:
        """Clasifica el sentimiento basado en el score compuesto.

        Args:
            compound_score: Score compuesto de sentimiento

        Returns:
            str: Etiqueta de sentimiento
        """
        if compound_score >= 0.1:
            return "positive"
        elif compound_score <= -0.1:
            return "negative"
        else:
            return "neutral"

    def _load_financial_lexicon(self) -> Dict[str, List[str]]:
        """Carga léxico financiero para ajuste de sentimiento.

        Returns:
            Dict[str, List[str]]: Léxico financiero
        """
        return {
            "positive": [
                "bullish", "bull", "buy", "long", "profit", "gains", "rally", "surge",
                "beat expectations", "earnings beat", "revenue growth", "profit increase",
                "upgrade", "outperform", "strong", "robust", "excellent", "breakthrough",
                "acquisition", "merger", "expansion", "growth", "momentum", "uptrend"
            ],
            "negative": [
                "bearish", "bear", "sell", "short", "loss", "decline", "drop", "fall",
                "miss expectations", "earnings miss", "revenue decline", "profit drop",
                "downgrade", "underperform", "weak", "disappointing", "terrible", "crash",
                "lawsuit", "scandal", "bankruptcy", "recession", "slump", "downtrend"
            ]
        }
