#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para la ingesta de noticias financieras desde APIs externas.
"""

import logging
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json
import os

logger = logging.getLogger(__name__)


class NewsIngestionManager:
    """Gestor de ingesta de noticias financieras."""

    def __init__(self, config: Dict[str, Any]):
        """Inicializa el gestor de noticias.

        Args:
            config: Configuración del sistema
        """
        self.config = config
        self.api_key = config.get("NEWSAPI_KEY")
        self.base_url = "https://newsapi.org/v2"
        self.cache_dir = config.get("NEWS_CACHE_DIR", "data/news_cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        # Configuración de búsqueda
        self.news_config = {
            "sources": config.get("NEWS_SOURCES", "bloomberg,reuters,cnbc,financial-times"),
            "language": config.get("NEWS_LANGUAGE", "en"),
            "sort_by": config.get("NEWS_SORT_BY", "publishedAt"),
            "max_articles_per_request": int(config.get("MAX_ARTICLES_PER_REQUEST", "100")),
            "cache_expiry_hours": int(config.get("NEWS_CACHE_EXPIRY_HOURS", "1")),
        }

        # Cache en memoria
        self.cache = {}

        if not self.api_key:
            logger.warning("NEWSAPI_KEY no configurada. El módulo de noticias no funcionará.")
        else:
            logger.info("News Ingestion Manager inicializado")

    def get_news_for_symbol(self, symbol: str, days_back: int = 7,
                           max_articles: int = 50) -> pd.DataFrame:
        """Obtiene noticias relacionadas con un símbolo específico.

        Args:
            symbol: Símbolo del instrumento (ej: AAPL, TSLA)
            days_back: Número de días hacia atrás para buscar
            max_articles: Máximo número de artículos a retornar

        Returns:
            pd.DataFrame: DataFrame con artículos de noticias
        """
        try:
            # Verificar cache primero
            cache_key = f"{symbol}_{days_back}_{max_articles}"
            if cache_key in self.cache:
                cached_data, timestamp = self.cache[cache_key]
                if (datetime.now() - timestamp).total_seconds() < self.news_config["cache_expiry_hours"] * 3600:
                    logger.debug(f"Usando datos de noticias cacheados para {symbol}")
                    return cached_data

            # Construir query de búsqueda
            query = self._build_search_query(symbol)

            # Calcular fechas
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)

            # Hacer request a NewsAPI
            articles = self._fetch_news(query, start_date, end_date, max_articles)

            if not articles:
                logger.warning(f"No se encontraron noticias para {symbol}")
                return pd.DataFrame()

            # Convertir a DataFrame
            df = self._articles_to_dataframe(articles, symbol)

            # Guardar en cache
            self.cache[cache_key] = (df, datetime.now())

            # Guardar en archivo para persistencia
            self._save_to_cache_file(symbol, df)

            logger.info(f"Obtenidas {len(df)} noticias para {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error obteniendo noticias para {symbol}: {e}", exc_info=True)
            return pd.DataFrame()

    def get_market_news(self, symbols: List[str], days_back: int = 1) -> Dict[str, pd.DataFrame]:
        """Obtiene noticias de mercado para múltiples símbolos.

        Args:
            symbols: Lista de símbolos
            days_back: Días hacia atrás

        Returns:
            Dict[str, pd.DataFrame]: Diccionario con noticias por símbolo
        """
        market_news = {}

        for symbol in symbols:
            try:
                news_df = self.get_news_for_symbol(symbol, days_back)
                if not news_df.empty:
                    market_news[symbol] = news_df
            except Exception as e:
                logger.error(f"Error obteniendo noticias para {symbol}: {e}")

        return market_news

    def _build_search_query(self, symbol: str) -> str:
        """Construye la query de búsqueda para NewsAPI.

        Args:
            symbol: Símbolo del instrumento

        Returns:
            str: Query de búsqueda
        """
        # Mapeo de símbolos a nombres de compañías para mejor búsqueda
        company_names = {
            "AAPL": "Apple OR Apple Inc",
            "MSFT": "Microsoft OR Microsoft Corporation",
            "GOOGL": "Google OR Alphabet",
            "AMZN": "Amazon OR Amazon.com",
            "TSLA": "Tesla OR Tesla Motors",
            "NVDA": "NVIDIA OR Nvidia",
            "META": "Meta OR Facebook",
            "NFLX": "Netflix",
            "SPY": "S&P 500 OR SP500",
            "QQQ": "Nasdaq OR NASDAQ",
        }

        # Usar nombre de compañía si está disponible, sino el símbolo
        query_terms = company_names.get(symbol.upper(), symbol)

        # Agregar términos financieros relevantes
        query = f'({query_terms}) AND (stock OR shares OR market OR trading OR earnings OR revenue OR profit)'

        return query

    def _fetch_news(self, query: str, start_date: datetime, end_date: datetime,
                   max_articles: int) -> List[Dict]:
        """Hace la petición a NewsAPI.

        Args:
            query: Query de búsqueda
            start_date: Fecha de inicio
            end_date: Fecha de fin
            max_articles: Máximo número de artículos

        Returns:
            List[Dict]: Lista de artículos
        """
        if not self.api_key:
            logger.error("NEWSAPI_KEY no configurada")
            return []

        all_articles = []
        page = 1
        max_pages = min(5, (max_articles // self.news_config["max_articles_per_request"]) + 1)

        while len(all_articles) < max_articles and page <= max_pages:
            try:
                params = {
                    "q": query,
                    "from": start_date.strftime("%Y-%m-%d"),
                    "to": end_date.strftime("%Y-%m-%d"),
                    "sortBy": self.news_config["sort_by"],
                    "language": self.news_config["language"],
                    "pageSize": min(self.news_config["max_articles_per_request"], max_articles - len(all_articles)),
                    "page": page,
                    "apiKey": self.api_key
                }

                response = requests.get(f"{self.base_url}/everything", params=params, timeout=10)
                response.raise_for_status()

                data = response.json()

                if data.get("status") == "ok":
                    articles = data.get("articles", [])
                    all_articles.extend(articles)

                    # Si no hay más artículos, salir
                    if len(articles) < params["pageSize"]:
                        break
                else:
                    logger.error(f"Error en respuesta de NewsAPI: {data}")
                    break

                page += 1
                time.sleep(1)  # Rate limiting

            except requests.RequestException as e:
                logger.error(f"Error en petición a NewsAPI: {e}")
                break

        return all_articles[:max_articles]

    def _articles_to_dataframe(self, articles: List[Dict], symbol: str) -> pd.DataFrame:
        """Convierte lista de artículos a DataFrame.

        Args:
            articles: Lista de artículos de NewsAPI
            symbol: Símbolo asociado

        Returns:
            pd.DataFrame: DataFrame con artículos
        """
        if not articles:
            return pd.DataFrame()

        # Extraer datos relevantes
        processed_articles = []
        for article in articles:
            try:
                processed_article = {
                    "symbol": symbol,
                    "title": article.get("title", ""),
                    "description": article.get("description", ""),
                    "content": article.get("content", ""),
                    "url": article.get("url", ""),
                    "source": article.get("source", {}).get("name", ""),
                    "published_at": pd.to_datetime(article.get("publishedAt")),
                    "author": article.get("author", ""),
                    "url_to_image": article.get("urlToImage", ""),
                }
                processed_articles.append(processed_article)
            except Exception as e:
                logger.warning(f"Error procesando artículo: {e}")
                continue

        df = pd.DataFrame(processed_articles)

        # Limpiar datos
        df = df.dropna(subset=["published_at"])
        df = df.sort_values("published_at", ascending=False)

        # Crear texto completo para análisis
        df["full_text"] = df["title"].fillna("") + " " + df["description"].fillna("") + " " + df["content"].fillna("")

        return df

    def _save_to_cache_file(self, symbol: str, df: pd.DataFrame):
        """Guarda datos en archivo de cache.

        Args:
            symbol: Símbolo
            df: DataFrame con noticias
        """
        try:
            cache_file = os.path.join(self.cache_dir, f"{symbol}_news_cache.json")

            # Convertir a formato serializable
            cache_data = {
                "timestamp": datetime.now().isoformat(),
                "articles": df.to_dict("records")
            }

            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2, default=str)

        except Exception as e:
            logger.warning(f"Error guardando cache para {symbol}: {e}")

    def _load_from_cache_file(self, symbol: str) -> Optional[pd.DataFrame]:
        """Carga datos desde archivo de cache.

        Args:
            symbol: Símbolo

        Returns:
            Optional[pd.DataFrame]: DataFrame con noticias cacheadas
        """
        try:
            cache_file = os.path.join(self.cache_dir, f"{symbol}_news_cache.json")

            if not os.path.exists(cache_file):
                return None

            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)

            # Verificar si el cache no ha expirado
            cache_timestamp = pd.to_datetime(cache_data["timestamp"])
            if (datetime.now() - cache_timestamp).total_seconds() > self.news_config["cache_expiry_hours"] * 3600:
                return None

            # Convertir a DataFrame
            df = pd.DataFrame(cache_data["articles"])
            df["published_at"] = pd.to_datetime(df["published_at"])

            return df

        except Exception as e:
            logger.warning(f"Error cargando cache para {symbol}: {e}")
            return None
