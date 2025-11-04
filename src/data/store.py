#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para el almacenamiento y gestión de datos (feature store).
"""

import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import sqlite3
import json
import time

logger = logging.getLogger(__name__)


class FeatureStore:
    """Gestor de almacenamiento de features para el sistema de trading."""

    def __init__(self, config: Dict[str, Any]):
        """Inicializa el feature store.

        Args:
            config: Configuración del sistema
        """
        self.config = config
        self.data_dir = config.get("DATA_DIR")
        self.feature_dir = os.path.join(self.data_dir, "features")
        os.makedirs(self.feature_dir, exist_ok=True)
        
        # Caché en memoria para acceso rápido
        self.feature_cache = {}
        
        logger.info("Feature store inicializado")

    def save_features(self, symbol: str, features: pd.DataFrame, feature_set: str = "default"):
        """Guarda features en el store.

        Args:
            symbol: Símbolo del instrumento
            features: DataFrame con features
            feature_set: Nombre del conjunto de features
        """
        try:
            # Guardar en caché en memoria
            cache_key = f"{symbol}_{feature_set}"
            self.feature_cache[cache_key] = features
            
            # Guardar en disco
            file_path = self._get_feature_path(symbol, feature_set)
            features.to_parquet(file_path)
            
            logger.debug(f"Features guardadas para {symbol} (set: {feature_set})")
            
        except Exception as e:
            logger.error(f"Error al guardar features para {symbol}: {e}", exc_info=True)

    def load_features(self, symbol: str, feature_set: str = "default") -> Optional[pd.DataFrame]:
        """Carga features desde el store.

        Args:
            symbol: Símbolo del instrumento
            feature_set: Nombre del conjunto de features

        Returns:
            Optional[pd.DataFrame]: DataFrame con features o None si no existe
        """
        try:
            # Intentar cargar desde caché en memoria
            cache_key = f"{symbol}_{feature_set}"
            if cache_key in self.feature_cache:
                return self.feature_cache[cache_key]
            
            # Cargar desde disco
            file_path = self._get_feature_path(symbol, feature_set)
            if os.path.exists(file_path):
                features = pd.read_parquet(file_path)
                self.feature_cache[cache_key] = features
                logger.debug(f"Features cargadas para {symbol} (set: {feature_set})")
                return features
            
            logger.warning(f"No se encontraron features para {symbol} (set: {feature_set})")
            return None
            
        except Exception as e:
            logger.error(f"Error al cargar features para {symbol}: {e}", exc_info=True)
            return None

    def update_features(self, symbol: str, new_features: pd.DataFrame, feature_set: str = "default"):
        """Actualiza features existentes con nuevos datos.

        Args:
            symbol: Símbolo del instrumento
            new_features: DataFrame con nuevas features
            feature_set: Nombre del conjunto de features
        """
        try:
            # Cargar features existentes
            existing_features = self.load_features(symbol, feature_set)
            
            if existing_features is not None:
                # Combinar features existentes con nuevas
                combined = pd.concat([existing_features, new_features])
                # Eliminar duplicados por índice (timestamp)
                combined = combined[~combined.index.duplicated(keep='last')]
                # Ordenar por índice
                combined = combined.sort_index()
                
                # Guardar features actualizadas
                self.save_features(symbol, combined, feature_set)
                logger.debug(f"Features actualizadas para {symbol} (set: {feature_set})")
            else:
                # Si no existen features previas, guardar las nuevas
                self.save_features(symbol, new_features, feature_set)
                logger.debug(f"Nuevas features creadas para {symbol} (set: {feature_set})")
                
        except Exception as e:
            logger.error(f"Error al actualizar features para {symbol}: {e}", exc_info=True)

    def get_latest_features(self, symbol: str, feature_set: str = "default") -> Optional[pd.DataFrame]:
        """Obtiene las features más recientes.

        Args:
            symbol: Símbolo del instrumento
            feature_set: Nombre del conjunto de features

        Returns:
            Optional[pd.DataFrame]: DataFrame con la última fila de features o None
        """
        features = self.load_features(symbol, feature_set)
        if features is not None and not features.empty:
            return features.iloc[[-1]]
        return None

    def clear_cache(self):
        """Limpia la caché en memoria."""
        self.feature_cache.clear()
        logger.debug("Caché de features limpiada")

    def _get_feature_path(self, symbol: str, feature_set: str) -> str:
        """Obtiene la ruta del archivo de features.

        Args:
            symbol: Símbolo del instrumento
            feature_set: Nombre del conjunto de features

        Returns:
            str: Ruta del archivo
        """
        return os.path.join(self.feature_dir, f"{symbol}_{feature_set}.parquet")


class DataCache:
    """Caché de datos para el sistema de trading."""

    def __init__(self, config: Dict[str, Any]):
        """Inicializa el caché de datos.

        Args:
            config: Configuración del sistema
        """
        self.config = config
        self.data_dir = config.get("DATA_DIR")
        self.cache_dir = os.path.join(self.data_dir, "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Configuración de caché
        self.max_cache_age = config.get("MAX_CACHE_AGE_HOURS", 24)  # Horas
        
        logger.info("Caché de datos inicializado")

    def save_data(self, key: str, data: pd.DataFrame, category: str = "market_data"):
        """Guarda datos en el caché.

        Args:
            key: Clave de identificación (ej: símbolo)
            data: DataFrame con datos
            category: Categoría de datos
        """
        try:
            file_path = self._get_cache_path(key, category)
            data.to_parquet(file_path)
            logger.debug(f"Datos guardados en caché: {key} ({category})")
            
        except Exception as e:
            logger.error(f"Error al guardar datos en caché: {e}", exc_info=True)

    def load_data(self, key: str, category: str = "market_data") -> Optional[pd.DataFrame]:
        """Carga datos desde el caché.

        Args:
            key: Clave de identificación (ej: símbolo)
            category: Categoría de datos

        Returns:
            Optional[pd.DataFrame]: DataFrame con datos o None si no existe o está expirado
        """
        try:
            file_path = self._get_cache_path(key, category)
            
            if not os.path.exists(file_path):
                logger.debug(f"No se encontró caché para: {key} ({category})")
                return None
            
            # Verificar si el caché está expirado
            if self._is_cache_expired(file_path):
                logger.debug(f"Caché expirado para: {key} ({category})")
                return None
            
            data = pd.read_parquet(file_path)
            logger.debug(f"Datos cargados desde caché: {key} ({category})")
            return data
            
        except Exception as e:
            logger.error(f"Error al cargar datos desde caché: {e}", exc_info=True)
            return None

    def clear_expired_cache(self):
        """Limpia entradas de caché expiradas."""
        try:
            count = 0
            for filename in os.listdir(self.cache_dir):
                file_path = os.path.join(self.cache_dir, filename)
                if os.path.isfile(file_path) and self._is_cache_expired(file_path):
                    os.remove(file_path)
                    count += 1
            
            if count > 0:
                logger.info(f"Se eliminaron {count} archivos de caché expirados")
                
        except Exception as e:
            logger.error(f"Error al limpiar caché expirado: {e}", exc_info=True)

    def _get_cache_path(self, key: str, category: str) -> str:
        """Obtiene la ruta del archivo de caché.

        Args:
            key: Clave de identificación
            category: Categoría de datos

        Returns:
            str: Ruta del archivo
        """
        return os.path.join(self.cache_dir, f"{category}_{key}.parquet")

    def _is_cache_expired(self, file_path: str) -> bool:
        """Verifica si un archivo de caché está expirado.

        Args:
            file_path: Ruta del archivo

        Returns:
            bool: True si está expirado, False en caso contrario
        """
        if self.max_cache_age <= 0:
            return False
            
        file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
        age_hours = (datetime.now() - file_time).total_seconds() / 3600
        
        return age_hours > self.max_cache_age


class PersistenceStore:
    """Persistencia ligera en SQLite para jobs, órdenes y señales."""

    def __init__(self, db_path: str = "data/persistence.db"):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as conn:
            c = conn.cursor()
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  job_type TEXT,
                  payload TEXT,
                  status TEXT,
                  created_at REAL,
                  updated_at REAL
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  symbol TEXT,
                  side TEXT,
                  qty REAL,
                  status TEXT,
                  external_id TEXT,
                  meta TEXT,
                  created_at REAL,
                  updated_at REAL
                )
                """
            )
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  symbol TEXT,
                  timeframe TEXT,
                  signal TEXT,
                  strength REAL,
                  meta TEXT,
                  created_at REAL
                )
                """
            )
            conn.commit()

    def record_job(self, job_type: str, payload: Dict[str, Any], status: str = "queued") -> int:
        now = time.time()
        with self._conn() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO jobs (job_type, payload, status, created_at, updated_at) VALUES (?,?,?,?,?)",
                (job_type, json.dumps(payload), status, now, now),
            )
            conn.commit()
            return c.lastrowid

    def update_job_status(self, job_id: int, status: str):
        with self._conn() as conn:
            c = conn.cursor()
            c.execute("UPDATE jobs SET status=?, updated_at=? WHERE id=?", (status, time.time(), job_id))
            conn.commit()

    def list_recent_jobs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            c = conn.cursor()
            c.execute("SELECT id, job_type, payload, status, created_at, updated_at FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
            rows = c.fetchall()
            return [
                {
                    "id": r[0],
                    "job_type": r[1],
                    "payload": json.loads(r[2]) if r[2] else None,
                    "status": r[3],
                    "created_at": r[4],
                    "updated_at": r[5],
                }
                for r in rows
            ]

    def record_order(self, symbol: str, side: str, qty: float, status: str, external_id: Optional[str] = None, meta: Optional[Dict[str, Any]] = None) -> int:
        now = time.time()
        with self._conn() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO orders (symbol, side, qty, status, external_id, meta, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (symbol, side, qty, status, external_id, json.dumps(meta or {}), now, now),
            )
            conn.commit()
            return c.lastrowid

    def update_order_status(self, order_id: int, status: str, meta: Optional[Dict[str, Any]] = None):
        with self._conn() as conn:
            c = conn.cursor()
            c.execute(
                "UPDATE orders SET status=?, meta=?, updated_at=? WHERE id=?",
                (status, json.dumps(meta or {}), time.time(), order_id),
            )
            conn.commit()

    def list_recent_orders(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            c = conn.cursor()
            c.execute(
                "SELECT id, symbol, side, qty, status, external_id, meta, created_at, updated_at FROM orders ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            rows = c.fetchall()
            return [
                {
                    "id": r[0],
                    "symbol": r[1],
                    "side": r[2],
                    "qty": r[3],
                    "status": r[4],
                    "external_id": r[5],
                    "meta": json.loads(r[6]) if r[6] else None,
                    "created_at": r[7],
                    "updated_at": r[8],
                }
                for r in rows
            ]

    def record_signal(self, symbol: str, timeframe: str, signal: str, strength: float, meta: Optional[Dict[str, Any]] = None) -> int:
        now = time.time()
        with self._conn() as conn:
            c = conn.cursor()
            c.execute(
                "INSERT INTO signals (symbol, timeframe, signal, strength, meta, created_at) VALUES (?,?,?,?,?,?)",
                (symbol, timeframe, signal, strength, json.dumps(meta or {}), now),
            )
            conn.commit()
            return c.lastrowid

    def list_recent_signals(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            c = conn.cursor()
            c.execute(
                "SELECT id, symbol, timeframe, signal, strength, meta, created_at FROM signals ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            rows = c.fetchall()
            return [
                {
                    "id": r[0],
                    "symbol": r[1],
                    "timeframe": r[2],
                    "signal": r[3],
                    "strength": r[4],
                    "meta": json.loads(r[5]) if r[5] else None,
                    "created_at": r[6],
                }
                for r in rows
            ]