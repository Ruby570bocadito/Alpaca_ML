#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para la gestión de modelos de machine learning.
"""

import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any, Tuple

# Importaciones para ML
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
import joblib

logger = logging.getLogger(__name__)


class ModelManager:
    """Gestor de modelos de machine learning para el sistema de trading."""

    def __init__(self, config: Dict[str, Any]):
        """Inicializa el gestor de modelos.

        Args:
            config: Configuración del sistema
        """
        self.config = config
        self.models_dir = config.get("MODELS_DIR")
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Configuración de modelos
        self.model_config = {
            "type": config.get("MODEL_TYPE", "classifier"),  # classifier o regressor
            "horizon": int(config.get("PREDICTION_HORIZON", "1")),  # Horizonte de predicción
            "features": config.get("SELECTED_FEATURES", None),  # Lista de features a usar
            "cv_folds": int(config.get("CV_FOLDS", "5")),  # Folds para validación cruzada
        }
        
        # Modelos cargados en memoria
        self.models = {}
        
        logger.info("Gestor de modelos inicializado")

    def train_model(self, data: pd.DataFrame, symbol: str):
        """Método público para entrenar un modelo por símbolo."""
        return self._train_model(data, symbol)

    def train(self, features_data: Dict[str, pd.DataFrame], force_retrain: bool = False) -> Dict[str, Any]:
        """Entrena modelos para cada símbolo.

        Args:
            features_data: Diccionario con DataFrames de features por símbolo
            force_retrain: Forzar reentrenamiento aunque exista modelo guardado

        Returns:
            Dict[str, Any]: Resultados del entrenamiento
        """
        results = {}
        
        for symbol, df in features_data.items():
            try:
                # Verificar si ya existe un modelo entrenado
                model_path = self._get_model_path(symbol)
                if os.path.exists(model_path) and not force_retrain:
                    logger.info(f"Modelo para {symbol} ya existe. Omitiendo entrenamiento.")
                    # Cargar modelo existente
                    self._load_model(symbol)
                    results[symbol] = {"status": "loaded", "path": model_path}
                    continue
                
                # Preparar datos para entrenamiento
                X, y, feature_names = self._prepare_training_data(df)
                
                if X is None or y is None:
                    logger.warning(f"No hay suficientes datos para entrenar modelo para {symbol}")
                    results[symbol] = {"status": "error", "message": "Datos insuficientes"}
                    continue
                
                # Crear y entrenar modelo
                model, metrics = self._train_model(X, y, feature_names)
                
                # Guardar modelo
                self._save_model(symbol, model, feature_names)
                
                # Almacenar modelo en memoria
                self.models[symbol] = {
                    "model": model,
                    "feature_names": feature_names,
                    "trained_at": datetime.now().isoformat(),
                }
                
                results[symbol] = {
                    "status": "trained",
                    "metrics": metrics,
                    "path": model_path,
                }
                
                # Imprimir métricas detalladas
                logger.info(f"Modelo entrenado para {symbol}. Métricas detalladas:")
                for metric_name, value in metrics.items():
                    logger.info(f"  {metric_name}: {value:.4f}")

                logger.info(f"Modelo entrenado para {symbol}. Métricas: {metrics}")
                
            except Exception as e:
                logger.error(f"Error al entrenar modelo para {symbol}: {e}", exc_info=True)
                results[symbol] = {"status": "error", "message": str(e)}
        
        return results

    def predict(self, features_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Genera predicciones para cada símbolo.

        Args:
            features_data: Diccionario con DataFrames de features por símbolo

        Returns:
            Dict[str, pd.DataFrame]: Predicciones por símbolo
        """
        predictions = {}
        
        for symbol, df in features_data.items():
            try:
                # Cargar modelo si no está en memoria
                if symbol not in self.models:
                    loaded = self._load_model(symbol)
                    if not loaded:
                        logger.warning(f"No se encontró modelo para {symbol}. Omitiendo predicción.")
                        continue
                
                model_info = self.models[symbol]
                model = model_info["model"]
                feature_names = model_info["feature_names"]
                
                # Preparar datos para predicción
                X = self._prepare_prediction_data(df, feature_names)
                
                if X is None or X.empty:
                    logger.warning(f"No hay datos válidos para predicción de {symbol}")
                    continue
                
                # Generar predicciones
                if self.model_config["type"] == "classifier":
                    # Predicción de clase y probabilidades
                    y_pred = model.predict(X)
                    y_prob = model.predict_proba(X)[:, 1]  # Probabilidad de clase positiva
                    
                    # Crear DataFrame con resultados
                    pred_df = pd.DataFrame({
                        "prediction": y_pred,
                        "probability": y_prob,
                    }, index=X.index)
                    
                else:  # regressor
                    # Predicción de valor
                    y_pred = model.predict(X)
                    
                    # Crear DataFrame con resultados
                    pred_df = pd.DataFrame({
                        "prediction": y_pred,
                    }, index=X.index)
                
                predictions[symbol] = pred_df
                logger.debug(f"Predicciones generadas para {symbol}: {len(pred_df)} registros")
                
            except Exception as e:
                logger.error(f"Error al generar predicciones para {symbol}: {e}", exc_info=True)
        
        return predictions

    def _prepare_training_data(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], List[str]]:
        """Prepara datos para entrenamiento.

        Args:
            df: DataFrame con features

        Returns:
            Tuple: X (features), y (target), lista de nombres de features
        """
        if df is None or df.empty:
            return None, None, []
        
        # Eliminar filas con NaN
        df = df.dropna()
        
        if len(df) < 40:  # Mínimo de datos para entrenar
            logger.warning(f"Datos insuficientes para entrenamiento: {len(df)} filas")
            return None, None, []
        
        # Seleccionar variable objetivo según tipo de modelo
        if self.model_config["type"] == "classifier":
            target_col = "target_direction"
        else:  # regressor
            target_col = "future_return"
        
        if target_col not in df.columns:
            logger.error(f"Columna objetivo {target_col} no encontrada en los datos")
            return None, None, []
        
        # Seleccionar features
        exclude_cols = [
            "open", "high", "low", "close", "volume",  # Datos originales
            "future_return", "target_direction", "target_class"  # Variables objetivo
        ]
        
        if self.model_config["features"] is not None:
            # Usar lista específica de features
            feature_names = self.model_config["features"]
        else:
            # Usar todas las columnas excepto las excluidas
            feature_names = [col for col in df.columns if col not in exclude_cols]
        
        # Verificar que todas las features existen
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            logger.warning(f"Features no encontradas: {missing_features}")
            feature_names = [f for f in feature_names if f in df.columns]
        
        # Extraer features y target
        X = df[feature_names]
        y = df[target_col]
        
        return X, y, feature_names

    def _prepare_prediction_data(self, df: pd.DataFrame, feature_names: List[str]) -> Optional[pd.DataFrame]:
        """Prepara datos para predicción.

        Args:
            df: DataFrame con features
            feature_names: Lista de nombres de features requeridas

        Returns:
            Optional[pd.DataFrame]: DataFrame con features para predicción
        """
        if df is None or df.empty:
            logger.warning("DataFrame vacío o None para predicción")
            return None

        # Verificar que tenemos suficientes datos para features rolling
        min_required_rows = 10  # Reducido para permitir predicciones con menos datos
        if len(df) < min_required_rows:
            logger.warning(f"Datos insuficientes para predicción: {len(df)} filas, mínimo requerido: {min_required_rows}")
            return None

        # Verificar que todas las features requeridas existen
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            logger.warning(f"Features requeridas no encontradas: {missing_features}")
            return None

        # Seleccionar solo las features necesarias
        X = df[feature_names]

        # Verificar que tenemos datos después de seleccionar features
        if X.empty:
            logger.warning("No hay datos después de seleccionar features requeridas")
            return None

        # Rellenar NaN con 0 para predicción (en lugar de eliminar filas)
        X = X.fillna(0)

        # Verificar que queden datos después del procesamiento
        if X.empty:
            logger.warning("No quedan datos después del procesamiento para predicción")
            return None

        # Verificar que tenemos al menos una fila válida para predecir
        if len(X) == 0:
            logger.warning("No hay filas válidas para predicción después del procesamiento")
            return None

        # Para backtest, devolver todas las filas válidas; para live trading, usar la última
        if self.model_config.get("prediction_mode") == "backtest":
            logger.debug(f"Usando todas las filas para predicción en backtest: {len(X)} filas")
        else:
            # Para predicción en tiempo real, usar solo la última fila disponible (más reciente)
            if len(X) > 1:
                X = X.tail(1)
                logger.debug(f"Usando última fila para predicción: {len(X)} filas")

        return X

    def _train_model(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> Tuple[Any, Dict[str, float]]:
        """Entrena un modelo.

        Args:
            X: DataFrame con features
            y: Series con variable objetivo
            feature_names: Lista de nombres de features

        Returns:
            Tuple: Modelo entrenado, diccionario con métricas
        """
        # Crear validación cruzada temporal
        tscv = TimeSeriesSplit(n_splits=self.model_config["cv_folds"])
        
        # Crear pipeline con escalado y modelo
        if self.model_config["type"] == "classifier":
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ))
            ])
        else:  # regressor
            model = Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', GradientBoostingRegressor(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42
                ))
            ])
        
        # Métricas para validación cruzada
        cv_metrics = {
            "accuracy": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "mse": [],
        }
        
        # Validación cruzada temporal
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Entrenar modelo
            model.fit(X_train, y_train)
            
            # Evaluar modelo
            y_pred = model.predict(X_test)
            
            if self.model_config["type"] == "classifier":
                cv_metrics["accuracy"].append(accuracy_score(y_test, y_pred))
                # Get unique labels to determine pos_label
                unique_labels = np.unique(y_test)
                if len(unique_labels) > 1:
                    pos_label = unique_labels[1] if len(unique_labels) > 1 else unique_labels[0]
                    cv_metrics["precision"].append(precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0))
                    cv_metrics["recall"].append(recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0))
                    cv_metrics["f1"].append(f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0))
                else:
                    cv_metrics["precision"].append(0.0)
                    cv_metrics["recall"].append(0.0)
                    cv_metrics["f1"].append(0.0)
            
            cv_metrics["mse"].append(mean_squared_error(y_test, y_pred))
        
        # Calcular métricas promedio
        metrics = {}
        for metric, values in cv_metrics.items():
            if values:  # Solo si hay valores
                metrics[metric] = np.mean(values)
        
        # Entrenar modelo final con todos los datos
        model.fit(X, y)
        
        return model, metrics

    def _save_model(self, symbol: str, model: Any, feature_names: List[str]) -> bool:
        """Guarda un modelo entrenado.

        Args:
            symbol: Símbolo del instrumento
            model: Modelo entrenado
            feature_names: Lista de nombres de features

        Returns:
            bool: True si se guardó correctamente, False en caso contrario
        """
        try:
            model_path = self._get_model_path(symbol)
            
            # Guardar modelo y metadatos
            model_data = {
                "model": model,
                "feature_names": feature_names,
                "model_type": self.model_config["type"],
                "trained_at": datetime.now().isoformat(),
                "config": self.model_config,
            }
            
            joblib.dump(model_data, model_path)
            logger.info(f"Modelo guardado en: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error al guardar modelo para {symbol}: {e}", exc_info=True)
            return False

    def _load_model(self, symbol: str) -> bool:
        """Carga un modelo guardado.

        Args:
            symbol: Símbolo del instrumento

        Returns:
            bool: True si se cargó correctamente, False en caso contrario
        """
        try:
            model_path = self._get_model_path(symbol)
            
            if not os.path.exists(model_path):
                logger.warning(f"No existe modelo guardado para {symbol}")
                return False
            
            # Cargar modelo y metadatos
            model_data = joblib.load(model_path)
            
            # Almacenar en memoria
            self.models[symbol] = {
                "model": model_data["model"],
                "feature_names": model_data["feature_names"],
                "trained_at": model_data["trained_at"],
            }
            
            logger.info(f"Modelo cargado para {symbol} (entrenado: {model_data['trained_at']})")
            return True
            
        except Exception as e:
            logger.error(f"Error al cargar modelo para {symbol}: {e}", exc_info=True)
            return False

    def _get_model_path(self, symbol: str) -> str:
        """Obtiene la ruta del archivo de modelo.

        Args:
            symbol: Símbolo del instrumento

        Returns:
            str: Ruta del archivo
        """
        return os.path.join(self.models_dir, f"{symbol}_{self.model_config['type']}.joblib")