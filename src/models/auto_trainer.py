"""
Sistema de entrenamiento automático programado para modelos ML.
"""
import os
import sys
import logging
import schedule
import time
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Any, Optional
import joblib
from dotenv import load_dotenv

# Importaciones locales
from ..data.ingest import DataIngestionManager
from ..features.engineering import FeatureEngineer
from ..models.model import ModelManager
from ..config import load_config

# Cargar variables de entorno
load_dotenv()

# Configuración (valores por defecto desde entorno)
MODELS_DIR = os.getenv("MODELS_DIR", "models")
TRAINING_SCHEDULE = os.getenv("TRAINING_SCHEDULE", "weekly")  # daily, weekly, monthly
SYMBOLS = os.getenv("TRAINING_SYMBOLS", "SPY,QQQ,AAPL,MSFT,GOOGL").split(",")
TIMEFRAMES = os.getenv("TRAINING_TIMEFRAMES", "1D,1H").split(",")
LOOKBACK_DAYS = int(os.getenv("TRAINING_LOOKBACK_DAYS", "365"))
MODEL_MIN_F1 = float(os.getenv("MODEL_MIN_F1", "0.55"))
MODEL_MIN_ACCURACY = float(os.getenv("MODEL_MIN_ACCURACY", "0.55"))
MODEL_MAX_MSE = float(os.getenv("MODEL_MAX_MSE", "0.25"))
MODEL_ACTIVATE_ON_TRAIN = os.getenv("MODEL_ACTIVATE_ON_TRAIN", "true").lower() == "true"

# Configurar logging
logger = logging.getLogger(__name__)

class AutoTrainer:
    """
    Sistema de entrenamiento automático programado para modelos ML.
    """

    def __init__(
        self,
        models_dir: str = MODELS_DIR,
        training_schedule: str = TRAINING_SCHEDULE,
        symbols: List[str] = None,
        timeframes: List[str] = None,
        lookback_days: int = LOOKBACK_DAYS,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Inicializa el sistema de entrenamiento automático.

        Args:
            models_dir: Directorio para guardar modelos
            training_schedule: Frecuencia de entrenamiento (daily, weekly, monthly)
            symbols: Lista de símbolos para entrenar
            timeframes: Lista de timeframes para entrenar
            lookback_days: Días de histórico para entrenamiento
            config: Configuración del sistema (si no se pasa, se carga automáticamente)
        """
        # Configuración
        self.config = config or load_config()

        self.models_dir = models_dir or self.config.get("MODELS_DIR", MODELS_DIR)
        self.training_schedule = training_schedule or self.config.get("TRAINING_SCHEDULE", TRAINING_SCHEDULE)
        self.symbols = symbols or self.config.get("TRAINING_SYMBOLS", ",".join(SYMBOLS)).split(",")
        self.timeframes = timeframes or self.config.get("TRAINING_TIMEFRAMES", ",".join(TIMEFRAMES)).split(",")
        self.lookback_days = lookback_days or int(self.config.get("TRAINING_LOOKBACK_DAYS", LOOKBACK_DAYS))

        # Componentes internos
        # Alinear MODELS_DIR de la configuración con el argumento recibido
        self.config["MODELS_DIR"] = self.models_dir
        self.data_ingester = DataIngestionManager(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.model_manager = ModelManager(self.config)

        # Crear directorio de modelos si no existe
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(os.path.join(self.models_dir, "active"), exist_ok=True)

        logger.info(
            f"AutoTrainer inicializado con schedule={self.training_schedule}, symbols={len(self.symbols)}, timeframes={self.timeframes}"
        )
    
    def setup_schedule(self):
        """
        Configura el schedule de entrenamiento automático.
        """
        if self.training_schedule == "daily":
            # Entrenar todos los días a las 2 AM
            schedule.every().day.at("02:00").do(self.train_all_models)
        elif self.training_schedule == "weekly":
            # Entrenar todos los domingos a las 3 AM
            schedule.every().sunday.at("03:00").do(self.train_all_models)
        elif self.training_schedule == "monthly":
            # Entrenar el primer día de cada mes a las 4 AM (schedule no soporta 'month')
            def _run_if_first_of_month():
                if datetime.now().day == 1:
                    self.train_all_models()
            schedule.every().day.at("04:00").do(_run_if_first_of_month)
        else:
            logger.warning(f"Schedule no reconocido: {self.training_schedule}")
        
        logger.info(f"Schedule configurado: {self.training_schedule}")
    
    def train_all_models(self) -> Dict[str, Any]:
        """
        Entrena todos los modelos para todos los símbolos y timeframes.
        
        Returns:
            Diccionario con resultados del entrenamiento
        """
        start_time = datetime.now()
        results = {
            "start_time": start_time.isoformat(),
            "models_trained": 0,
            "errors": 0,
            "details": []
        }
        
        logger.info(f"Iniciando entrenamiento de todos los modelos: {len(self.symbols)} símbolos, "
                   f"{len(self.timeframes)} timeframes")
        
        for symbol in self.symbols:
            for timeframe in self.timeframes:
                try:
                    # Entrenar modelo para este símbolo y timeframe
                    model_result = self.train_model(symbol, timeframe)
                    
                    if model_result["success"]:
                        results["models_trained"] += 1
                    else:
                        results["errors"] += 1
                    
                    results["details"].append({
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "success": model_result["success"],
                        "model_path": model_result.get("model_path"),
                        "error": model_result.get("error")
                    })
                    
                except Exception as e:
                    logger.error(f"Error al entrenar modelo para {symbol} {timeframe}: {str(e)}")
                    results["errors"] += 1
                    results["details"].append({
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "success": False,
                        "error": str(e)
                    })
        
        # Calcular tiempo total
        end_time = datetime.now()
        results["end_time"] = end_time.isoformat()
        results["duration_seconds"] = (end_time - start_time).total_seconds()
        
        logger.info(f"Entrenamiento completado: {results['models_trained']} modelos entrenados, "
                   f"{results['errors']} errores, duración: {results['duration_seconds']} segundos")
        
        return results
    
    def train_model(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Entrena un modelo para un símbolo y timeframe específicos.
        
        Args:
            symbol: Símbolo para entrenar
            timeframe: Timeframe para entrenar
            
        Returns:
            Diccionario con resultado del entrenamiento
        """
        logger.info(f"Entrenando modelo para {symbol} {timeframe}")
        
        try:
            # 1. Obtener datos históricos
            start_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime("%Y-%m-%d")
            end_date = datetime.now().strftime("%Y-%m-%d")

            historical = self.data_ingester.get_historical_data(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
            )

            df = historical.get(symbol)
            if df is None or df.empty or len(df) < 30:  # Mínimo de datos requeridos
                return {
                    "success": False,
                    "error": f"Datos insuficientes para {symbol} {timeframe}"
                }

            # 2. Generar características
            features_df = self.feature_engineer.process_data({symbol: df}).get(symbol)
            if features_df is None or features_df.empty:
                return {"success": False, "error": "No se pudieron generar features"}

            # 2b. Crear variable objetivo y features retardadas
            features_df = self.feature_engineer.create_target_variable(
                features_df,
                horizon=int(self.config.get("PREDICTION_HORIZON", "1")),
            )
            features_df = self.feature_engineer.create_lagged_features(
                features_df,
                lag_periods=[1, 2, 3, 5, 10],
            ).dropna()

            features_data = {symbol: features_df}

            # 3. Entrenar modelo usando ModelManager
            training_results = self.model_manager.train(features_data, force_retrain=True)
            result = training_results.get(symbol, {})
            if result.get("status") not in ("trained", "loaded"):
                return {"success": False, "error": result.get("message", "Fallo de entrenamiento")}

            metrics = result.get("metrics", {})
            model_path = result.get("path")

            # Guardar métricas junto al modelo
            try:
                metrics_filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d')}_metrics.json"
                metrics_path = os.path.join(self.models_dir, metrics_filename)
                pd.Series(metrics).to_json(metrics_path)
            except Exception:
                pass

            logger.info(f"Modelo entrenado y guardado: {model_path}")

            # Validación continua y activación controlada
            try:
                if MODEL_ACTIVATE_ON_TRAIN and self._is_model_valid(metrics):
                    self._activate_model(symbol, timeframe, model_path, metrics)
                else:
                    logger.info(
                        f"Modelo no activado: criterios no cumplidos (symbol={symbol}, timeframe={timeframe})",
                    )
            except Exception as e:
                logger.warning(f"Fallo al activar modelo validado: {str(e)}")

            return {
                "success": True,
                "model_path": model_path,
                "metrics": metrics,
                "data_points": len(df),
                "features": features_df.shape[1],
            }

        except Exception as e:
            logger.error(f"Error al entrenar modelo para {symbol} {timeframe}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_scheduler(self):
        """
        Ejecuta el scheduler en un bucle infinito.
        """
        self.setup_schedule()
        
        logger.info("Iniciando scheduler de entrenamiento automático")
        
        while True:
            schedule.run_pending()
            time.sleep(60)  # Verificar cada minuto
    
    def get_latest_model(self, symbol: str, timeframe: str) -> Optional[str]:
        """
        Obtiene la ruta al modelo más reciente para un símbolo y timeframe.
        
        Args:
            symbol: Símbolo del modelo
            timeframe: Timeframe del modelo
            
        Returns:
            Ruta al modelo o None si no existe
        """
        try:
            # Buscar archivos de modelo para este símbolo y timeframe
            prefix = f"{symbol}_{timeframe}_"
            model_files = [
                f for f in os.listdir(self.models_dir)
                if f.startswith(prefix) and f.endswith(".joblib")
            ]
            
            if not model_files:
                return None
            
            # Ordenar por fecha (parte del nombre del archivo)
            model_files.sort(reverse=True)
            
            # Devolver la ruta completa al modelo más reciente
            return os.path.join(self.models_dir, model_files[0])
            
        except Exception as e:
            logger.error(f"Error al buscar modelo para {symbol} {timeframe}: {str(e)}")
            return None
    
    def load_model(self, model_path: str):
        """
        Carga un modelo desde un archivo.
        
        Args:
            model_path: Ruta al archivo del modelo
            
        Returns:
            Modelo cargado
        """
        try:
            return joblib.load(model_path)
        except Exception as e:
            logger.error(f"Error al cargar modelo {model_path}: {str(e)}")
            return None

    # --- Validación y despliegue controlado ---
    def _is_model_valid(self, metrics: Dict[str, Any]) -> bool:
        """Valida métricas contra umbrales mínimos."""
        try:
            f1 = float(metrics.get("f1", 0))
            acc = float(metrics.get("accuracy", 0))
            mse = float(metrics.get("mse", 1e9))
            valid = (f1 >= MODEL_MIN_F1) and (acc >= MODEL_MIN_ACCURACY) and (mse <= MODEL_MAX_MSE)
            logger.info(
                f"Validación de modelo: f1={f1:.3f} acc={acc:.3f} mse={mse:.4f} -> {'OK' if valid else 'RECHAZADO'}"
            )
            return valid
        except Exception as e:
            logger.warning(f"Error validando métricas: {str(e)}")
            return False

    def _activate_model(self, symbol: str, timeframe: str, model_path: str, metrics: Dict[str, Any]) -> None:
        """Activa el modelo si supera la validación, promoviendo a 'active/'."""
        active_dir = os.path.join(self.models_dir, "active")
        os.makedirs(active_dir, exist_ok=True)
        active_name = f"{symbol}_{timeframe}.joblib"
        active_path = os.path.join(active_dir, active_name)
        # Copiar el archivo del modelo entrenado al active_path (evita permisos de symlink en Windows)
        try:
            import shutil
            shutil.copy2(model_path, active_path)
            # Guardar métricas junto al activo
            metrics_path = os.path.join(active_dir, f"{symbol}_{timeframe}_metrics.json")
            pd.Series(metrics).to_json(metrics_path)
            logger.info(f"Modelo activado: {active_path}")
        except Exception as e:
            logger.error(f"Error al activar modelo: {str(e)}")