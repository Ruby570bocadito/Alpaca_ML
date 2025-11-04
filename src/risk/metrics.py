#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para la monitorización y métricas del sistema de trading.
"""

import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import threading
import os

try:
    from prometheus_client import Counter, Gauge, Histogram, Summary, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Importaciones internas
from execution.alpaca_client import AlpacaClient
from risk.risk_manager import RiskManager

logger = logging.getLogger(__name__)


class MetricsManager:
    """Clase para gestionar métricas y monitorización del sistema de trading."""

    def __init__(self, config: Dict[str, Any], alpaca_client: AlpacaClient, risk_manager: Optional[RiskManager] = None):
        """Inicializa el gestor de métricas.

        Args:
            config: Configuración del sistema
            alpaca_client: Cliente de Alpaca
            risk_manager: Gestor de riesgo (opcional)
        """
        self.config = config
        self.alpaca_client = alpaca_client
        self.risk_manager = risk_manager
        
        # Configuración de métricas
        self.metrics_config = {
            "enabled": config.get("METRICS_ENABLED", "true").lower() == "true",
            "prometheus_enabled": config.get("PROMETHEUS_ENABLED", "false").lower() == "true",
            "prometheus_port": int(config.get("PROMETHEUS_PORT", "8000")),
            "log_metrics_interval_seconds": int(config.get("LOG_METRICS_INTERVAL_SECONDS", "60")),
            "metrics_history_size": int(config.get("METRICS_HISTORY_SIZE", "1000")),
            "metrics_export_path": config.get("METRICS_EXPORT_PATH", "data/metrics"),
        }
        
        # Estado de métricas
        self.metrics_state = {
            "start_time": datetime.now(),
            "last_metrics_log": datetime.now(),
            "trade_count": 0,
            "order_count": 0,
            "filled_order_count": 0,
            "canceled_order_count": 0,
            "rejected_order_count": 0,
            "signal_count": 0,
            "win_count": 0,
            "loss_count": 0,
            "total_pnl": 0.0,
            "total_fees": 0.0,
            "max_drawdown": 0.0,
            "current_drawdown": 0.0,
            "peak_equity": 0.0,
            "initial_equity": 0.0,
            "current_equity": 0.0,
            "model_prediction_count": 0,
            "model_accuracy": 0.0,
            "latency_ms": {},
            "error_count": 0,
        }
        
        # Historial de métricas
        self.metrics_history = {
            "timestamp": [],
            "equity": [],
            "cash": [],
            "position_value": [],
            "pnl_daily": [],
            "trade_count_daily": [],
            "win_rate": [],
            "drawdown": [],
            "sharpe_ratio": [],
            "model_accuracy": [],
            "latency_avg_ms": [],
        }
        
        # Métricas de Prometheus
        self.prometheus_metrics = {}
        
        # Inicializar
        self._initialize_metrics()
        
        # Iniciar servidor Prometheus si está habilitado
        if self.metrics_config["enabled"] and self.metrics_config["prometheus_enabled"] and PROMETHEUS_AVAILABLE:
            self._start_prometheus_server()
            
        # Iniciar hilo de logging de métricas
        if self.metrics_config["enabled"] and self.metrics_config["log_metrics_interval_seconds"] > 0:
            self._start_metrics_logging_thread()
        
        logger.info("Gestor de métricas inicializado")

    def _initialize_metrics(self):
        """Inicializa las métricas."""
        try:
            # Obtener información de la cuenta
            account = self.alpaca_client.get_account()
            
            if account and "equity" in account:
                equity = float(account["equity"])
                self.metrics_state["initial_equity"] = equity
                self.metrics_state["current_equity"] = equity
                self.metrics_state["peak_equity"] = equity
                
                logger.info(f"Métricas inicializadas con equity: {equity}")
            else:
                logger.error("No se pudo obtener información de la cuenta para métricas")
            
            # Inicializar métricas de Prometheus
            if self.metrics_config["prometheus_enabled"] and PROMETHEUS_AVAILABLE:
                self._initialize_prometheus_metrics()
                
        except Exception as e:
            logger.error(f"Error al inicializar métricas: {e}", exc_info=True)

    def _initialize_prometheus_metrics(self):
        """Inicializa métricas de Prometheus."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus no está disponible. Instale prometheus_client para habilitar métricas.")
            return
        
        try:
            # Contadores
            self.prometheus_metrics["trade_count"] = Counter(
                "trading_bot_trade_count", "Número total de operaciones realizadas")
            self.prometheus_metrics["order_count"] = Counter(
                "trading_bot_order_count", "Número total de órdenes enviadas")
            self.prometheus_metrics["filled_order_count"] = Counter(
                "trading_bot_filled_order_count", "Número total de órdenes ejecutadas")
            self.prometheus_metrics["canceled_order_count"] = Counter(
                "trading_bot_canceled_order_count", "Número total de órdenes canceladas")
            self.prometheus_metrics["rejected_order_count"] = Counter(
                "trading_bot_rejected_order_count", "Número total de órdenes rechazadas")
            self.prometheus_metrics["signal_count"] = Counter(
                "trading_bot_signal_count", "Número total de señales generadas")
            self.prometheus_metrics["win_count"] = Counter(
                "trading_bot_win_count", "Número total de operaciones ganadoras")
            self.prometheus_metrics["loss_count"] = Counter(
                "trading_bot_loss_count", "Número total de operaciones perdedoras")
            self.prometheus_metrics["error_count"] = Counter(
                "trading_bot_error_count", "Número total de errores")
            self.prometheus_metrics["model_prediction_count"] = Counter(
                "trading_bot_model_prediction_count", "Número total de predicciones del modelo")
            
            # Gauges
            self.prometheus_metrics["equity"] = Gauge(
                "trading_bot_equity", "Valor actual del equity")
            self.prometheus_metrics["cash"] = Gauge(
                "trading_bot_cash", "Valor actual del efectivo disponible")
            self.prometheus_metrics["position_value"] = Gauge(
                "trading_bot_position_value", "Valor actual de las posiciones")
            self.prometheus_metrics["pnl_daily"] = Gauge(
                "trading_bot_pnl_daily", "P&L diario")
            self.prometheus_metrics["win_rate"] = Gauge(
                "trading_bot_win_rate", "Tasa de acierto (win rate)")
            self.prometheus_metrics["drawdown"] = Gauge(
                "trading_bot_drawdown", "Drawdown actual en porcentaje")
            self.prometheus_metrics["model_accuracy"] = Gauge(
                "trading_bot_model_accuracy", "Precisión del modelo")
            
            # Histogramas
            self.prometheus_metrics["latency_ms"] = Histogram(
                "trading_bot_latency_ms", "Latencia en milisegundos", 
                ['operation'], buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000])
            
            logger.info("Métricas de Prometheus inicializadas")
            
        except Exception as e:
            logger.error(f"Error al inicializar métricas de Prometheus: {e}", exc_info=True)

    def _start_prometheus_server(self):
        """Inicia el servidor HTTP de Prometheus."""
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus no está disponible. No se puede iniciar el servidor.")
            return
        
        try:
            port = self.metrics_config["prometheus_port"]
            start_http_server(port)
            logger.info(f"Servidor de métricas Prometheus iniciado en puerto {port}")
        except Exception as e:
            logger.error(f"Error al iniciar servidor Prometheus: {e}", exc_info=True)

    def _start_metrics_logging_thread(self):
        """Inicia un hilo para logging periódico de métricas."""
        def log_metrics_periodically():
            while True:
                try:
                    interval = self.metrics_config["log_metrics_interval_seconds"]
                    time.sleep(interval)
                    self.log_current_metrics()
                except Exception as e:
                    logger.error(f"Error en hilo de logging de métricas: {e}", exc_info=True)
        
        metrics_thread = threading.Thread(target=log_metrics_periodically, daemon=True)
        metrics_thread.start()
        logger.info(f"Hilo de logging de métricas iniciado (intervalo: {self.metrics_config['log_metrics_interval_seconds']}s)")

    def update_metrics(self):
        """Actualiza las métricas con información actual."""
        try:
            # Obtener información de la cuenta
            account = self.alpaca_client.get_account()
            
            if account and "equity" in account:
                current_equity = float(account["equity"])
                cash = float(account["cash"])
                previous_equity = self.metrics_state["current_equity"]
                
                self.metrics_state["current_equity"] = current_equity
                
                # Actualizar pico de equity
                if current_equity > self.metrics_state["peak_equity"]:
                    self.metrics_state["peak_equity"] = current_equity
                
                # Calcular drawdown
                if self.metrics_state["peak_equity"] > 0:
                    current_drawdown = (self.metrics_state["peak_equity"] - current_equity) / self.metrics_state["peak_equity"] * 100
                    self.metrics_state["current_drawdown"] = current_drawdown
                    
                    if current_drawdown > self.metrics_state["max_drawdown"]:
                        self.metrics_state["max_drawdown"] = current_drawdown
                
                # Calcular win rate
                total_trades = self.metrics_state["win_count"] + self.metrics_state["loss_count"]
                win_rate = self.metrics_state["win_count"] / total_trades * 100 if total_trades > 0 else 0
                
                # Obtener posiciones
                positions = self.alpaca_client.get_positions()
                position_value = sum(float(p["market_value"]) for p in positions)
                
                # Actualizar historial de métricas
                timestamp = datetime.now()
                
                # Limitar tamaño del historial
                max_size = self.metrics_config["metrics_history_size"]
                if len(self.metrics_history["timestamp"]) >= max_size:
                    for key in self.metrics_history:
                        self.metrics_history[key] = self.metrics_history[key][-max_size+1:]
                
                # Añadir nuevos valores
                self.metrics_history["timestamp"].append(timestamp)
                self.metrics_history["equity"].append(current_equity)
                self.metrics_history["cash"].append(cash)
                self.metrics_history["position_value"].append(position_value)
                
                # Calcular P&L diario (diferencia desde el inicio del día)
                if self.risk_manager:
                    daily_starting_equity = self.risk_manager.risk_state.get("daily_starting_equity", self.metrics_state["initial_equity"])
                    pnl_daily = current_equity - daily_starting_equity
                else:
                    # Aproximación si no hay risk manager
                    pnl_daily = current_equity - previous_equity
                
                self.metrics_history["pnl_daily"].append(pnl_daily)
                self.metrics_history["trade_count_daily"].append(self.metrics_state.get("trade_count", 0))
                self.metrics_history["win_rate"].append(win_rate)
                self.metrics_history["drawdown"].append(self.metrics_state["current_drawdown"])
                
                # Calcular Sharpe ratio (simplificado)
                if len(self.metrics_history["equity"]) > 1:
                    returns = np.diff(self.metrics_history["equity"]) / self.metrics_history["equity"][:-1]
                    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                else:
                    sharpe = 0
                
                self.metrics_history["sharpe_ratio"].append(sharpe)
                self.metrics_history["model_accuracy"].append(self.metrics_state["model_accuracy"])
                
                # Calcular latencia promedio
                latency_values = list(self.metrics_state["latency_ms"].values())
                avg_latency = sum(latency_values) / len(latency_values) if latency_values else 0
                self.metrics_history["latency_avg_ms"].append(avg_latency)
                
                # Actualizar métricas de Prometheus
                if self.metrics_config["prometheus_enabled"] and PROMETHEUS_AVAILABLE:
                    self._update_prometheus_metrics(current_equity, cash, position_value, pnl_daily, win_rate)
                
                logger.debug(f"Métricas actualizadas: equity={current_equity}, drawdown={self.metrics_state['current_drawdown']:.2f}%")
                
                return True
            else:
                logger.error("No se pudo obtener información de la cuenta para métricas")
                return False
                
        except Exception as e:
            logger.error(f"Error al actualizar métricas: {e}", exc_info=True)
            self.metrics_state["error_count"] += 1
            
            if self.metrics_config["prometheus_enabled"] and PROMETHEUS_AVAILABLE and "error_count" in self.prometheus_metrics:
                self.prometheus_metrics["error_count"].inc()
                
            return False

    def _update_prometheus_metrics(self, equity, cash, position_value, pnl_daily, win_rate):
        """Actualiza métricas de Prometheus.

        Args:
            equity: Valor actual del equity
            cash: Valor actual del efectivo disponible
            position_value: Valor actual de las posiciones
            pnl_daily: P&L diario
            win_rate: Tasa de acierto
        """
        if not PROMETHEUS_AVAILABLE:
            return
        
        try:
            self.prometheus_metrics["equity"].set(equity)
            self.prometheus_metrics["cash"].set(cash)
            self.prometheus_metrics["position_value"].set(position_value)
            self.prometheus_metrics["pnl_daily"].set(pnl_daily)
            self.prometheus_metrics["win_rate"].set(win_rate)
            self.prometheus_metrics["drawdown"].set(self.metrics_state["current_drawdown"])
            self.prometheus_metrics["model_accuracy"].set(self.metrics_state["model_accuracy"])
        except Exception as e:
            logger.error(f"Error al actualizar métricas de Prometheus: {e}", exc_info=True)

    def log_current_metrics(self):
        """Registra las métricas actuales en el log."""
        try:
            # Actualizar métricas primero
            self.update_metrics()
            
            # Registrar métricas principales
            uptime = datetime.now() - self.metrics_state["start_time"]
            uptime_str = str(uptime).split('.')[0]  # Eliminar microsegundos
            
            metrics_log = {
                "timestamp": datetime.now().isoformat(),
                "uptime": uptime_str,
                "equity": self.metrics_state["current_equity"],
                "drawdown_pct": self.metrics_state["current_drawdown"],
                "max_drawdown_pct": self.metrics_state["max_drawdown"],
                "trade_count": self.metrics_state["trade_count"],
                "win_count": self.metrics_state["win_count"],
                "loss_count": self.metrics_state["loss_count"],
                "win_rate": self.metrics_state["win_count"] / (self.metrics_state["win_count"] + self.metrics_state["loss_count"]) * 100 
                            if (self.metrics_state["win_count"] + self.metrics_state["loss_count"]) > 0 else 0,
                "total_pnl": self.metrics_state["total_pnl"],
                "model_accuracy": self.metrics_state["model_accuracy"],
            }
            
            # Añadir métricas de riesgo si está disponible
            if self.risk_manager:
                risk_metrics = self.risk_manager.get_risk_metrics()
                metrics_log["daily_loss_pct"] = risk_metrics.get("daily_loss_pct", 0)
                metrics_log["total_positions"] = risk_metrics.get("total_positions", 0)
                metrics_log["total_exposure_pct"] = risk_metrics.get("total_exposure_pct", 0)
            
            logger.info(f"MÉTRICAS: {metrics_log}")
            self.metrics_state["last_metrics_log"] = datetime.now()
            
            return metrics_log
            
        except Exception as e:
            logger.error(f"Error al registrar métricas: {e}", exc_info=True)
            return None

    def record_trade(self, trade_data: Dict[str, Any]):
        """Registra una operación completada.

        Args:
            trade_data: Datos de la operación
        """
        try:
            self.metrics_state["trade_count"] += 1
            
            # Determinar si es ganadora o perdedora
            pnl = trade_data.get("pnl", 0)
            if pnl > 0:
                self.metrics_state["win_count"] += 1
            elif pnl < 0:
                self.metrics_state["loss_count"] += 1
            
            # Actualizar P&L total
            self.metrics_state["total_pnl"] += pnl
            
            # Actualizar comisiones
            fees = trade_data.get("fees", 0)
            self.metrics_state["total_fees"] += fees
            
            # Actualizar métricas de Prometheus
            if self.metrics_config["prometheus_enabled"] and PROMETHEUS_AVAILABLE:
                self.prometheus_metrics["trade_count"].inc()
                
                if pnl > 0:
                    self.prometheus_metrics["win_count"].inc()
                elif pnl < 0:
                    self.prometheus_metrics["loss_count"].inc()
            
            logger.info(f"Operación registrada: {trade_data['symbol']} {trade_data.get('side', '')} "
                      f"PnL: {pnl:.2f} Fees: {fees:.2f}")
            
        except Exception as e:
            logger.error(f"Error al registrar operación: {e}", exc_info=True)

    def record_order(self, order_data: Dict[str, Any]):
        """Registra una orden.

        Args:
            order_data: Datos de la orden
        """
        try:
            self.metrics_state["order_count"] += 1
            
            # Actualizar contadores según estado
            status = order_data.get("status", "")
            
            if status in ["filled", "partially_filled"]:
                self.metrics_state["filled_order_count"] += 1
            elif status in ["canceled", "expired"]:
                self.metrics_state["canceled_order_count"] += 1
            elif status in ["rejected"]:
                self.metrics_state["rejected_order_count"] += 1
            
            # Actualizar métricas de Prometheus
            if self.metrics_config["prometheus_enabled"] and PROMETHEUS_AVAILABLE:
                self.prometheus_metrics["order_count"].inc()
                
                if status in ["filled", "partially_filled"]:
                    self.prometheus_metrics["filled_order_count"].inc()
                elif status in ["canceled", "expired"]:
                    self.prometheus_metrics["canceled_order_count"].inc()
                elif status in ["rejected"]:
                    self.prometheus_metrics["rejected_order_count"].inc()
            
            logger.debug(f"Orden registrada: {order_data.get('symbol', '')} {order_data.get('side', '')} "
                       f"Status: {status}")
            
        except Exception as e:
            logger.error(f"Error al registrar orden: {e}", exc_info=True)

    def record_signal(self, signal_data: Dict[str, Any]):
        """Registra una señal generada.

        Args:
            signal_data: Datos de la señal
        """
        try:
            self.metrics_state["signal_count"] += 1
            
            # Actualizar métricas de Prometheus
            if self.metrics_config["prometheus_enabled"] and PROMETHEUS_AVAILABLE:
                self.prometheus_metrics["signal_count"].inc()
            
            logger.debug(f"Señal registrada: {signal_data.get('symbol', '')} {signal_data.get('direction', '')} "
                       f"Confianza: {signal_data.get('confidence', 0):.2f}")
            
        except Exception as e:
            logger.error(f"Error al registrar señal: {e}", exc_info=True)

    def record_model_prediction(self, prediction_data: Dict[str, Any]):
        """Registra una predicción del modelo.

        Args:
            prediction_data: Datos de la predicción
        """
        try:
            self.metrics_state["model_prediction_count"] += 1
            
            # Actualizar precisión del modelo si hay ground truth
            if "actual" in prediction_data and "predicted" in prediction_data:
                actual = prediction_data["actual"]
                predicted = prediction_data["predicted"]
                
                # Actualizar precisión (media móvil)
                alpha = 0.05  # Factor de suavizado
                current_accuracy = self.metrics_state["model_accuracy"]
                new_accuracy = 1.0 if actual == predicted else 0.0
                
                self.metrics_state["model_accuracy"] = current_accuracy * (1 - alpha) + new_accuracy * alpha
            
            # Actualizar métricas de Prometheus
            if self.metrics_config["prometheus_enabled"] and PROMETHEUS_AVAILABLE:
                self.prometheus_metrics["model_prediction_count"].inc()
                self.prometheus_metrics["model_accuracy"].set(self.metrics_state["model_accuracy"])
            
            logger.debug(f"Predicción registrada: {prediction_data.get('symbol', '')} "
                       f"Valor: {prediction_data.get('predicted', '')}")
            
        except Exception as e:
            logger.error(f"Error al registrar predicción: {e}", exc_info=True)

    def record_latency(self, operation: str, latency_ms: float):
        """Registra la latencia de una operación.

        Args:
            operation: Nombre de la operación
            latency_ms: Latencia en milisegundos
        """
        try:
            self.metrics_state["latency_ms"][operation] = latency_ms
            
            # Actualizar métricas de Prometheus
            if self.metrics_config["prometheus_enabled"] and PROMETHEUS_AVAILABLE:
                self.prometheus_metrics["latency_ms"].labels(operation=operation).observe(latency_ms)
            
            logger.debug(f"Latencia registrada: {operation} {latency_ms:.2f}ms")
            
        except Exception as e:
            logger.error(f"Error al registrar latencia: {e}", exc_info=True)

    def record_error(self, error_data: Dict[str, Any]):
        """Registra un error.

        Args:
            error_data: Datos del error
        """
        try:
            self.metrics_state["error_count"] += 1
            
            # Actualizar métricas de Prometheus
            if self.metrics_config["prometheus_enabled"] and PROMETHEUS_AVAILABLE:
                self.prometheus_metrics["error_count"].inc()
            
            logger.error(f"Error registrado: {error_data.get('message', 'Error desconocido')} "
                       f"Tipo: {error_data.get('type', 'N/A')} "
                       f"Componente: {error_data.get('component', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Error al registrar error: {e}", exc_info=True)

    def export_metrics_to_csv(self, filename: Optional[str] = None):
        """Exporta métricas a un archivo CSV.

        Args:
            filename: Nombre del archivo (opcional)

        Returns:
            str: Ruta del archivo exportado
        """
        try:
            # Crear directorio si no existe
            os.makedirs(self.metrics_config["metrics_export_path"], exist_ok=True)
            
            # Generar nombre de archivo si no se proporciona
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"metrics_{timestamp}.csv"
            
            # Crear DataFrame
            df = pd.DataFrame(self.metrics_history)
            
            # Guardar a CSV
            filepath = os.path.join(self.metrics_config["metrics_export_path"], filename)
            df.to_csv(filepath, index=False)
            
            logger.info(f"Métricas exportadas a {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error al exportar métricas: {e}", exc_info=True)
            return None

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calcula y devuelve métricas de rendimiento.

        Returns:
            Dict[str, Any]: Métricas de rendimiento
        """
        try:
            # Actualizar métricas primero
            self.update_metrics()
            
            # Calcular métricas de rendimiento
            total_trades = self.metrics_state["win_count"] + self.metrics_state["loss_count"]
            win_rate = self.metrics_state["win_count"] / total_trades * 100 if total_trades > 0 else 0
            
            # Calcular retorno total
            initial_equity = self.metrics_state["initial_equity"]
            current_equity = self.metrics_state["current_equity"]
            total_return_pct = (current_equity - initial_equity) / initial_equity * 100 if initial_equity > 0 else 0
            
            # Calcular Sharpe ratio
            if len(self.metrics_history["equity"]) > 1:
                returns = np.diff(self.metrics_history["equity"]) / self.metrics_history["equity"][:-1]
                sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
                
                # Calcular Sortino ratio (solo retornos negativos)
                neg_returns = returns[returns < 0]
                sortino = returns.mean() / neg_returns.std() * np.sqrt(252) if len(neg_returns) > 0 and neg_returns.std() > 0 else 0
            else:
                sharpe = 0
                sortino = 0
            
            # Calcular profit factor
            total_wins = sum(trade["pnl"] for trade in self.metrics_history.get("trades", []) if trade["pnl"] > 0)
            total_losses = sum(abs(trade["pnl"]) for trade in self.metrics_history.get("trades", []) if trade["pnl"] < 0)
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
            
            # Calcular expectancy
            avg_win = total_wins / self.metrics_state["win_count"] if self.metrics_state["win_count"] > 0 else 0
            avg_loss = total_losses / self.metrics_state["loss_count"] if self.metrics_state["loss_count"] > 0 else 0
            expectancy = (win_rate/100 * avg_win) - ((1-win_rate/100) * avg_loss) if avg_loss > 0 else 0
            
            return {
                "timestamp": datetime.now().isoformat(),
                "total_trades": total_trades,
                "win_count": self.metrics_state["win_count"],
                "loss_count": self.metrics_state["loss_count"],
                "win_rate": win_rate,
                "total_pnl": self.metrics_state["total_pnl"],
                "total_fees": self.metrics_state["total_fees"],
                "net_pnl": self.metrics_state["total_pnl"] - self.metrics_state["total_fees"],
                "initial_equity": initial_equity,
                "current_equity": current_equity,
                "total_return_pct": total_return_pct,
                "max_drawdown": self.metrics_state["max_drawdown"],
                "current_drawdown": self.metrics_state["current_drawdown"],
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "profit_factor": profit_factor,
                "expectancy": expectancy,
                "model_accuracy": self.metrics_state["model_accuracy"],
                "avg_latency_ms": sum(self.metrics_state["latency_ms"].values()) / len(self.metrics_state["latency_ms"]) 
                                  if self.metrics_state["latency_ms"] else 0,
            }
            
        except Exception as e:
            logger.error(f"Error al calcular métricas de rendimiento: {e}", exc_info=True)
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
            }

    def save_metrics(self):
        """Guarda las métricas de rendimiento en un archivo."""
        if self.metrics_config["enabled"] and self.metrics_config.get("metrics_export_path"):
            self.export_metrics_to_csv()
        else:
            logger.info("La exportación de métricas no está habilitada o no se ha configurado una ruta.")