#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para ejecutar backtests de estrategias de trading.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Callable

from ..data.ingest import DataIngestionManager
from ..features.engineering import FeatureEngineer
from ..models.model import ModelManager
from ..strategy.signals import SignalGenerator
from ..strategy.portfolio import PortfolioManager
from ..risk.risk_manager import RiskManager
from ..utils.time_utils import get_trading_days, get_execution_time_str

logger = logging.getLogger(__name__)


class BacktestRunner:
    """Clase para ejecutar backtests de estrategias de trading."""

    def __init__(self, config: Dict[str, Any]):
        """Inicializa el runner de backtests.

        Args:
            config: Configuración del sistema
        """
        self.config = config
        self.data_manager = None
        self.feature_engineer = None
        self.model_manager = None
        self.signal_generator = None
        self.portfolio_manager = None
        self.risk_manager = None
        
        # Configuración de backtest
        self.backtest_config = {
            "start_date": config.get("BACKTEST_START_DATE", ""),
            "end_date": config.get("BACKTEST_END_DATE", ""),
            "initial_capital": float(config.get("BACKTEST_INITIAL_CAPITAL", "100000")),
            "symbols": config.get("SYMBOLS", "").split(","),
            "timeframe": config.get("TIMEFRAME", "1d"),
            "strategy": config.get("STRATEGY", "ml_prediction"),
            "commission": float(config.get("BACKTEST_COMMISSION", "0.001")),  # 0.1% por defecto
            "slippage": float(config.get("BACKTEST_SLIPPAGE", "0.0005")),   # 0.05% por defecto
            "output_dir": config.get("BACKTEST_OUTPUT_DIR", "backtest_results"),
        }
        
        # Estado del backtest
        self.backtest_state = {
            "current_date": None,
            "equity": self.backtest_config["initial_capital"],
            "cash": self.backtest_config["initial_capital"],
            "positions": {},  # symbol -> {qty, entry_price, entry_date}
            "trades": [],     # Lista de operaciones
            "equity_curve": [],  # Lista de {date, equity}
            "signals": [],    # Lista de señales generadas
            "metrics": {},    # Métricas finales
        }
        
        # Crear directorio de salida si no existe
        os.makedirs(self.backtest_config["output_dir"], exist_ok=True)
        
        logger.info(f"BacktestRunner inicializado con {len(self.backtest_config['symbols'])} símbolos")

    def initialize_components(self):
        """Inicializa los componentes necesarios para el backtest."""
        # Inicializar componentes con modo backtest
        self.data_manager = DataIngestionManager(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.model_manager = ModelManager(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.portfolio_manager = PortfolioManager(self.config)
        self.risk_manager = RiskManager(self.config, alpaca_client=None, order_manager=None)

        # Configurar el modo backtest en los componentes
        # self.portfolio_manager.set_backtest_mode(True)
        # self.risk_manager.set_backtest_mode(True)

        logger.info("Componentes de backtest inicializados")

    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Carga los datos históricos para el backtest.

        Returns:
            Dict[str, pd.DataFrame]: Datos históricos por símbolo
        """
        start_date = self.backtest_config["start_date"]
        end_date = self.backtest_config["end_date"]
        symbols = self.backtest_config["symbols"]
        timeframe = self.backtest_config["timeframe"]
        
        logger.info(f"Cargando datos históricos desde {start_date} hasta {end_date} para {len(symbols)} símbolos")
        
        # Cargar datos históricos
        historical_data = {}
        for symbol in symbols:
            try:
                # Obtener datos históricos
                df = self.data_manager.get_historical_data(
                    symbols=[symbol],
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=timeframe
                ).get(symbol)
                
                if df is not None and not df.empty:
                    historical_data[symbol] = df
                    logger.info(f"Datos cargados para {symbol}: {len(df)} barras")
                else:
                    logger.warning(f"No se pudieron cargar datos para {symbol}")
            except Exception as e:
                logger.error(f"Error al cargar datos para {symbol}: {e}")
        
        if not historical_data:
            raise ValueError("No se pudieron cargar datos históricos para ningún símbolo")
        
        return historical_data

    def prepare_features(self, historical_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepara las características para el backtest.

        Args:
            historical_data: Datos históricos por símbolo

        Returns:
            Dict[str, pd.DataFrame]: Datos con características por símbolo
        """
        logger.info("Preparando características para el backtest")

        try:
            # Procesar datos y generar features usando process_data
            feature_data = self.feature_engineer.process_data(historical_data, drop_na=False)

            if feature_data:
                total_features = sum(len(df.columns) for df in feature_data.values())
                logger.info(f"Características generadas para {len(feature_data)} símbolos: {total_features} columnas totales")
            else:
                logger.warning("No se pudieron generar características para ningún símbolo")

        except Exception as e:
            logger.error(f"Error al generar características: {e}")
            feature_data = {}

        return feature_data

    def run_backtest(self) -> Dict[str, Any]:
        """Ejecuta el backtest completo.

        Returns:
            Dict[str, Any]: Resultados del backtest
        """
        import time
        start_time = time.time()
        
        logger.info("Iniciando backtest")
        
        # Inicializar componentes
        self.initialize_components()
        
        # Cargar datos históricos
        historical_data = self.load_data()
        
        # Preparar características
        feature_data = self.prepare_features(historical_data)
        
        # Obtener fechas de trading
        start_date = self.backtest_config["start_date"]
        end_date = self.backtest_config["end_date"]
        trading_days = get_trading_days(start_date, end_date)
        
        logger.info(f"Ejecutando backtest en {len(trading_days)} días de trading")
        
        # Inicializar estado del backtest
        self.backtest_state["current_date"] = trading_days[0]
        self.backtest_state["equity"] = self.backtest_config["initial_capital"]
        self.backtest_state["cash"] = self.backtest_config["initial_capital"]
        self.backtest_state["positions"] = {}
        self.backtest_state["trades"] = []
        self.backtest_state["equity_curve"] = []
        self.backtest_state["signals"] = []
        
        # Iterar por cada día de trading
        for day_idx, current_date in enumerate(trading_days):
            self.backtest_state["current_date"] = current_date
            
            # Actualizar precios de posiciones y equity
            self._update_positions_value(current_date, feature_data)
            
            # Registrar equity curve
            self.backtest_state["equity_curve"].append({
                "date": current_date,
                "equity": self.backtest_state["equity"],
                "cash": self.backtest_state["cash"]
            })
            
            # Generar señales para el día actual
            signals = self._generate_signals(current_date, feature_data)
            
            # Ejecutar operaciones basadas en señales
            if signals:
                self._execute_signals(signals, current_date, feature_data)
            
            # Log de progreso cada 10% o 30 días
            if day_idx % max(1, min(30, len(trading_days) // 10)) == 0:
                progress = (day_idx + 1) / len(trading_days) * 100
                logger.info(f"Progreso del backtest: {progress:.1f}% - Fecha: {current_date} - "
                           f"Equity: ${self.backtest_state['equity']:.2f}")
        
        # Calcular métricas finales
        self._calculate_metrics()
        
        # Guardar resultados
        results_file = self._save_results()
        
        # Generar gráficos
        charts_file = self._generate_charts()
        
        elapsed_time = get_execution_time_str(start_time)
        logger.info(f"Backtest completado en {elapsed_time}")
        logger.info(f"Resultados guardados en {results_file}")
        logger.info(f"Gráficos guardados en {charts_file}")
        
        return self.backtest_state["metrics"]

    def _update_positions_value(self, current_date: datetime.date, 
                              feature_data: Dict[str, pd.DataFrame]) -> None:
        """Actualiza el valor de las posiciones para la fecha actual.

        Args:
            current_date: Fecha actual
            feature_data: Datos con características por símbolo
        """
        total_position_value = 0
        
        # Actualizar valor de cada posición
        for symbol, position in list(self.backtest_state["positions"].items()):
            # Obtener precio actual
            current_price = self._get_price_for_date(symbol, current_date, feature_data)
            
            if current_price is None:
                logger.warning(f"No hay precio disponible para {symbol} en {current_date}, "
                             f"manteniendo precio anterior")
                continue
            
            # Actualizar valor de la posición
            position_value = position["qty"] * current_price
            position["current_price"] = current_price
            position["current_value"] = position_value
            position["unrealized_pnl"] = position_value - (position["qty"] * position["entry_price"])
            
            total_position_value += position_value
        
        # Actualizar equity
        self.backtest_state["equity"] = self.backtest_state["cash"] + total_position_value

    def _generate_signals(self, current_date: datetime.date, 
                         feature_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Genera señales de trading para la fecha actual.

        Args:
            current_date: Fecha actual
            feature_data: Datos con características por símbolo

        Returns:
            List[Dict[str, Any]]: Lista de señales generadas
        """
        signals = []
        
        for symbol, df in feature_data.items():
            try:
                # Filtrar datos hasta la fecha actual (inclusive)
                df_until_date = df[df.index.date <= current_date].copy()
                
                if df_until_date.empty:
                    continue
                
                # Obtener la última fila disponible
                last_row = df_until_date.iloc[-1]
                
                # Verificar si la fecha de la última fila es la fecha actual
                if last_row.name.date() != current_date:
                    continue
                
                # Preparar datos para predicción
                prediction_data = self.feature_engineer.prepare_prediction_data({symbol: df_until_date})
                X = prediction_data.get(symbol)

                if X is None or X.empty:
                    continue
                
                # Hacer predicción con el modelo
                prediction = None
                if self.backtest_config["strategy"] == "ml_prediction":
                    prediction = self.model_manager.predict(X.iloc[-1:], symbol)
                
                # Generar señal
                current_positions = {s: p["qty"] for s, p in self.backtest_state["positions"].items()}
                signal = self.signal_generator.generate_signal(
                    symbol=symbol,
                    data=df_until_date,
                    prediction=prediction,
                    current_positions=current_positions,
                    strategy=self.backtest_config["strategy"]
                )
                
                if signal and signal["direction"] != "hold":
                    # Añadir fecha a la señal
                    signal["date"] = current_date
                    signals.append(signal)
                    
                    # Registrar señal en el historial
                    self.backtest_state["signals"].append(signal)
            
            except Exception as e:
                logger.error(f"Error al generar señal para {symbol} en {current_date}: {e}")
        
        return signals

    def _execute_signals(self, signals: List[Dict[str, Any]], current_date: datetime.date,
                        feature_data: Dict[str, pd.DataFrame]) -> None:
        """Ejecuta las señales de trading generadas.

        Args:
            signals: Lista de señales a ejecutar
            current_date: Fecha actual
            feature_data: Datos con características por símbolo
        """
        # Obtener posiciones actuales
        current_positions = {s: p["qty"] for s, p in self.backtest_state["positions"].items()}
        
        # Calcular tamaños de posición
        position_sizes = self.portfolio_manager.calculate_position_sizes(
            signals=signals,
            equity=self.backtest_state["equity"],
            current_positions=current_positions
        )
        
        # Ejecutar operaciones
        for signal in signals:
            symbol = signal["symbol"]
            direction = signal["direction"]
            
            # Obtener tamaño de posición calculado
            target_position = position_sizes.get(symbol, 0)
            
            # Obtener posición actual
            current_position = self.backtest_state["positions"].get(symbol, {"qty": 0})["qty"]
            
            # Calcular cantidad a operar
            qty_to_trade = target_position - current_position
            
            if qty_to_trade == 0:
                continue
            
            # Verificar dirección de la señal
            if (direction == "buy" and qty_to_trade < 0) or (direction == "sell" and qty_to_trade > 0):
                logger.warning(f"Dirección de señal {direction} no coincide con cantidad a operar {qty_to_trade}")
                continue
            
            # Obtener precio actual
            current_price = self._get_price_for_date(symbol, current_date, feature_data)
            
            if current_price is None:
                logger.warning(f"No hay precio disponible para {symbol} en {current_date}, "
                             f"no se puede ejecutar la operación")
                continue
            
            # Aplicar slippage
            execution_price = current_price * (1 + self.backtest_config["slippage"] * 
                                            (1 if qty_to_trade > 0 else -1))
            
            # Calcular valor de la operación
            trade_value = abs(qty_to_trade) * execution_price
            
            # Calcular comisión
            commission = trade_value * self.backtest_config["commission"]
            
            # Verificar si hay suficiente efectivo para comprar
            if qty_to_trade > 0 and trade_value + commission > self.backtest_state["cash"]:
                # Ajustar cantidad según efectivo disponible
                max_qty = int((self.backtest_state["cash"] - commission) / execution_price)
                if max_qty <= 0:
                    logger.warning(f"No hay suficiente efectivo para comprar {symbol}")
                    continue
                
                qty_to_trade = max_qty
                trade_value = qty_to_trade * execution_price
                commission = trade_value * self.backtest_config["commission"]
            
            # Ejecutar la operación
            if qty_to_trade > 0:  # Compra
                # Actualizar efectivo
                self.backtest_state["cash"] -= (trade_value + commission)
                
                # Actualizar o crear posición
                if symbol in self.backtest_state["positions"]:
                    # Actualizar posición existente (promedio de precio)
                    current_qty = self.backtest_state["positions"][symbol]["qty"]
                    current_price = self.backtest_state["positions"][symbol]["entry_price"]
                    new_qty = current_qty + qty_to_trade
                    new_price = ((current_qty * current_price) + (qty_to_trade * execution_price)) / new_qty
                    
                    self.backtest_state["positions"][symbol]["qty"] = new_qty
                    self.backtest_state["positions"][symbol]["entry_price"] = new_price
                else:
                    # Crear nueva posición
                    self.backtest_state["positions"][symbol] = {
                        "qty": qty_to_trade,
                        "entry_price": execution_price,
                        "entry_date": current_date,
                        "current_price": execution_price,
                        "current_value": qty_to_trade * execution_price,
                        "unrealized_pnl": 0
                    }
            
            else:  # Venta
                qty_to_sell = abs(qty_to_trade)
                
                # Verificar si existe la posición
                if symbol not in self.backtest_state["positions"] or \
                   self.backtest_state["positions"][symbol]["qty"] < qty_to_sell:
                    logger.warning(f"No hay suficientes acciones de {symbol} para vender")
                    continue
                
                # Calcular P&L realizado
                entry_price = self.backtest_state["positions"][symbol]["entry_price"]
                realized_pnl = (execution_price - entry_price) * qty_to_sell - commission
                
                # Actualizar efectivo
                self.backtest_state["cash"] += (trade_value - commission)
                
                # Actualizar posición
                current_qty = self.backtest_state["positions"][symbol]["qty"]
                new_qty = current_qty - qty_to_sell
                
                if new_qty > 0:
                    # Mantener posición con cantidad reducida
                    self.backtest_state["positions"][symbol]["qty"] = new_qty
                    self.backtest_state["positions"][symbol]["current_value"] = new_qty * execution_price
                    self.backtest_state["positions"][symbol]["unrealized_pnl"] = \
                        (execution_price - entry_price) * new_qty
                else:
                    # Eliminar posición
                    del self.backtest_state["positions"][symbol]
            
            # Registrar operación
            trade = {
                "date": current_date,
                "symbol": symbol,
                "side": "buy" if qty_to_trade > 0 else "sell",
                "qty": abs(qty_to_trade),
                "price": execution_price,
                "value": trade_value,
                "commission": commission,
                "signal_confidence": signal.get("confidence", 0),
                "signal_strategy": signal.get("strategy", ""),
            }
            
            if qty_to_trade < 0:  # Solo para ventas
                trade["entry_price"] = entry_price
                trade["realized_pnl"] = realized_pnl
            
            self.backtest_state["trades"].append(trade)
            
            logger.debug(f"Ejecutada operación: {trade['side']} {trade['qty']} {symbol} @ "
                       f"${trade['price']:.2f}")

    def _get_price_for_date(self, symbol: str, date: datetime.date, 
                          feature_data: Dict[str, pd.DataFrame]) -> Optional[float]:
        """Obtiene el precio para un símbolo en una fecha específica.

        Args:
            symbol: Símbolo
            date: Fecha
            feature_data: Datos con características por símbolo

        Returns:
            Optional[float]: Precio o None si no está disponible
        """
        if symbol not in feature_data:
            return None
        
        df = feature_data[symbol]
        
        # Filtrar por fecha
        df_date = df[df.index.date == date]
        
        if df_date.empty:
            return None
        
        # Usar precio de cierre
        return df_date["close"].iloc[-1]

    def _calculate_metrics(self) -> None:
        """Calcula las métricas finales del backtest."""
        # Obtener equity curve como DataFrame
        equity_curve = pd.DataFrame(self.backtest_state["equity_curve"])
        equity_curve.set_index("date", inplace=True)
        
        # Calcular retornos diarios
        equity_curve["daily_return"] = equity_curve["equity"].pct_change()
        
        # Calcular métricas básicas
        initial_equity = self.backtest_config["initial_capital"]
        final_equity = self.backtest_state["equity"]
        total_return = (final_equity / initial_equity) - 1
        
        # Calcular drawdown
        equity_curve["cummax"] = equity_curve["equity"].cummax()
        equity_curve["drawdown"] = (equity_curve["equity"] / equity_curve["cummax"]) - 1
        max_drawdown = equity_curve["drawdown"].min()
        
        # Calcular Sharpe Ratio (anualizado, asumiendo 252 días de trading)
        risk_free_rate = 0.0  # Podría ser configurable
        sharpe_ratio = 0
        sortino_ratio = 0
        
        if len(equity_curve) > 1:
            daily_returns = equity_curve["daily_return"].dropna()
            if len(daily_returns) > 0:
                excess_returns = daily_returns - (risk_free_rate / 252)
                annual_return = ((1 + daily_returns.mean()) ** 252) - 1
                annual_volatility = daily_returns.std() * np.sqrt(252)
                
                if annual_volatility > 0:
                    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
                
                # Sortino Ratio (solo considera volatilidad negativa)
                downside_returns = daily_returns[daily_returns < 0]
                if len(downside_returns) > 0:
                    downside_volatility = downside_returns.std() * np.sqrt(252)
                    if downside_volatility > 0:
                        sortino_ratio = (annual_return - risk_free_rate) / downside_volatility
        
        # Calcular métricas de operaciones
        trades_df = pd.DataFrame(self.backtest_state["trades"])
        
        num_trades = len(trades_df)
        win_rate = 0
        profit_factor = 0
        avg_profit = 0
        avg_loss = 0
        largest_profit = 0
        largest_loss = 0
        avg_holding_days = 0
        
        if not trades_df.empty and "realized_pnl" in trades_df.columns:
            # Filtrar solo operaciones cerradas (ventas)
            closed_trades = trades_df[trades_df["side"] == "sell"]
            
            if not closed_trades.empty:
                # Calcular métricas de operaciones
                winning_trades = closed_trades[closed_trades["realized_pnl"] > 0]
                losing_trades = closed_trades[closed_trades["realized_pnl"] <= 0]
                
                num_winning = len(winning_trades)
                num_losing = len(losing_trades)
                
                win_rate = num_winning / len(closed_trades) if len(closed_trades) > 0 else 0
                
                total_profit = winning_trades["realized_pnl"].sum() if not winning_trades.empty else 0
                total_loss = abs(losing_trades["realized_pnl"].sum()) if not losing_trades.empty else 0
                
                profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
                
                avg_profit = winning_trades["realized_pnl"].mean() if not winning_trades.empty else 0
                avg_loss = losing_trades["realized_pnl"].mean() if not losing_trades.empty else 0
                
                largest_profit = winning_trades["realized_pnl"].max() if not winning_trades.empty else 0
                largest_loss = losing_trades["realized_pnl"].min() if not losing_trades.empty else 0
        
        # Guardar métricas
        self.backtest_state["metrics"] = {
            "initial_equity": initial_equity,
            "final_equity": final_equity,
            "total_return": total_return,
            "total_return_pct": total_return * 100,
            "annualized_return": ((1 + total_return) ** (252 / len(equity_curve))) - 1 if len(equity_curve) > 0 else 0,
            "max_drawdown": max_drawdown,
            "max_drawdown_pct": max_drawdown * 100,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "num_trades": num_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "largest_profit": largest_profit,
            "largest_loss": largest_loss,
            "avg_holding_days": avg_holding_days,
            "backtest_period": f"{self.backtest_config['start_date']} to {self.backtest_config['end_date']}",
            "backtest_days": len(equity_curve),
            "symbols": self.backtest_config["symbols"],
            "strategy": self.backtest_config["strategy"],
            "timeframe": self.backtest_config["timeframe"],
        }

    def _save_results(self) -> str:
        """Guarda los resultados del backtest.

        Returns:
            str: Ruta del archivo de resultados
        """
        # Crear nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy = self.backtest_config["strategy"]
        symbols_str = "_".join(self.backtest_config["symbols"][:3])  # Primeros 3 símbolos
        
        if len(self.backtest_config["symbols"]) > 3:
            symbols_str += f"_and_{len(self.backtest_config['symbols']) - 3}_more"
        
        filename = f"backtest_{strategy}_{symbols_str}_{timestamp}.json"
        filepath = os.path.join(self.backtest_config["output_dir"], filename)
        
        # Preparar datos para guardar
        results = {
            "config": self.backtest_config,
            "metrics": self.backtest_state["metrics"],
            "equity_curve": [
                {"date": ec["date"].isoformat(), "equity": ec["equity"], "cash": ec["cash"]}
                for ec in self.backtest_state["equity_curve"]
            ],
            "trades": [
                {**trade, "date": trade["date"].isoformat()}
                for trade in self.backtest_state["trades"]
            ],
            "signals": [
                {**signal, "date": signal["date"].isoformat()}
                for signal in self.backtest_state["signals"]
            ],
        }
        
        # Guardar como JSON
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
        
        return filepath

    def _generate_charts(self) -> str:
        """Genera gráficos del backtest.

        Returns:
            str: Ruta del archivo de gráficos
        """
        # Configurar estilo de gráficos
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("viridis")
        
        # Crear nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy = self.backtest_config["strategy"]
        symbols_str = "_".join(self.backtest_config["symbols"][:3])  # Primeros 3 símbolos
        
        if len(self.backtest_config["symbols"]) > 3:
            symbols_str += f"_and_{len(self.backtest_config['symbols']) - 3}_more"
        
        filename = f"backtest_{strategy}_{symbols_str}_{timestamp}.png"
        filepath = os.path.join(self.backtest_config["output_dir"], filename)
        
        # Convertir equity curve a DataFrame
        equity_curve = pd.DataFrame(self.backtest_state["equity_curve"])
        equity_curve.set_index("date", inplace=True)
        
        # Calcular drawdown
        equity_curve["cummax"] = equity_curve["equity"].cummax()
        equity_curve["drawdown"] = (equity_curve["equity"] / equity_curve["cummax"]) - 1
        
        # Crear figura con subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 18), gridspec_kw={'height_ratios': [3, 1, 2]})
        
        # Plot 1: Equity Curve
        axes[0].plot(equity_curve.index, equity_curve["equity"], label="Equity", linewidth=2)
        axes[0].set_title("Equity Curve", fontsize=14)
        axes[0].set_ylabel("Equity ($)", fontsize=12)
        axes[0].grid(True)
        axes[0].legend()
        
        # Añadir anotaciones de métricas
        metrics = self.backtest_state["metrics"]
        metrics_text = (
            f"Total Return: {metrics['total_return_pct']:.2f}%\n"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
            f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%\n"
            f"Win Rate: {metrics['win_rate']*100:.1f}%"
        )
        axes[0].annotate(metrics_text, xy=(0.02, 0.95), xycoords='axes fraction',
                       fontsize=10, va='top', bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        # Plot 2: Drawdown
        axes[1].fill_between(equity_curve.index, equity_curve["drawdown"] * 100, 0, 
                          color='red', alpha=0.3)
        axes[1].set_title("Drawdown (%)", fontsize=14)
        axes[1].set_ylabel("Drawdown (%)", fontsize=12)
        axes[1].grid(True)
        
        # Plot 3: Trades
        if self.backtest_state["trades"]:
            trades_df = pd.DataFrame(self.backtest_state["trades"])
            trades_df.set_index("date", inplace=True)
            
            # Filtrar solo operaciones cerradas (ventas)
            sell_trades = trades_df[trades_df["side"] == "sell"]
            
            if not sell_trades.empty and "realized_pnl" in sell_trades.columns:
                # Crear gráfico de barras de P&L por operación
                colors = ['green' if pnl > 0 else 'red' for pnl in sell_trades["realized_pnl"]]
                axes[2].bar(range(len(sell_trades)), sell_trades["realized_pnl"], color=colors)
                axes[2].set_title("P&L por Operación", fontsize=14)
                axes[2].set_xlabel("Operación #", fontsize=12)
                axes[2].set_ylabel("P&L ($)", fontsize=12)
                axes[2].grid(True)
                
                # Añadir línea de P&L acumulado
                cumulative_pnl = sell_trades["realized_pnl"].cumsum()
                ax2 = axes[2].twinx()
                ax2.plot(range(len(sell_trades)), cumulative_pnl, color='blue', linewidth=2)
                ax2.set_ylabel("P&L Acumulado ($)", fontsize=12, color='blue')
            else:
                axes[2].text(0.5, 0.5, "No hay operaciones cerradas para mostrar", 
                           ha='center', va='center', fontsize=12)
        else:
            axes[2].text(0.5, 0.5, "No hay operaciones para mostrar", 
                       ha='center', va='center', fontsize=12)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Añadir título general
        fig.suptitle(
            f"Backtest: {strategy} - {self.backtest_config['start_date']} a {self.backtest_config['end_date']}",
            fontsize=16, y=0.99
        )
        
        # Guardar figura
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath

    def load_results(self, results_file: str) -> Dict[str, Any]:
        """Carga resultados de un backtest anterior.

        Args:
            results_file: Ruta del archivo de resultados

        Returns:
            Dict[str, Any]: Resultados del backtest
        """
        try:
            with open(results_file, "r") as f:
                results = json.load(f)
            
            logger.info(f"Resultados cargados desde {results_file}")
            return results
        except Exception as e:
            logger.error(f"Error al cargar resultados: {e}")
            return {}

    def compare_strategies(self, strategies: List[str], 
                         start_date: str, end_date: str) -> Dict[str, Any]:
        """Compara múltiples estrategias en el mismo período.

        Args:
            strategies: Lista de estrategias a comparar
            start_date: Fecha de inicio
            end_date: Fecha de fin

        Returns:
            Dict[str, Any]: Resultados de la comparación
        """
        results = {}
        
        for strategy in strategies:
            # Configurar estrategia
            self.backtest_config["strategy"] = strategy
            self.backtest_config["start_date"] = start_date
            self.backtest_config["end_date"] = end_date
            
            # Ejecutar backtest
            logger.info(f"Ejecutando backtest para estrategia: {strategy}")
            metrics = self.run_backtest()
            
            # Guardar resultados
            results[strategy] = metrics
        
        # Generar comparativa
        self._generate_comparison_chart(results)
        
        return results

    def _generate_comparison_chart(self, results: Dict[str, Dict[str, Any]]) -> str:
        """Genera un gráfico comparativo de estrategias.

        Args:
            results: Resultados por estrategia

        Returns:
            str: Ruta del archivo de gráfico
        """
        # Configurar estilo de gráficos
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("viridis")
        
        # Crear nombre de archivo con timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"strategy_comparison_{timestamp}.png"
        filepath = os.path.join(self.backtest_config["output_dir"], filename)
        
        # Crear figura
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extraer métricas para comparación
        strategies = list(results.keys())
        returns = [results[s]["total_return_pct"] for s in strategies]
        sharpe = [results[s]["sharpe_ratio"] for s in strategies]
        drawdowns = [results[s]["max_drawdown_pct"] for s in strategies]
        win_rates = [results[s]["win_rate"] * 100 for s in strategies]
        
        # Plot 1: Retornos totales
        axes[0, 0].bar(strategies, returns, color='green')
        axes[0, 0].set_title("Retorno Total (%)", fontsize=14)
        axes[0, 0].set_ylabel("Retorno (%)", fontsize=12)
        axes[0, 0].grid(True)
        
        # Plot 2: Sharpe Ratio
        axes[0, 1].bar(strategies, sharpe, color='blue')
        axes[0, 1].set_title("Sharpe Ratio", fontsize=14)
        axes[0, 1].set_ylabel("Sharpe", fontsize=12)
        axes[0, 1].grid(True)
        
        # Plot 3: Max Drawdown
        axes[1, 0].bar(strategies, drawdowns, color='red')
        axes[1, 0].set_title("Máximo Drawdown (%)", fontsize=14)
        axes[1, 0].set_ylabel("Drawdown (%)", fontsize=12)
        axes[1, 0].grid(True)
        
        # Plot 4: Win Rate
        axes[1, 1].bar(strategies, win_rates, color='purple')
        axes[1, 1].set_title("Win Rate (%)", fontsize=14)
        axes[1, 1].set_ylabel("Win Rate (%)", fontsize=12)
        axes[1, 1].grid(True)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Añadir título general
        fig.suptitle(
            f"Comparación de Estrategias: {self.backtest_config['start_date']} a {self.backtest_config['end_date']}",
            fontsize=16, y=0.99
        )
        
        # Guardar figura
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico comparativo guardado en {filepath}")
        
        return filepath