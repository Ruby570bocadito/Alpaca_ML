#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ejemplo de uso del sistema de backtesting
"""

import os
import sys
import logging
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt

# Añadir el directorio raíz al path para importar los módulos del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.backtest.runner import BacktestRunner
from src.utils.logging import setup_logging

# Configurar logging
config = {
    "LOG_LEVEL": "INFO",
    "LOG_TO_CONSOLE": "true",
    "LOG_TO_FILE": "true",
    "LOG_DIR": "logs",
    "LOG_FILE": "trading_bot.log"
}
setup_logging(config)
logger = logging.getLogger(__name__)

def main():
    # Configuración del backtest
    config = {
        # Configuración de Alpaca (no se usa en backtest pero es requerido por los componentes)
        "ALPACA_API_KEY": "YOUR_API_KEY",  # No se usa en backtest
        "ALPACA_API_SECRET": "YOUR_API_SECRET",  # No se usa en backtest
        "ALPACA_BASE_URL": "https://paper-api.alpaca.markets",  # No se usa en backtest
        
        # Configuración de backtest
        "BACKTEST_START_DATE": "2022-01-01",
        "BACKTEST_END_DATE": "2022-12-31",
        "BACKTEST_INITIAL_CAPITAL": "100000",
        "BACKTEST_COMMISSION": "0.001",  # 0.1%
        "BACKTEST_SLIPPAGE": "0.0005",  # 0.05%
        "BACKTEST_OUTPUT_DIR": "backtest_results",
        
        # Configuración de trading
        "SYMBOLS": "AAPL,MSFT,GOOGL,AMZN,TSLA",
        "TIMEFRAME": "1d",
        "STRATEGY": "ml_prediction",  # ml_prediction, mean_reversion, trend_following, ensemble
        
        # Configuración de modelo ML
        "MODEL_TYPE": "random_forest",  # random_forest, gradient_boosting, lstm
        "MODELS_DIR": "models",
        "FEATURE_WINDOW": "20",  # Ventana para características
        "PREDICTION_HORIZON": "5",  # Horizonte de predicción en barras
        
        # Configuración de riesgo
        "MAX_POSITION_SIZE": "0.1",  # 10% del capital por posición
        "MAX_DRAWDOWN": "0.1",  # 10% máximo drawdown
        "STOP_LOSS_PCT": "0.03",  # 3% stop loss
        "TRAILING_STOP_PCT": "0.02",  # 2% trailing stop
        
        # Otras configuraciones
        "LOG_LEVEL": "INFO",
        "DATA_DIR": "data",
    }
    
    # Crear directorio de salida si no existe
    os.makedirs(config["BACKTEST_OUTPUT_DIR"], exist_ok=True)
    
    # Inicializar el runner de backtest
    backtest_runner = BacktestRunner(config)
    
    # Ejecutar backtest
    logger.info("Iniciando backtest...")
    metrics = backtest_runner.run_backtest()
    
    # Mostrar resultados
    logger.info("Backtest completado. Resultados:")
    logger.info(f"Retorno total: {metrics['total_return_pct']:.2f}%")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
    logger.info(f"Win Rate: {metrics['win_rate']*100:.1f}%")
    logger.info(f"Profit Factor: {metrics['profit_factor']:.2f}")
    logger.info(f"Número de operaciones: {metrics['num_trades']}")
    
    # Comparar estrategias
    logger.info("\nComparando estrategias...")
    strategies = ["ml_prediction", "mean_reversion", "trend_following", "ensemble"]
    
    # Usar un período más corto para la comparación para que sea más rápido
    start_date = "2022-07-01"
    end_date = "2022-12-31"
    
    comparison_results = backtest_runner.compare_strategies(
        strategies=strategies,
        start_date=start_date,
        end_date=end_date
    )
    
    # Mostrar resultados de la comparación
    logger.info("Comparación de estrategias completada.")
    for strategy, metrics in comparison_results.items():
        logger.info(f"\nEstrategia: {strategy}")
        logger.info(f"Retorno total: {metrics['total_return_pct']:.2f}%")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {metrics['max_drawdown_pct']:.2f}%")
        logger.info(f"Win Rate: {metrics['win_rate']*100:.1f}%")

if __name__ == "__main__":
    main()