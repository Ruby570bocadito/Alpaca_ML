#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Punto de entrada principal para el sistema de trading con IA para Alpaca.
Este módulo orquesta todos los componentes del sistema.
"""

import argparse
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import pandas as pd

# Importaciones internas
from src.config import load_config
from src.data.ingest import DataIngestionManager
from src.features.engineering import FeatureEngineer
from src.models.model import ModelManager
from src.models.reinforcement_learning import TradingReinforcementLearner
from src.strategy.signals import SignalGenerator
from src.strategy.portfolio import PortfolioManager
from src.execution.alpaca_client import AlpacaClient
from src.execution.order_manager import OrderManager
from src.risk.risk_manager import RiskManager
from src.monitoring.metrics import MetricsManager
from src.utils.logging import setup_logging


def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Sistema de Trading con IA para Alpaca")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["backtest", "paper", "live", "sim_rl_100"],
        default="paper",
        help="Modo de ejecución: backtest, paper, live, sim_rl_100",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="ml_strategy",
        help="Estrategia a utilizar",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Fecha de inicio para backtesting (formato: YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="Fecha de fin para backtesting (formato: YYYY-MM-DD)",
    )
    parser.add_argument(
        "--additional-days",
        type=int,
        default=100,
        help="Número de días adicionales para simulación RL",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        help="Lista de símbolos separados por comas",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Nivel de logging",
    )
    return parser.parse_args()


class TradingSystem:
    """Clase principal que orquesta el sistema de trading."""

    def __init__(self, config: Dict):
        """Inicializa el sistema de trading.

        Args:
            config: Configuración del sistema
        """
        self.config = config
        self.mode = config.get("TRADING_MODE", "paper")
        self.symbols = config.get("SYMBOLS", "").split(",")
        self.logger = logging.getLogger(__name__)

        # Inicializar componentes
        if self.mode in ["sim_rl_100", "backtest"]:
            # Modos de simulación no requieren cliente Alpaca
            self.alpaca_client = None
            self.data_manager = DataIngestionManager(config, None)
            self.order_manager = None
            self.risk_manager = None
            self.metrics_collector = None
        else:
            self.alpaca_client = AlpacaClient(
                api_key=config.get("ALPACA_API_KEY"),
                api_secret=config.get("ALPACA_API_SECRET"),
                base_url=config.get("ALPACA_BASE_URL"),
                trading_mode=config.get("TRADING_MODE", "paper"),
                use_mcp=config.get("ALPACA_MCP_ENABLED", False),
                mcp_url=f"http://localhost:{config.get('ALPACA_MCP_PORT', 5000)}"
            )
            self.data_manager = DataIngestionManager(config, self.alpaca_client)
            self.order_manager = OrderManager(config, self.alpaca_client)
            self.risk_manager = RiskManager(config, self.alpaca_client, self.order_manager)
            self.metrics_collector = MetricsManager(config, self.alpaca_client, self.risk_manager)

        self.feature_engineer = FeatureEngineer(config)
        self.model_manager = ModelManager(config)
        self.signal_generator = SignalGenerator(config)
        self.portfolio_manager = PortfolioManager(config)

        # Inicializar agente de aprendizaje por refuerzo
        self.rl_agent = TradingReinforcementLearner(config)

        self.logger.info(f"Sistema de trading inicializado en modo: {self.mode}")
        self.logger.info(f"Símbolos configurados: {self.symbols}")

    def run_backtest(self, start_date: str, end_date: str):
        """Ejecuta el sistema en modo backtest.

        Args:
            start_date: Fecha de inicio del backtest
            end_date: Fecha de fin del backtest
        """
        self.logger.info(f"Iniciando backtest desde {start_date} hasta {end_date}")

        # Obtener datos históricos
        historical_data = self.data_manager.get_historical_data(
            symbols=self.symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe="1D",
        )

        # Procesar datos y generar features (no dropear NaN para backtest)
        features_data = self.feature_engineer.process_data(historical_data, drop_na=False)

        # Generar predicciones
        predictions = self.model_manager.predict(features_data)

        # Generar señales
        signals = self.signal_generator.generate_signals(predictions)

        # Simular ejecución de órdenes
        portfolio_history = self.portfolio_manager.simulate_portfolio(
            signals=signals,
            historical_data=historical_data,
        )

        # Calcular métricas
        metrics = self.metrics_collector.calculate_backtest_metrics(portfolio_history)

        self.logger.info(f"Backtest completado. Métricas: {metrics}")
        return metrics

    def run_rl_simulation(self, additional_days: int = 100):
        """Ejecuta una simulación de aprendizaje por refuerzo durante días adicionales.

        Args:
            additional_days: Número de días adicionales para la simulación
        """
        if not self.rl_agent:
            self.logger.error("Agente de aprendizaje por refuerzo no disponible")
            return

        self.logger.info(f"Iniciando simulación RL con {additional_days} días adicionales")

        # Configuración de la simulación
        simulation_days = additional_days
        current_date = pd.Timestamp.now()

        # Obtener datos históricos para la simulación
        end_date = current_date.strftime("%Y-%m-%d")
        start_date = (current_date - pd.Timedelta(days=simulation_days + 100)).strftime("%Y-%m-%d")  # Datos adicionales para features

        try:
            # Obtener datos históricos
            historical_data = self.data_manager.get_historical_data(
                symbols=self.symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe="1D",
            )

            # Procesar datos y generar features
            features_data = self.feature_engineer.process_data(historical_data)

            # Simular trading con RL
            experiences = []
            portfolio_value = 100000  # Valor inicial del portfolio
            positions = {symbol: 0 for symbol in self.symbols}

            # Usar los últimos días para la simulación
            simulation_dates = list(features_data[self.symbols[0]].index)[-simulation_days:]

            for i, date in enumerate(simulation_dates):
                self.logger.info(f"Simulando día {i+1}/{simulation_days}: {date.strftime('%Y-%m-%d')}")

                # Obtener features del día actual
                current_features = {}
                for symbol in self.symbols:
                    if symbol in features_data and date in features_data[symbol].index:
                        current_features[symbol] = features_data[symbol].loc[date]

                if not current_features:
                    continue

                # Para cada símbolo, decidir acción usando RL
                for symbol in self.symbols:
                    if symbol not in current_features:
                        continue

                    features = current_features[symbol]
                    state = self.rl_agent.get_state(features)
                    action = self.rl_agent.choose_action(state, training=True)

                    # Simular resultado de la acción con lotajes dinámicos
                    next_date = simulation_dates[i + 1] if i + 1 < len(simulation_dates) else date
                    next_features = {}
                    if symbol in features_data and next_date in features_data[symbol].index:
                        next_features = features_data[symbol].loc[next_date]

                    # Calcular lotaje basado en confianza del agente RL
                    confidence = self.rl_agent.get_action_confidence(state, action)
                    position_size_pct = min(confidence * 0.2, 0.15)  # Máximo 15% del portfolio por posición

                    # Calcular retorno basado en la acción
                    if action == 'BUY' and positions[symbol] == 0:
                        # Simular compra con lotaje dinámico
                        price = features.get('close', 100)
                        position_size = portfolio_value * position_size_pct
                        shares = position_size / price
                        positions[symbol] = shares
                        portfolio_value -= position_size
                        actual_return = 0  # No hay retorno inmediato en compra

                    elif action == 'SELL' and positions[symbol] > 0:
                        # Simular venta
                        price = features.get('close', 100)
                        sale_value = positions[symbol] * price
                        portfolio_value += sale_value
                        actual_return = (price - features.get('open', price)) / features.get('open', price)  # Retorno diario
                        positions[symbol] = 0

                    elif action == 'HOLD':
                        # Mantener posición
                        if positions[symbol] > 0:
                            price_change = (features.get('close', 100) - features.get('open', 100)) / features.get('open', 100)
                            portfolio_value += positions[symbol] * features.get('close', 100) * price_change
                        actual_return = (features.get('close', 100) - features.get('open', 100)) / features.get('open', 100)

                    else:
                        actual_return = 0

                    # Calcular recompensa
                    reward = self.rl_agent.calculate_reward(action, actual_return)

                    # Estado siguiente
                    next_state = self.rl_agent.get_state(next_features) if next_features else state

                    # Almacenar experiencia con información adicional
                    experience = {
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'next_state': next_state,
                        'confidence': confidence,
                        'position_size_pct': position_size_pct,
                        'symbol': symbol,
                        'date': date.strftime('%Y-%m-%d')
                    }
                    experiences.append(experience)

                    # Actualizar Q-table
                    self.rl_agent.update_q_value(state, action, reward, next_state)

                # Mostrar progreso cada 10 días
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Progreso: Día {i+1}/{simulation_days}, Portfolio: ${portfolio_value:.2f}")

            # Aprender de todas las experiencias
            self.rl_agent.learn_from_experience(self.symbols[0], experiences)  # Usar primer símbolo como ejemplo

            # Guardar memoria del agente
            self.rl_agent.save_memory()

            # Mostrar resumen final
            final_metrics = self.rl_agent.get_strategy_summary(self.symbols[0])
            self.logger.info("=" * 60)
            self.logger.info("SIMULACIÓN RL COMPLETADA")
            self.logger.info("=" * 60)
            self.logger.info(f"Valor final del portfolio: ${portfolio_value:.2f}")
            self.logger.info(f"Experiencias recolectadas: {len(experiences)}")
            self.logger.info(f"Resumen de estrategia: {final_metrics}")
            self.logger.info("=" * 60)

        except Exception as e:
            self.logger.error(f"Error en simulación RL: {e}", exc_info=True)
            raise

    def run_trading_loop(self):
        """Ejecuta el bucle principal de trading en modo paper o live."""
        self.logger.info(f"Iniciando bucle de trading en modo: {self.mode}")

        try:
            while True:
                # Verificar horario de mercado
                if not self._is_market_open():
                    self.logger.info("Mercado cerrado. Esperando...")
                    time.sleep(60)  # Esperar 1 minuto
                    continue

                # Obtener datos en tiempo real
                market_data = self.data_manager.get_realtime_data(self.symbols)

                # Procesar datos y generar features
                features_data = self.feature_engineer.process_data(market_data)

                # Generar predicciones
                predictions = self.model_manager.predict(features_data)

                # Generar señales
                signals = self.signal_generator.generate_signals(predictions)

                # Verificar restricciones de riesgo
                approved_signals = self.risk_manager.validate_signals(signals)

                # Generar órdenes basadas en señales aprobadas
                orders = self.portfolio_manager.generate_orders(approved_signals)

                # Ejecutar órdenes
                if orders:
                    self.order_manager.execute_orders(orders)

                # Recolectar métricas
                self.metrics_collector.collect_metrics()

                # Esperar para el siguiente ciclo
                time.sleep(self.config.get("TRADING_INTERVAL_SECONDS", 60))

        except KeyboardInterrupt:
            self.logger.info("Deteniendo el sistema de trading...")
            self._cleanup()
        except Exception as e:
            self.logger.error(f"Error en el bucle de trading: {e}", exc_info=True)
            self._cleanup()
            raise

    def _is_market_open(self) -> bool:
        """Verifica si el mercado está abierto."""
        if self.mode in ["backtest", "sim_rl_100"]:
            return True
        return self.alpaca_client.is_market_open()

    def _cleanup(self):
        """Limpia recursos y realiza tareas de cierre."""
        self.logger.info("Realizando limpieza de recursos...")
        # Cerrar conexiones, guardar estado, etc.
        if self.metrics_collector:
            self.metrics_collector.save_metrics()


def main():
    """Función principal."""
    # Parsear argumentos
    args = parse_args()

    try:
        # Cargar configuración
        config = load_config()

        # Ajustar nivel de logging desde CLI y configurar logging correctamente
        if args.log_level:
            config["LOG_LEVEL"] = args.log_level
        setup_logging(config)
        logger = logging.getLogger(__name__)

        # Sobrescribir configuración con argumentos de línea de comandos
        if args.mode:
            config["TRADING_MODE"] = args.mode
        if args.strategy:
            config["STRATEGY"] = args.strategy
        if args.symbols:
            config["SYMBOLS"] = args.symbols

        # Inicializar sistema
        trading_system = TradingSystem(config)

        # Ejecutar en modo correspondiente
        if args.mode == "backtest":
            if not args.start_date or not args.end_date:
                logger.error("Se requieren fechas de inicio y fin para el modo backtest")
                sys.exit(1)
            trading_system.run_backtest(args.start_date, args.end_date)
        elif args.mode == "sim_rl_100":
            trading_system.run_rl_simulation(args.additional_days)
        else:
            trading_system.run_trading_loop()

    except Exception as e:
        logger.error(f"Error al iniciar el sistema: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
