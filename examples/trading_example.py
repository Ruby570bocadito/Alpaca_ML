#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Ejemplo de uso del sistema de trading en modo real o papel
"""

import os
import sys
import logging
import argparse
from datetime import datetime, timedelta
import time

# Añadir el directorio raíz al path para importar los módulos del proyecto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import load_config
from src.data.ingest import DataIngestionManager
from src.features.engineering import FeatureEngineer
from src.models.model import ModelManager
from src.strategy.signals import SignalGenerator
from src.strategy.portfolio import PortfolioManager
from src.execution.alpaca_client import AlpacaClient
from src.execution.order_manager import OrderManager
from src.risk.risk_manager import RiskManager
from src.monitoring.metrics import MetricsManager
from src.monitoring.alerts import AlertManager
from src.utils.logging import setup_logging
from src.utils.time_utils import is_market_hours, get_next_market_open, get_next_market_close

# Configurar logging
setup_logging(log_level="INFO", console=True, log_file="trading_bot.log")
logger = logging.getLogger(__name__)

def parse_args():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Sistema de Trading con Alpaca")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["paper", "live"], 
        default="paper",
        help="Modo de trading: paper o live"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=".env",
        help="Ruta al archivo de configuración"
    )
    parser.add_argument(
        "--symbols", 
        type=str, 
        help="Lista de símbolos separados por comas (sobreescribe la configuración)"
    )
    parser.add_argument(
        "--strategy", 
        type=str, 
        choices=["ml_prediction", "mean_reversion", "trend_following", "ensemble"],
        help="Estrategia a utilizar (sobreescribe la configuración)"
    )
    return parser.parse_args()

def initialize_components(config):
    """Inicializa todos los componentes del sistema.
    
    Args:
        config: Configuración del sistema
        
    Returns:
        Tuple con todos los componentes inicializados
    """
    logger.info("Inicializando componentes del sistema...")
    
    # Inicializar cliente de Alpaca
    alpaca_client = AlpacaClient(
        api_key=config["ALPACA_API_KEY"],
        api_secret=config["ALPACA_API_SECRET"],
        base_url=config["ALPACA_BASE_URL"],
        data_url=config["ALPACA_DATA_URL"]
    )
    
    # Verificar conexión con Alpaca
    account = alpaca_client.get_account()
    logger.info(f"Conectado a Alpaca. Cuenta: {account.id}, Saldo: ${float(account.equity):.2f}")
    
    # Inicializar componentes
    data_manager = DataIngestionManager(config, alpaca_client=alpaca_client)
    feature_engineer = FeatureEngineer(config)
    model_manager = ModelManager(config)
    signal_generator = SignalGenerator(config)
    portfolio_manager = PortfolioManager(config)
    order_manager = OrderManager(config, alpaca_client=alpaca_client)
    risk_manager = RiskManager(config, alpaca_client=alpaca_client, order_manager=order_manager)
    metrics_manager = MetricsManager(config)
    alert_manager = AlertManager(config)
    
    # Cargar modelos pre-entrenados
    symbols = config["SYMBOLS"].split(",")
    for symbol in symbols:
        try:
            model_manager.load_model(symbol)
            logger.info(f"Modelo cargado para {symbol}")
        except Exception as e:
            logger.warning(f"No se pudo cargar el modelo para {symbol}: {e}")
    
    return (
        alpaca_client, data_manager, feature_engineer, model_manager,
        signal_generator, portfolio_manager, order_manager, risk_manager,
        metrics_manager, alert_manager
    )

def run_trading_loop(config, components):
    """Ejecuta el bucle principal de trading.
    
    Args:
        config: Configuración del sistema
        components: Tuple con todos los componentes inicializados
    """
    (
        alpaca_client, data_manager, feature_engineer, model_manager,
        signal_generator, portfolio_manager, order_manager, risk_manager,
        metrics_manager, alert_manager
    ) = components
    
    symbols = config["SYMBOLS"].split(",")
    timeframe = config["TIMEFRAME"]
    strategy = config["STRATEGY"]
    update_interval = int(config.get("UPDATE_INTERVAL", "60"))  # segundos
    
    logger.info(f"Iniciando bucle de trading con {len(symbols)} símbolos, "
               f"timeframe {timeframe}, estrategia {strategy}")
    
    # Iniciar streaming de datos si es necesario
    if timeframe in ["1m", "5m", "15m"]:
        logger.info("Iniciando streaming de datos en tiempo real...")
        data_manager.start_data_stream(symbols)
    
    # Bucle principal
    try:
        while True:
            # Verificar si el mercado está abierto
            if not is_market_hours():
                next_open = get_next_market_open()
                logger.info(f"Mercado cerrado. Próxima apertura: {next_open}")
                
                # Dormir hasta 5 minutos antes de la apertura
                sleep_time = (next_open - datetime.now()).total_seconds() - 300
                if sleep_time > 0:
                    logger.info(f"Durmiendo por {sleep_time/60:.1f} minutos...")
                    time.sleep(min(sleep_time, 3600))  # Máximo 1 hora para poder hacer comprobaciones
                    continue
            
            # Verificar si el mercado está por cerrar
            next_close = get_next_market_close()
            time_to_close = (next_close - datetime.now()).total_seconds()
            
            if time_to_close < 300:  # 5 minutos antes del cierre
                logger.info("Mercado a punto de cerrar. Cerrando posiciones...")
                risk_manager.execute_circuit_breaker(reason="market_closing")
                
                # Dormir hasta después del cierre
                sleep_time = time_to_close + 60  # 1 minuto después del cierre
                logger.info(f"Durmiendo por {sleep_time/60:.1f} minutos...")
                time.sleep(sleep_time)
                continue
            
            # Actualizar métricas de cuenta
            account = alpaca_client.get_account()
            metrics_manager.update_equity(float(account.equity))
            metrics_manager.update_cash(float(account.cash))
            
            # Procesar cada símbolo
            for symbol in symbols:
                try:
                    # Obtener datos recientes
                    df = data_manager.get_recent_data(symbol, timeframe, bars=100)
                    
                    if df is None or df.empty:
                        logger.warning(f"No hay datos disponibles para {symbol}")
                        continue
                    
                    # Generar características
                    df_features = feature_engineer.generate_features(df, symbol)
                    
                    if df_features is None or df_features.empty:
                        logger.warning(f"No se pudieron generar características para {symbol}")
                        continue
                    
                    # Preparar datos para predicción
                    X = feature_engineer.prepare_prediction_data(df_features)
                    
                    if X is None or X.empty:
                        logger.warning(f"No se pudieron preparar datos para predicción de {symbol}")
                        continue
                    
                    # Hacer predicción
                    prediction = None
                    if strategy == "ml_prediction":
                        prediction = model_manager.predict(X.iloc[-1:], symbol)
                        metrics_manager.update_model_prediction(symbol, prediction)
                    
                    # Obtener posiciones actuales
                    positions = alpaca_client.get_positions()
                    current_positions = {p.symbol: int(p.qty) for p in positions}
                    
                    # Generar señal
                    signal = signal_generator.generate_signal(
                        symbol=symbol,
                        data=df_features,
                        prediction=prediction,
                        current_positions=current_positions,
                        strategy=strategy
                    )
                    
                    if signal:
                        metrics_manager.update_signal_count(signal["direction"])
                        logger.info(f"Señal generada para {symbol}: {signal['direction']} "
                                   f"(confianza: {signal.get('confidence', 0):.2f})")
                        
                        # Registrar señal en alertas
                        alert_manager.send_signal_alert(
                            symbol=symbol,
                            direction=signal["direction"],
                            confidence=signal.get("confidence", 0),
                            strategy=signal.get("strategy", strategy)
                        )
                    
                except Exception as e:
                    logger.error(f"Error procesando {symbol}: {e}", exc_info=True)
            
            # Calcular tamaños de posición para todas las señales
            all_signals = signal_generator.get_active_signals()
            
            if all_signals:
                # Obtener posiciones actuales
                positions = alpaca_client.get_positions()
                current_positions = {p.symbol: int(p.qty) for p in positions}
                
                # Calcular tamaños de posición
                position_sizes = portfolio_manager.calculate_position_sizes(
                    signals=all_signals,
                    equity=float(account.equity),
                    current_positions=current_positions
                )
                
                # Ejecutar órdenes
                for symbol, target_size in position_sizes.items():
                    current_size = current_positions.get(symbol, 0)
                    
                    # Si hay cambio en la posición
                    if target_size != current_size:
                        # Verificar con el gestor de riesgo
                        if risk_manager.validate_order(symbol, target_size - current_size):
                            try:
                                # Crear orden
                                if target_size > current_size:  # Compra
                                    qty = target_size - current_size
                                    order_id = order_manager.submit_order(
                                        symbol=symbol,
                                        qty=qty,
                                        side="buy",
                                        order_type="market",
                                        time_in_force="day"
                                    )
                                    logger.info(f"Orden de compra enviada para {symbol}: {qty} acciones")
                                    metrics_manager.update_order_count("buy")
                                    
                                    # Configurar stop loss si está habilitado
                                    if float(config.get("STOP_LOSS_PCT", "0")) > 0:
                                        risk_manager.set_stop_loss(symbol, qty)
                                    
                                elif target_size < current_size:  # Venta
                                    qty = current_size - target_size
                                    order_id = order_manager.submit_order(
                                        symbol=symbol,
                                        qty=qty,
                                        side="sell",
                                        order_type="market",
                                        time_in_force="day"
                                    )
                                    logger.info(f"Orden de venta enviada para {symbol}: {qty} acciones")
                                    metrics_manager.update_order_count("sell")
                            
                            except Exception as e:
                                logger.error(f"Error al enviar orden para {symbol}: {e}", exc_info=True)
                                metrics_manager.update_error_count("order_submission")
                                alert_manager.send_error_alert(
                                    error_type="order_submission",
                                    message=f"Error al enviar orden para {symbol}: {str(e)}"
                                )
            
            # Verificar circuit breakers
            if risk_manager.check_circuit_breakers():
                logger.warning("Circuit breaker activado. Cerrando todas las posiciones.")
                risk_manager.execute_circuit_breaker(reason="risk_limit_exceeded")
                alert_manager.send_risk_alert(
                    alert_type="circuit_breaker",
                    message="Circuit breaker activado. Se han cerrado todas las posiciones."
                )
            
            # Registrar métricas
            metrics_manager.log_metrics()
            
            # Dormir hasta la próxima actualización
            logger.info(f"Durmiendo por {update_interval} segundos...")
            time.sleep(update_interval)
    
    except KeyboardInterrupt:
        logger.info("Interrupción de teclado detectada. Cerrando el sistema...")
    
    except Exception as e:
        logger.critical(f"Error crítico en el bucle de trading: {e}", exc_info=True)
        alert_manager.send_system_alert(
            alert_type="critical_error",
            message=f"Error crítico en el sistema: {str(e)}"
        )
    
    finally:
        # Limpiar recursos
        logger.info("Limpiando recursos...")
        if timeframe in ["1m", "5m", "15m"]:
            data_manager.stop_data_stream()
        
        # Guardar métricas finales
        metrics_manager.export_metrics_to_csv("trading_metrics.csv")
        logger.info("Sistema cerrado correctamente.")

def main():
    # Parsear argumentos
    args = parse_args()
    
    # Cargar configuración
    config = load_config(args.config)
    
    # Sobreescribir configuración con argumentos
    if args.symbols:
        config["SYMBOLS"] = args.symbols
    
    if args.strategy:
        config["STRATEGY"] = args.strategy
    
    # Configurar modo de trading
    if args.mode == "paper":
        config["ALPACA_BASE_URL"] = "https://paper-api.alpaca.markets"
    elif args.mode == "live":
        config["ALPACA_BASE_URL"] = "https://api.alpaca.markets"
        
        # Confirmación adicional para trading en vivo
        confirm = input("¡ADVERTENCIA! Estás a punto de iniciar el trading en modo REAL. "
                       "¿Estás seguro? (s/N): ")
        if confirm.lower() != "s":
            logger.info("Trading en vivo cancelado por el usuario.")
            return
    
    logger.info(f"Iniciando sistema en modo {args.mode.upper()}")
    
    # Inicializar componentes
    components = initialize_components(config)
    
    # Ejecutar bucle de trading
    run_trading_loop(config, components)

if __name__ == "__main__":
    main()