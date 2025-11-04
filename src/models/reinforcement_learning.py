#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo de aprendizaje por refuerzo para el sistema de trading.
Implementa Q-Learning para optimizar estrategias de trading basadas en resultados.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
from collections import defaultdict
import json
import os

logger = logging.getLogger(__name__)


class TradingReinforcementLearner:
    """Agente de aprendizaje por refuerzo para optimizar estrategias de trading."""

    def __init__(self, config: Dict[str, Any]):
        """Inicializa el agente de aprendizaje por refuerzo.

        Args:
            config: Configuración del sistema
        """
        self.config = config
        self.learning_rate = config.get("LEARNING_RATE", 0.1)
        self.discount_factor = config.get("DISCOUNT_FACTOR", 0.95)
        self.epsilon = config.get("EPSILON", 0.1)  # Para exploración
        self.memory_dir = config.get("MEMORY_DIR", "models/memory")
        os.makedirs(self.memory_dir, exist_ok=True)

        # Q-table: estado -> acción -> valor Q
        self.q_table = defaultdict(lambda: defaultdict(float))

        # Memoria de experiencias exitosas
        self.successful_strategies = defaultdict(list)

        # Historial de rendimiento
        self.performance_history = defaultdict(list)

        # Estados discretizados
        self.state_bins = {
            'rsi': [0, 30, 50, 70, 100],
            'sma_ratio': [0.95, 0.98, 1.0, 1.02, 1.05],
            'volatility': [0, 0.02, 0.05, 0.1, 1.0],
            'trend': [-0.1, -0.02, 0, 0.02, 0.1]
        }

        # Acciones posibles
        self.actions = ['BUY', 'SELL', 'HOLD']

        # Cargar memoria si existe
        self._load_memory()

        logger.info("Agente de aprendizaje por refuerzo inicializado")

    def get_state(self, features: pd.Series) -> str:
        """Discretiza el estado actual para la Q-table.

        Args:
            features: Serie con features del mercado

        Returns:
            str: Estado discretizado
        """
        try:
            # Extraer features relevantes
            rsi = features.get('rsi', 50)
            sma_ratio = features.get('sma_5', 1) / features.get('sma_20', 1) if features.get('sma_20', 1) != 0 else 1
            volatility = abs(features.get('returns', 0))
            trend = features.get('returns', 0)

            # Discretizar
            rsi_bin = np.digitize(rsi, self.state_bins['rsi']) - 1
            sma_bin = np.digitize(sma_ratio, self.state_bins['sma_ratio']) - 1
            vol_bin = np.digitize(volatility, self.state_bins['volatility']) - 1
            trend_bin = np.digitize(trend, self.state_bins['trend']) - 1

            # Crear estado como tupla
            state = (rsi_bin, sma_bin, vol_bin, trend_bin)
            return str(state)

        except Exception as e:
            logger.warning(f"Error al obtener estado: {e}")
            return "(2,2,0,2)"  # Estado neutral por defecto

    def choose_action(self, state: str, training: bool = True) -> str:
        """Elige una acción usando política epsilon-greedy.

        Args:
            state: Estado actual
            training: Si está en modo entrenamiento

        Returns:
            str: Acción elegida
        """
        if training and np.random.random() < self.epsilon:
            # Exploración: acción aleatoria
            return np.random.choice(self.actions)
        else:
            # Explotación: mejor acción según Q-table
            q_values = self.q_table[state]
            if not q_values:
                return np.random.choice(self.actions)

            # Encontrar acción con mayor valor Q
            best_action = max(q_values, key=q_values.get)
            return best_action

    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Actualiza el valor Q usando la ecuación de Bellman.

        Args:
            state: Estado actual
            action: Acción tomada
            reward: Recompensa obtenida
            next_state: Estado siguiente
        """
        # Valor Q actual
        current_q = self.q_table[state][action]

        # Mejor valor Q del siguiente estado
        next_q_values = self.q_table[next_state]
        max_next_q = max(next_q_values.values()) if next_q_values else 0

        # Actualizar Q usando Q-learning
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )

        self.q_table[state][action] = new_q

    def calculate_reward(self, action: str, actual_return: float, transaction_cost: float = 0.001) -> float:
        """Calcula la recompensa basada en la acción y el resultado.

        Args:
            action: Acción tomada ('BUY', 'SELL', 'HOLD')
            actual_return: Retorno real obtenido
            transaction_cost: Costo de transacción

        Returns:
            float: Recompensa calculada
        """
        if action == 'HOLD':
            # Recompensa pequeña por mantener posición sin costo
            reward = actual_return * 0.1
        elif action in ['BUY', 'SELL']:
            # Recompensa basada en el retorno menos costo de transacción
            if action == 'BUY' and actual_return > 0:
                reward = actual_return - transaction_cost
            elif action == 'SELL' and actual_return < 0:
                reward = abs(actual_return) - transaction_cost
            else:
                reward = -transaction_cost - abs(actual_return) * 0.5
        else:
            reward = 0

        # Escalar recompensa para estabilidad del aprendizaje
        return np.clip(reward, -1, 1)

    def learn_from_experience(self, symbol: str, experiences: List[Dict[str, Any]]):
        """Aprende de una lista de experiencias de trading.

        Args:
            symbol: Símbolo del instrumento
            experiences: Lista de experiencias de trading
        """
        total_reward = 0
        successful_trades = 0

        for i, exp in enumerate(experiences):
            state = exp['state']
            action = exp['action']
            reward = exp['reward']
            next_state = exp.get('next_state', state)  # Último estado si no hay siguiente

            # Actualizar Q-table
            self.update_q_value(state, action, reward, next_state)

            total_reward += reward
            if reward > 0:
                successful_trades += 1

        # Registrar rendimiento
        performance = {
            'timestamp': datetime.now().isoformat(),
            'total_reward': total_reward,
            'successful_trades': successful_trades,
            'total_trades': len(experiences),
            'win_rate': successful_trades / len(experiences) if experiences else 0
        }

        self.performance_history[symbol].append(performance)

        # Si el rendimiento es bueno, guardar estrategia
        if performance['win_rate'] > 0.6 and len(experiences) > 10:
            self._save_successful_strategy(symbol, performance)

        logger.info(f"Aprendizaje completado para {symbol}: Win Rate = {performance['win_rate']:.2%}")

    def get_optimal_action(self, state: str) -> str:
        """Obtiene la acción óptima para un estado dado.

        Args:
            state: Estado actual

        Returns:
            str: Acción óptima
        """
        q_values = self.q_table[state]
        if not q_values:
            return 'HOLD'  # Acción por defecto

        return max(q_values, key=q_values.get)

    def get_action_confidence(self, state: str, action: str) -> float:
        """Calcula la confianza en una acción específica para un estado dado.

        Args:
            state: Estado actual
            action: Acción a evaluar

        Returns:
            float: Confianza en la acción (0-1)
        """
        q_values = self.q_table[state]
        if not q_values:
            return 0.5  # Confianza neutral si no hay datos

        # Calcular confianza basada en la diferencia con otras acciones
        action_q = q_values.get(action, 0)
        other_q_values = [q for a, q in q_values.items() if a != action]

        if not other_q_values:
            return 0.5

        # Confianza = (acción_q - promedio_otras) / (máximo_q - mínimo_q + epsilon)
        max_q = max(q_values.values())
        min_q = min(q_values.values())
        range_q = max_q - min_q + 1e-6  # Evitar división por cero

        confidence = (action_q - sum(other_q_values) / len(other_q_values)) / range_q
        return max(0.0, min(1.0, confidence + 0.5))  # Normalizar a [0,1]

    def get_strategy_summary(self, symbol: str) -> Dict[str, Any]:
        """Obtiene un resumen de la estrategia aprendida para un símbolo.

        Args:
            symbol: Símbolo del instrumento

        Returns:
            Dict[str, Any]: Resumen de la estrategia
        """
        history = self.performance_history.get(symbol, [])

        if not history:
            return {"status": "no_history"}

        recent_performance = history[-10:]  # Últimas 10 sesiones

        return {
            "total_sessions": len(history),
            "avg_win_rate": np.mean([p['win_rate'] for p in recent_performance]),
            "avg_total_reward": np.mean([p['total_reward'] for p in recent_performance]),
            "best_win_rate": max([p['win_rate'] for p in history]),
            "q_table_size": len(self.q_table),
            "successful_strategies": len(self.successful_strategies.get(symbol, []))
        }

    def _save_successful_strategy(self, symbol: str, performance: Dict[str, Any]):
        """Guarda una estrategia exitosa.

        Args:
            symbol: Símbolo del instrumento
            performance: Datos de rendimiento
        """
        strategy = {
            'symbol': symbol,
            'performance': performance,
            'q_table_snapshot': dict(self.q_table),  # Copia de la Q-table
            'saved_at': datetime.now().isoformat()
        }

        self.successful_strategies[symbol].append(strategy)

        # Mantener solo las últimas 5 estrategias exitosas
        if len(self.successful_strategies[symbol]) > 5:
            self.successful_strategies[symbol] = self.successful_strategies[symbol][-5:]

    def _load_memory(self):
        """Carga la memoria guardada del agente."""
        memory_file = os.path.join(self.memory_dir, 'rl_memory.json')

        if os.path.exists(memory_file):
            try:
                with open(memory_file, 'r') as f:
                    data = json.load(f)

                self.q_table = defaultdict(lambda: defaultdict(float), data.get('q_table', {}))
                self.successful_strategies = defaultdict(list, data.get('successful_strategies', {}))
                self.performance_history = defaultdict(list, data.get('performance_history', {}))

                logger.info("Memoria del agente cargada exitosamente")

            except Exception as e:
                logger.error(f"Error al cargar memoria: {e}")

    def save_memory(self):
        """Guarda la memoria del agente."""
        memory_file = os.path.join(self.memory_dir, 'rl_memory.json')

        try:
            # Convertir defaultdicts a dicts regulares para serialización
            data = {
                'q_table': dict(self.q_table),
                'successful_strategies': dict(self.successful_strategies),
                'performance_history': dict(self.performance_history),
                'saved_at': datetime.now().isoformat()
            }

            with open(memory_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.info("Memoria del agente guardada")

        except Exception as e:
            logger.error(f"Error al guardar memoria: {e}")

    def reset_learning(self, symbol: Optional[str] = None):
        """Reinicia el aprendizaje para un símbolo o todos.

        Args:
            symbol: Símbolo específico o None para todos
        """
        if symbol:
            if symbol in self.q_table:
                del self.q_table[symbol]
            if symbol in self.successful_strategies:
                del self.successful_strategies[symbol]
            if symbol in self.performance_history:
                del self.performance_history[symbol]
        else:
            self.q_table.clear()
            self.successful_strategies.clear()
            self.performance_history.clear()

        logger.info(f"Aprendizaje reiniciado para {symbol if symbol else 'todos los símbolos'}")
