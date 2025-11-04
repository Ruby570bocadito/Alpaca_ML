#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Módulo para la gestión de alertas y notificaciones del sistema de trading.
"""

import logging
import json
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import threading
import time
import os

logger = logging.getLogger(__name__)


class AlertManager:
    """Clase para gestionar alertas y notificaciones del sistema de trading."""

    def __init__(self, config: Dict[str, Any]):
        """Inicializa el gestor de alertas.

        Args:
            config: Configuración del sistema
        """
        self.config = config
        
        # Configuración de alertas
        self.alerts_config = {
            "enabled": config.get("ALERTS_ENABLED", "true").lower() == "true",
            "email_enabled": config.get("EMAIL_ALERTS_ENABLED", "false").lower() == "true",
            "slack_enabled": config.get("SLACK_ALERTS_ENABLED", "false").lower() == "true",
            "telegram_enabled": config.get("TELEGRAM_ALERTS_ENABLED", "false").lower() == "true",
            "webhook_enabled": config.get("WEBHOOK_ALERTS_ENABLED", "false").lower() == "true",
            "log_alerts": config.get("LOG_ALERTS", "true").lower() == "true",
            "alert_levels": config.get("ALERT_LEVELS", "critical,error,warning").split(","),
            "throttle_seconds": int(config.get("ALERT_THROTTLE_SECONDS", "300")),
            "max_daily_alerts": int(config.get("MAX_DAILY_ALERTS", "50")),
            "email_from": config.get("EMAIL_FROM", ""),
            "email_to": config.get("EMAIL_TO", "").split(","),
            "email_subject_prefix": config.get("EMAIL_SUBJECT_PREFIX", "[Trading Bot Alert]"),
            "smtp_server": config.get("SMTP_SERVER", ""),
            "smtp_port": int(config.get("SMTP_PORT", "587")),
            "smtp_username": config.get("SMTP_USERNAME", ""),
            "smtp_password": config.get("SMTP_PASSWORD", ""),
            "slack_webhook_url": config.get("SLACK_WEBHOOK_URL", ""),
            "telegram_bot_token": config.get("TELEGRAM_BOT_TOKEN", ""),
            "telegram_chat_id": config.get("TELEGRAM_CHAT_ID", ""),
            "webhook_url": config.get("WEBHOOK_URL", ""),
        }
        
        # Estado de alertas
        self.alerts_state = {
            "last_alert_time": {},  # Última vez que se envió una alerta por tipo
            "daily_alert_count": 0,  # Contador diario de alertas
            "daily_reset_time": datetime.now(),  # Hora de reinicio del contador diario
            "alert_history": [],  # Historial de alertas
        }
        
        # Iniciar hilo de reinicio diario
        self._start_daily_reset_thread()
        
        logger.info("Gestor de alertas inicializado")

    def _start_daily_reset_thread(self):
        """Inicia un hilo para reiniciar contadores diarios."""
        def reset_daily_counters():
            while True:
                try:
                    # Calcular tiempo hasta medianoche
                    now = datetime.now()
                    tomorrow = now + timedelta(days=1)
                    midnight = datetime(year=tomorrow.year, month=tomorrow.month, day=tomorrow.day, 
                                       hour=0, minute=0, second=0)
                    seconds_until_midnight = (midnight - now).total_seconds()
                    
                    # Dormir hasta medianoche
                    time.sleep(seconds_until_midnight)
                    
                    # Reiniciar contadores
                    self.alerts_state["daily_alert_count"] = 0
                    self.alerts_state["daily_reset_time"] = datetime.now()
                    
                    logger.info("Contadores de alertas diarios reiniciados")
                    
                except Exception as e:
                    logger.error(f"Error en hilo de reinicio diario: {e}", exc_info=True)
                    time.sleep(3600)  # Esperar una hora y reintentar
        
        reset_thread = threading.Thread(target=reset_daily_counters, daemon=True)
        reset_thread.start()
        logger.info("Hilo de reinicio diario de alertas iniciado")

    def send_alert(self, level: str, message: str, details: Optional[Dict[str, Any]] = None, 
                  alert_type: str = "general", throttle: bool = True) -> bool:
        """Envía una alerta.

        Args:
            level: Nivel de alerta (critical, error, warning, info)
            message: Mensaje de alerta
            details: Detalles adicionales (opcional)
            alert_type: Tipo de alerta para throttling
            throttle: Si se debe aplicar throttling

        Returns:
            bool: True si la alerta se envió correctamente
        """
        if not self.alerts_config["enabled"]:
            return False
        
        # Verificar nivel de alerta
        if level.lower() not in self.alerts_config["alert_levels"]:
            return False
        
        # Verificar límite diario
        if self.alerts_state["daily_alert_count"] >= self.alerts_config["max_daily_alerts"]:
            logger.warning(f"Límite diario de alertas alcanzado ({self.alerts_config['max_daily_alerts']}). "
                         f"Alerta no enviada: {message}")
            return False
        
        # Verificar throttling
        if throttle and alert_type in self.alerts_state["last_alert_time"]:
            last_time = self.alerts_state["last_alert_time"][alert_type]
            elapsed_seconds = (datetime.now() - last_time).total_seconds()
            
            if elapsed_seconds < self.alerts_config["throttle_seconds"]:
                logger.debug(f"Alerta throttled: {alert_type} - {message}")
                return False
        
        # Preparar datos de alerta
        timestamp = datetime.now()
        alert_data = {
            "timestamp": timestamp.isoformat(),
            "level": level.lower(),
            "message": message,
            "details": details or {},
            "alert_type": alert_type,
        }
        
        # Registrar en historial
        self.alerts_state["alert_history"].append(alert_data)
        
        # Limitar tamaño del historial
        if len(self.alerts_state["alert_history"]) > 1000:
            self.alerts_state["alert_history"] = self.alerts_state["alert_history"][-1000:]
        
        # Actualizar contadores
        self.alerts_state["last_alert_time"][alert_type] = timestamp
        self.alerts_state["daily_alert_count"] += 1
        
        # Registrar en log
        if self.alerts_config["log_alerts"]:
            log_method = getattr(logger, level.lower(), logger.warning)
            log_method(f"ALERTA: {message}")
            if details:
                log_method(f"Detalles: {json.dumps(details, default=str)}")
        
        # Enviar por canales configurados
        success = True
        
        if self.alerts_config["email_enabled"]:
            email_success = self._send_email_alert(level, message, details, alert_type)
            success = success and email_success
        
        if self.alerts_config["slack_enabled"]:
            slack_success = self._send_slack_alert(level, message, details, alert_type)
            success = success and slack_success
        
        if self.alerts_config["telegram_enabled"]:
            telegram_success = self._send_telegram_alert(level, message, details, alert_type)
            success = success and telegram_success
        
        if self.alerts_config["webhook_enabled"]:
            webhook_success = self._send_webhook_alert(level, message, details, alert_type)
            success = success and webhook_success
        
        return success

    def _send_email_alert(self, level: str, message: str, details: Optional[Dict[str, Any]], 
                        alert_type: str) -> bool:
        """Envía una alerta por email.

        Args:
            level: Nivel de alerta
            message: Mensaje de alerta
            details: Detalles adicionales
            alert_type: Tipo de alerta

        Returns:
            bool: True si la alerta se envió correctamente
        """
        try:
            # Verificar configuración
            if not all([self.alerts_config["smtp_server"], 
                       self.alerts_config["email_from"], 
                       self.alerts_config["email_to"]]):
                logger.error("Configuración de email incompleta")
                return False
            
            # Crear mensaje
            msg = MIMEMultipart()
            msg["From"] = self.alerts_config["email_from"]
            msg["To"] = ", ".join(self.alerts_config["email_to"])
            msg["Subject"] = f"{self.alerts_config['email_subject_prefix']} {level.upper()}: {alert_type}"
            
            # Construir cuerpo del mensaje
            body = f"<h2>{message}</h2>\n"
            body += f"<p><strong>Nivel:</strong> {level.upper()}</p>\n"
            body += f"<p><strong>Tipo:</strong> {alert_type}</p>\n"
            body += f"<p><strong>Fecha:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n"
            
            if details:
                body += "<h3>Detalles:</h3>\n"
                body += "<pre>" + json.dumps(details, indent=2, default=str) + "</pre>\n"
            
            msg.attach(MIMEText(body, "html"))
            
            # Enviar email
            with smtplib.SMTP(self.alerts_config["smtp_server"], self.alerts_config["smtp_port"]) as server:
                if self.alerts_config["smtp_username"] and self.alerts_config["smtp_password"]:
                    server.starttls()
                    server.login(self.alerts_config["smtp_username"], self.alerts_config["smtp_password"])
                
                server.send_message(msg)
            
            logger.info(f"Alerta enviada por email: {level} - {message}")
            return True
            
        except Exception as e:
            logger.error(f"Error al enviar alerta por email: {e}", exc_info=True)
            return False

    def _send_slack_alert(self, level: str, message: str, details: Optional[Dict[str, Any]], 
                        alert_type: str) -> bool:
        """Envía una alerta a Slack.

        Args:
            level: Nivel de alerta
            message: Mensaje de alerta
            details: Detalles adicionales
            alert_type: Tipo de alerta

        Returns:
            bool: True si la alerta se envió correctamente
        """
        try:
            # Verificar configuración
            if not self.alerts_config["slack_webhook_url"]:
                logger.error("URL de webhook de Slack no configurada")
                return False
            
            # Determinar color según nivel
            color_map = {
                "critical": "#FF0000",  # Rojo
                "error": "#FF9900",    # Naranja
                "warning": "#FFCC00",  # Amarillo
                "info": "#36C5F0",     # Azul
            }
            color = color_map.get(level.lower(), "#CCCCCC")
            
            # Construir payload
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"{level.upper()}: {alert_type}",
                        "text": message,
                        "fields": [
                            {
                                "title": "Nivel",
                                "value": level.upper(),
                                "short": True
                            },
                            {
                                "title": "Tipo",
                                "value": alert_type,
                                "short": True
                            },
                            {
                                "title": "Fecha",
                                "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "short": True
                            }
                        ],
                        "footer": "Trading Bot Alert System",
                        "ts": int(time.time())
                    }
                ]
            }
            
            # Añadir detalles si existen
            if details:
                payload["attachments"][0]["fields"].append({
                    "title": "Detalles",
                    "value": f"```{json.dumps(details, indent=2, default=str)}```",
                    "short": False
                })
            
            # Enviar a Slack
            response = requests.post(
                self.alerts_config["slack_webhook_url"],
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                logger.info(f"Alerta enviada a Slack: {level} - {message}")
                return True
            else:
                logger.error(f"Error al enviar alerta a Slack: {response.status_code} - {response.text}")
                return False
            
        except Exception as e:
            logger.error(f"Error al enviar alerta a Slack: {e}", exc_info=True)
            return False

    def _send_telegram_alert(self, level: str, message: str, details: Optional[Dict[str, Any]], 
                           alert_type: str) -> bool:
        """Envía una alerta a Telegram.

        Args:
            level: Nivel de alerta
            message: Mensaje de alerta
            details: Detalles adicionales
            alert_type: Tipo de alerta

        Returns:
            bool: True si la alerta se envió correctamente
        """
        try:
            # Verificar configuración
            if not all([self.alerts_config["telegram_bot_token"], self.alerts_config["telegram_chat_id"]]):
                logger.error("Configuración de Telegram incompleta")
                return False
            
            # Construir mensaje
            emoji_map = {
                "critical": "🚨",
                "error": "❌",
                "warning": "⚠️",
                "info": "ℹ️",
            }
            emoji = emoji_map.get(level.lower(), "📊")
            
            text = f"{emoji} *{level.upper()}: {alert_type}*\n\n"
            text += f"{message}\n\n"
            text += f"*Fecha:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            if details:
                text += "\n*Detalles:*\n"
                text += f"```\n{json.dumps(details, indent=2, default=str)}\n```"
            
            # Enviar a Telegram
            url = f"https://api.telegram.org/bot{self.alerts_config['telegram_bot_token']}/sendMessage"
            payload = {
                "chat_id": self.alerts_config["telegram_chat_id"],
                "text": text,
                "parse_mode": "Markdown"
            }
            
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                logger.info(f"Alerta enviada a Telegram: {level} - {message}")
                return True
            else:
                logger.error(f"Error al enviar alerta a Telegram: {response.status_code} - {response.text}")
                return False
            
        except Exception as e:
            logger.error(f"Error al enviar alerta a Telegram: {e}", exc_info=True)
            return False

    def _send_webhook_alert(self, level: str, message: str, details: Optional[Dict[str, Any]], 
                          alert_type: str) -> bool:
        """Envía una alerta a un webhook genérico.

        Args:
            level: Nivel de alerta
            message: Mensaje de alerta
            details: Detalles adicionales
            alert_type: Tipo de alerta

        Returns:
            bool: True si la alerta se envió correctamente
        """
        try:
            # Verificar configuración
            if not self.alerts_config["webhook_url"]:
                logger.error("URL de webhook no configurada")
                return False
            
            # Construir payload
            payload = {
                "timestamp": datetime.now().isoformat(),
                "level": level.lower(),
                "message": message,
                "alert_type": alert_type,
                "source": "trading_bot",
            }
            
            if details:
                payload["details"] = details
            
            # Enviar a webhook
            response = requests.post(
                self.alerts_config["webhook_url"],
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"Alerta enviada a webhook: {level} - {message}")
                return True
            else:
                logger.error(f"Error al enviar alerta a webhook: {response.status_code} - {response.text}")
                return False
            
        except Exception as e:
            logger.error(f"Error al enviar alerta a webhook: {e}", exc_info=True)
            return False

    def send_trade_alert(self, trade_data: Dict[str, Any]) -> bool:
        """Envía una alerta de operación.

        Args:
            trade_data: Datos de la operación

        Returns:
            bool: True si la alerta se envió correctamente
        """
        try:
            symbol = trade_data.get("symbol", "UNKNOWN")
            side = trade_data.get("side", "UNKNOWN")
            qty = trade_data.get("qty", 0)
            price = trade_data.get("price", 0)
            pnl = trade_data.get("pnl", 0)
            
            # Determinar nivel según P&L
            level = "info"
            if pnl < 0:
                if abs(pnl) > 1000:
                    level = "error"
                elif abs(pnl) > 500:
                    level = "warning"
            
            # Construir mensaje
            message = f"Operación {'COMPRA' if side.lower() == 'buy' else 'VENTA'} de {symbol}: "
            message += f"{qty} acciones a ${price:.2f}"
            
            if pnl != 0:
                message += f" (P&L: ${pnl:.2f})"
            
            return self.send_alert(
                level=level,
                message=message,
                details=trade_data,
                alert_type="trade",
                throttle=False  # No aplicar throttling a alertas de operaciones
            )
            
        except Exception as e:
            logger.error(f"Error al enviar alerta de operación: {e}", exc_info=True)
            return False

    def send_position_alert(self, position_data: Dict[str, Any]) -> bool:
        """Envía una alerta de posición.

        Args:
            position_data: Datos de la posición

        Returns:
            bool: True si la alerta se envió correctamente
        """
        try:
            symbol = position_data.get("symbol", "UNKNOWN")
            qty = position_data.get("qty", 0)
            avg_entry_price = position_data.get("avg_entry_price", 0)
            current_price = position_data.get("current_price", 0)
            unrealized_pl = position_data.get("unrealized_pl", 0)
            unrealized_plpc = position_data.get("unrealized_plpc", 0) * 100  # Convertir a porcentaje
            
            # Determinar nivel según P&L
            level = "info"
            if unrealized_plpc < -5:
                level = "error"
            elif unrealized_plpc < -2:
                level = "warning"
            elif unrealized_plpc > 5:
                level = "info"
            
            # Construir mensaje
            message = f"Posición {symbol}: {qty} acciones, "
            message += f"entrada ${avg_entry_price:.2f}, actual ${current_price:.2f}, "
            message += f"P&L: ${unrealized_pl:.2f} ({unrealized_plpc:.2f}%)"
            
            return self.send_alert(
                level=level,
                message=message,
                details=position_data,
                alert_type=f"position_{symbol}",
                throttle=True  # Aplicar throttling a alertas de posiciones
            )
            
        except Exception as e:
            logger.error(f"Error al enviar alerta de posición: {e}", exc_info=True)
            return False

    def send_risk_alert(self, risk_data: Dict[str, Any]) -> bool:
        """Envía una alerta de riesgo.

        Args:
            risk_data: Datos de riesgo

        Returns:
            bool: True si la alerta se envió correctamente
        """
        try:
            alert_type = risk_data.get("alert_type", "risk_general")
            message = risk_data.get("message", "Alerta de riesgo")
            level = risk_data.get("level", "warning")
            
            return self.send_alert(
                level=level,
                message=message,
                details=risk_data,
                alert_type=alert_type,
                throttle=risk_data.get("throttle", True)
            )
            
        except Exception as e:
            logger.error(f"Error al enviar alerta de riesgo: {e}", exc_info=True)
            return False

    def send_system_alert(self, system_data: Dict[str, Any]) -> bool:
        """Envía una alerta del sistema.

        Args:
            system_data: Datos del sistema

        Returns:
            bool: True si la alerta se envió correctamente
        """
        try:
            alert_type = system_data.get("alert_type", "system_general")
            message = system_data.get("message", "Alerta del sistema")
            level = system_data.get("level", "info")
            
            return self.send_alert(
                level=level,
                message=message,
                details=system_data,
                alert_type=alert_type,
                throttle=system_data.get("throttle", True)
            )
            
        except Exception as e:
            logger.error(f"Error al enviar alerta del sistema: {e}", exc_info=True)
            return False

    def get_alert_history(self, limit: int = 100, level: Optional[str] = None, 
                        alert_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Obtiene el historial de alertas.

        Args:
            limit: Límite de alertas a devolver
            level: Filtrar por nivel (opcional)
            alert_type: Filtrar por tipo (opcional)

        Returns:
            List[Dict[str, Any]]: Historial de alertas
        """
        try:
            # Filtrar historial
            filtered_history = self.alerts_state["alert_history"]
            
            if level:
                filtered_history = [a for a in filtered_history if a["level"] == level.lower()]
            
            if alert_type:
                filtered_history = [a for a in filtered_history if a["alert_type"] == alert_type]
            
            # Ordenar por timestamp (más reciente primero)
            sorted_history = sorted(filtered_history, key=lambda x: x["timestamp"], reverse=True)
            
            # Limitar resultados
            return sorted_history[:limit]
            
        except Exception as e:
            logger.error(f"Error al obtener historial de alertas: {e}", exc_info=True)
            return []