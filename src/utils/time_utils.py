#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilidades para el manejo de tiempo y fechas en el sistema de trading.
"""

import datetime
import pytz
import pandas as pd
from typing import Optional, Union, Tuple, List
import time


def get_current_time(timezone: str = "UTC") -> datetime.datetime:
    """Obtiene la hora actual en la zona horaria especificada.

    Args:
        timezone: Zona horaria (por defecto UTC)

    Returns:
        datetime.datetime: Hora actual
    """
    return datetime.datetime.now(pytz.timezone(timezone))


def get_current_time_str(format_str: str = "%Y-%m-%d %H:%M:%S", timezone: str = "UTC") -> str:
    """Obtiene la hora actual como string formateado.

    Args:
        format_str: Formato de fecha/hora
        timezone: Zona horaria

    Returns:
        str: Hora actual formateada
    """
    return get_current_time(timezone).strftime(format_str)


def is_market_hours(dt: Optional[datetime.datetime] = None, 
                   timezone: str = "America/New_York") -> bool:
    """Verifica si la hora dada está dentro del horario de mercado (9:30-16:00 ET).

    Args:
        dt: Fecha y hora a verificar (por defecto, hora actual)
        timezone: Zona horaria

    Returns:
        bool: True si está dentro del horario de mercado
    """
    if dt is None:
        dt = get_current_time(timezone)
    elif dt.tzinfo is None:
        dt = pytz.timezone(timezone).localize(dt)
    
    # Convertir a timezone del mercado si es necesario
    if dt.tzinfo.zone != timezone:
        dt = dt.astimezone(pytz.timezone(timezone))
    
    # Verificar si es día de semana (0=lunes, 6=domingo)
    if dt.weekday() >= 5:  # Sábado o domingo
        return False
    
    # Verificar horario (9:30 - 16:00)
    market_open = datetime.time(9, 30)
    market_close = datetime.time(16, 0)
    current_time = dt.time()
    
    return market_open <= current_time <= market_close


def get_next_market_open(dt: Optional[datetime.datetime] = None, 
                        timezone: str = "America/New_York") -> datetime.datetime:
    """Obtiene la próxima apertura del mercado.

    Args:
        dt: Fecha y hora de referencia (por defecto, hora actual)
        timezone: Zona horaria

    Returns:
        datetime.datetime: Próxima apertura del mercado
    """
    if dt is None:
        dt = get_current_time(timezone)
    elif dt.tzinfo is None:
        dt = pytz.timezone(timezone).localize(dt)
    
    # Convertir a timezone del mercado si es necesario
    if dt.tzinfo.zone != timezone:
        dt = dt.astimezone(pytz.timezone(timezone))
    
    tz = pytz.timezone(timezone)
    
    # Si es antes de la apertura hoy, la próxima apertura es hoy
    if dt.time() < datetime.time(9, 30) and dt.weekday() < 5:
        next_open = tz.localize(
            datetime.datetime.combine(dt.date(), datetime.time(9, 30))
        )
        return next_open
    
    # Avanzar al siguiente día
    days_to_add = 1
    if dt.weekday() >= 4:  # Viernes o fin de semana
        days_to_add = 7 - dt.weekday()
    
    next_date = dt.date() + datetime.timedelta(days=days_to_add)
    
    # Ajustar si cae en fin de semana
    while next_date.weekday() >= 5:
        next_date += datetime.timedelta(days=1)
    
    next_open = tz.localize(
        datetime.datetime.combine(next_date, datetime.time(9, 30))
    )
    
    return next_open


def get_next_market_close(dt: Optional[datetime.datetime] = None, 
                         timezone: str = "America/New_York") -> datetime.datetime:
    """Obtiene el próximo cierre del mercado.

    Args:
        dt: Fecha y hora de referencia (por defecto, hora actual)
        timezone: Zona horaria

    Returns:
        datetime.datetime: Próximo cierre del mercado
    """
    if dt is None:
        dt = get_current_time(timezone)
    elif dt.tzinfo is None:
        dt = pytz.timezone(timezone).localize(dt)
    
    # Convertir a timezone del mercado si es necesario
    if dt.tzinfo.zone != timezone:
        dt = dt.astimezone(pytz.timezone(timezone))
    
    tz = pytz.timezone(timezone)
    
    # Si es antes del cierre hoy y es día de semana, el próximo cierre es hoy
    if dt.time() < datetime.time(16, 0) and dt.weekday() < 5:
        next_close = tz.localize(
            datetime.datetime.combine(dt.date(), datetime.time(16, 0))
        )
        return next_close
    
    # Avanzar al siguiente día
    days_to_add = 1
    if dt.weekday() >= 4:  # Viernes o fin de semana
        days_to_add = 7 - dt.weekday()
    
    next_date = dt.date() + datetime.timedelta(days=days_to_add)
    
    # Ajustar si cae en fin de semana
    while next_date.weekday() >= 5:
        next_date += datetime.timedelta(days=1)
    
    next_close = tz.localize(
        datetime.datetime.combine(next_date, datetime.time(16, 0))
    )
    
    return next_close


def get_trading_days(start_date: Union[str, datetime.date, datetime.datetime],
                    end_date: Union[str, datetime.date, datetime.datetime],
                    timezone: str = "America/New_York") -> List[datetime.date]:
    """Obtiene una lista de días de trading entre las fechas dadas.

    Args:
        start_date: Fecha de inicio
        end_date: Fecha de fin
        timezone: Zona horaria

    Returns:
        List[datetime.date]: Lista de días de trading
    """
    # Convertir a datetime.date si es necesario
    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date).date()
    elif isinstance(start_date, datetime.datetime):
        start_date = start_date.date()
    
    if isinstance(end_date, str):
        end_date = pd.Timestamp(end_date).date()
    elif isinstance(end_date, datetime.datetime):
        end_date = end_date.date()
    
    # Generar rango de fechas
    all_days = pd.date_range(start=start_date, end=end_date)
    
    # Filtrar solo días de semana (lunes a viernes)
    trading_days = [day.date() for day in all_days if day.weekday() < 5]
    
    # Nota: Esta es una implementación básica que no tiene en cuenta feriados
    # Para una implementación completa, se debería usar un calendario de mercado
    # como pandas_market_calendars o exchange_calendars
    
    return trading_days


def time_since(start_time: float) -> float:
    """Calcula el tiempo transcurrido desde start_time en milisegundos.

    Args:
        start_time: Tiempo de inicio (time.time())

    Returns:
        float: Tiempo transcurrido en milisegundos
    """
    return (time.time() - start_time) * 1000


def format_timeframe(timeframe: str) -> Tuple[int, str]:
    """Formatea un string de timeframe a una tupla (valor, unidad).

    Args:
        timeframe: String de timeframe (ej: "1m", "1h", "1d")

    Returns:
        Tuple[int, str]: Tupla (valor, unidad)

    Raises:
        ValueError: Si el formato no es válido
    """
    if not timeframe or not isinstance(timeframe, str):
        raise ValueError(f"Formato de timeframe inválido: {timeframe}")
    
    # Extraer valor y unidad
    for i, char in enumerate(timeframe):
        if not char.isdigit():
            value = int(timeframe[:i])
            unit = timeframe[i:]
            break
    else:
        raise ValueError(f"Formato de timeframe inválido: {timeframe}")
    
    # Validar unidad
    valid_units = {"s": "second", "m": "minute", "h": "hour", "d": "day", "w": "week"}
    if unit not in valid_units:
        raise ValueError(f"Unidad de timeframe inválida: {unit}")
    
    return value, unit


def timeframe_to_seconds(timeframe: str) -> int:
    """Convierte un string de timeframe a segundos.

    Args:
        timeframe: String de timeframe (ej: "1m", "1h", "1d")

    Returns:
        int: Segundos

    Raises:
        ValueError: Si el formato no es válido
    """
    value, unit = format_timeframe(timeframe)
    
    # Convertir a segundos
    multipliers = {
        "s": 1,
        "m": 60,
        "h": 3600,
        "d": 86400,
        "w": 604800
    }
    
    return value * multipliers[unit]


def timeframe_to_timedelta(timeframe: str) -> datetime.timedelta:
    """Convierte un string de timeframe a timedelta.

    Args:
        timeframe: String de timeframe (ej: "1m", "1h", "1d")

    Returns:
        datetime.timedelta: Timedelta

    Raises:
        ValueError: Si el formato no es válido
    """
    seconds = timeframe_to_seconds(timeframe)
    return datetime.timedelta(seconds=seconds)


def align_time_to_timeframe(dt: datetime.datetime, timeframe: str) -> datetime.datetime:
    """Alinea un datetime al inicio del período de timeframe.

    Args:
        dt: Datetime a alinear
        timeframe: String de timeframe (ej: "1m", "1h", "1d")

    Returns:
        datetime.datetime: Datetime alineado

    Raises:
        ValueError: Si el formato no es válido
    """
    value, unit = format_timeframe(timeframe)
    
    if unit == "s":
        seconds = dt.second - (dt.second % value)
        return dt.replace(second=seconds, microsecond=0)
    
    elif unit == "m":
        minutes = dt.minute - (dt.minute % value)
        return dt.replace(minute=minutes, second=0, microsecond=0)
    
    elif unit == "h":
        hours = dt.hour - (dt.hour % value)
        return dt.replace(hour=hours, minute=0, second=0, microsecond=0)
    
    elif unit == "d":
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)
    
    elif unit == "w":
        # Alinear al lunes de la semana
        days_to_subtract = dt.weekday()
        return (dt - datetime.timedelta(days=days_to_subtract)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
    
    raise ValueError(f"Unidad de timeframe no soportada: {unit}")


def get_execution_time_str(start_time: float) -> str:
    """Obtiene un string formateado con el tiempo de ejecución.

    Args:
        start_time: Tiempo de inicio (time.time())

    Returns:
        str: Tiempo de ejecución formateado
    """
    elapsed_ms = time_since(start_time)
    
    if elapsed_ms < 1000:
        return f"{elapsed_ms:.2f} ms"
    
    elapsed_sec = elapsed_ms / 1000
    if elapsed_sec < 60:
        return f"{elapsed_sec:.2f} seg"
    
    elapsed_min = elapsed_sec / 60
    if elapsed_min < 60:
        return f"{int(elapsed_min)}m {int(elapsed_sec % 60)}s"
    
    elapsed_hour = elapsed_min / 60
    return f"{int(elapsed_hour)}h {int(elapsed_min % 60)}m {int(elapsed_sec % 60)}s"