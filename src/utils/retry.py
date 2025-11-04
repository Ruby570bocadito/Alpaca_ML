#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Decoradores de reintentos con backoff exponencial para operaciones externas
(Alpaca REST, llamadas HTTP, Redis, etc.), usando tenacity.
"""

import os
from typing import Iterable, Type
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
)
import logging


logger = logging.getLogger(__name__)


def _get_int(env_var: str, default: int) -> int:
    try:
        return int(os.getenv(env_var, str(default)))
    except Exception:
        return default


def retry_on_exceptions(exceptions: Iterable[Type[BaseException]] = (Exception,)):
    """Crea un decorador de reintentos con configuración vía variables de entorno.

    Variables de entorno:
    - MAX_RETRY_ATTEMPTS (default: 5)
    - BASE_BACKOFF_TIME_MS (default: 500)
    - MAX_BACKOFF_TIME_MS (default: 15000)
    - JITTER_FACTOR (default: 0.1)
    """
    max_attempts = _get_int("MAX_RETRY_ATTEMPTS", 5)
    base_ms = _get_int("BASE_BACKOFF_TIME_MS", 500)
    max_ms = _get_int("MAX_BACKOFF_TIME_MS", 15000)
    jitter = float(os.getenv("JITTER_FACTOR", "0.1"))

    # tenacity usa segundos; convertimos milisegundos a segundos
    base_seconds = max(base_ms / 1000.0, 0.001)
    max_seconds = max(max_ms / 1000.0, base_seconds)

    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential_jitter(exp_base=base_seconds, max=max_seconds, jitter=jitter),
        retry=retry_if_exception_type(tuple(exceptions)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def retry_network():
    """Decorador para reintentos genéricos de red (requests, Redis, Alpaca)."""
    return retry_on_exceptions((Exception,))