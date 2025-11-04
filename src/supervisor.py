#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import logging
from typing import Optional, Dict, Any
import redis
from redis.exceptions import RedisError

from src.data.store import PersistenceStore
from src.utils.logging import setup_logging, get_logger


logger = logging.getLogger(__name__)


class QueueSupervisor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        setup_logging(config)
        self.log = get_logger()
        self.redis_url = config.get("REDIS_URL")
        self.redis = None
        if self.redis_url:
            try:
                self.redis = redis.from_url(self.redis_url)
                self.log.info("Supervisor conectado a Redis", extra={"component": "supervisor"})
            except RedisError:
                self.log.warning("No se pudo conectar a Redis", extra={"component": "supervisor"})
        self.store = PersistenceStore(config.get("PERSISTENCE_DB", "data/persistence.db"))
        self.alert_webhook = config.get("ALERT_WEBHOOK_URL")

    def queue_depth(self, queue_name: str) -> Optional[int]:
        if not self.redis:
            return None
        try:
            return self.redis.llen(f"rq:queue:{queue_name}")
        except RedisError:
            return None

    def check(self):
        metrics = {}
        for q in ["signals", "orders", "predictions", "training"]:
            depth = self.queue_depth(q)
            metrics[f"queue_{q}_depth"] = depth
            if depth is not None and depth > self.config.get("QUEUE_ALERT_THRESHOLD", 100):
                self.log.warning(
                    f"Cola {q} con profundidad alta", extra={"queue": q, "depth": depth}
                )

        recent_failed = [j for j in self.store.list_recent_jobs(50) if j["status"] == "failed"]
        if recent_failed:
            self.log.error("Jobs fallidos detectados", extra={"count": len(recent_failed)})

        return metrics

    def run(self):
        interval = int(self.config.get("SUPERVISOR_INTERVAL", 30))
        self.log.info("Iniciando supervisor", extra={"interval": interval})
        while True:
            try:
                metrics = self.check()
                self.log.metric("supervisor", metrics) if hasattr(self.log, "metric") else self.log.info("metrics", extra=metrics)
            except Exception as e:
                self.log.error("Error en supervisor", extra={"error": str(e)})
            time.sleep(interval)


def main():
    config = {
        "REDIS_URL": os.environ.get("REDIS_URL"),
        "PERSISTENCE_DB": os.environ.get("PERSISTENCE_DB", "data/persistence.db"),
        "SUPERVISOR_INTERVAL": os.environ.get("SUPERVISOR_INTERVAL", "30"),
        "QUEUE_ALERT_THRESHOLD": int(os.environ.get("QUEUE_ALERT_THRESHOLD", "100")),
        "LOG_FORMAT": os.environ.get("LOG_FORMAT", "json"),
        "USE_STRUCTLOG": os.environ.get("USE_STRUCTLOG", "true"),
    }

    supervisor = QueueSupervisor(config)
    supervisor.run()


if __name__ == "__main__":
    main()