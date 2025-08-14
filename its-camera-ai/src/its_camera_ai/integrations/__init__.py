"""Integration modules for external systems.

This package contains integrations with external systems and services
including message queues, databases, and third-party APIs.
"""

from .kafka_analytics_connector import (
    KafkaAnalyticsConnector,
    create_kafka_analytics_connector,
)

__all__ = [
    "KafkaAnalyticsConnector",
    "create_kafka_analytics_connector",
]
