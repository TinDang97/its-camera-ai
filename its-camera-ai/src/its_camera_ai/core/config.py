"""Configuration management for ITS Camera AI system.

This module provides backward compatibility for the refactored configuration structure.
All configuration classes are now organized in the configs/ subfolder for better modularity.
"""

# Re-export all configuration classes from the modular structure
from .configs import (
    CompressionConfig,
    DatabaseConfig,
    MinIOConfig,
    MLConfig,
    MLStreamingConfig,
    MonitoringConfig,
    PerformanceConfig,
    RateLimit,
    RedisConfig,
    RedisQueueConfig,
    SecurityConfig,
    Settings,
    SSEStreamingConfig,
    StreamingConfig,
    get_settings,
    get_settings_for_testing,
)

__all__ = [
    "Settings",
    "get_settings",
    "get_settings_for_testing",
    "DatabaseConfig",
    "RedisConfig",
    "RedisQueueConfig",
    "MLConfig",
    "MLStreamingConfig",
    "MonitoringConfig",
    "SecurityConfig",
    "MinIOConfig",
    "StreamingConfig",
    "SSEStreamingConfig",
    "PerformanceConfig",
    "CompressionConfig",
    "RateLimit",
]
