"""Backend integration components for CLI application.

Provides comprehensive integration with all backend services including:
- API client for service management
- Database connection management
- Service discovery and health checks
- Real-time event streaming
- Authentication integration
- Background task management
- Performance monitoring
"""

from .api_client import APIClient
from .auth_manager import CLIAuthManager as AuthManager
from .database_manager import CLIDatabaseManager as DatabaseManager
from .event_streamer import EventStreamer
from .health_checker import HealthChecker
from .metrics_collector import MetricsCollector
from .orchestrator import ServiceOrchestrator
from .queue_manager import CLIQueueManager as QueueManager
from .service_discovery import ServiceDiscovery

__all__ = [
    "APIClient",
    "AuthManager",
    "DatabaseManager",
    "EventStreamer",
    "HealthChecker",
    "QueueManager",
    "ServiceDiscovery",
    "MetricsCollector",
    "ServiceOrchestrator",
]
