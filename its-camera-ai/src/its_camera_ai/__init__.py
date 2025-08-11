"""ITS Camera AI - Real-time traffic monitoring and analytics system.

A production-grade AI system for intelligent traffic monitoring using computer vision
and machine learning to analyze traffic patterns, detect vehicles, and monitor road conditions.
"""

__version__ = "0.1.0"
__author__ = "ITS Camera AI Team"
__email__ = "team@its-camera-ai.com"

# Core exports
from .core.config import Settings, get_settings
from .core.exceptions import ITSCameraAIError
from .core.logging import setup_logging

__all__ = [
    "__version__",
    "Settings",
    "get_settings",
    "ITSCameraAIError",
    "setup_logging",
]
