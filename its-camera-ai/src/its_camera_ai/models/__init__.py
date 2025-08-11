"""Data models for ITS Camera AI system."""

from .base import BaseModel
from .database import get_database_session

__all__ = ["BaseModel", "get_database_session"]
