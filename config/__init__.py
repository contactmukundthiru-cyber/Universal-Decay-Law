"""
Configuration module for the Universal Decay Law project.

This module provides centralized configuration management using pydantic-settings,
ensuring type-safe access to environment variables and configuration parameters.

Example:
    >>> from config import settings
    >>> print(settings.database.url)
    >>> print(settings.analysis.default_decay_model)
"""

from config.settings import (
    Settings,
    DatabaseSettings,
    RedisSettings,
    APIKeysSettings,
    AnalysisSettings,
    ServerSettings,
    get_settings,
    settings,
)

__all__ = [
    "Settings",
    "DatabaseSettings",
    "RedisSettings",
    "APIKeysSettings",
    "AnalysisSettings",
    "ServerSettings",
    "get_settings",
    "settings",
]
