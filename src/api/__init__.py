"""
FastAPI Backend Module.

Provides REST API for:
    - Dataset management
    - Trial configuration and execution
    - Results retrieval
    - Visualization endpoints
    - Real-time status updates
"""

from src.api.main import app, run_server

__all__ = ["app", "run_server"]
