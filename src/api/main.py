"""
FastAPI Application Entry Point.

Main application setup with middleware, exception handlers, and route registration.
"""

from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config.settings import settings
from database.connection import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Runs on startup and shutdown.
    """
    # Startup
    await init_db()
    yield
    # Shutdown
    pass


# Create FastAPI application
app = FastAPI(
    title="Universal Decay Law API",
    description="""
    API for the Universal Decay Law of Human Engagement research project.

    This API provides endpoints for:
    - Managing datasets from various platforms
    - Configuring and running analysis trials
    - Retrieving results and visualizations
    - Real-time monitoring of analysis progress

    The Universal Decay Law posits that engagement decay across all digital platforms
    follows a single universal function when properly normalized by motivation parameters.
    """,
    version=settings.version,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.server.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled errors."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.server.debug else "An unexpected error occurred"
        }
    )


# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.version,
        "environment": settings.environment
    }


# Register routers
from src.api.routes import datasets, trials, analysis, visualization

app.include_router(datasets.router, prefix="/api/datasets", tags=["Datasets"])
app.include_router(trials.router, prefix="/api/trials", tags=["Trials"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["Analysis"])
app.include_router(visualization.router, prefix="/api/visualization", tags=["Visualization"])


def run_server():
    """Run the API server."""
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.debug,
        workers=settings.server.workers if not settings.server.debug else 1
    )


if __name__ == "__main__":
    run_server()
