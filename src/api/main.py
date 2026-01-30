"""
FastAPI application entry point.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.core.logging import configure_logging, get_logger
from src.core.config import get_config
from src.api.routes import health, documents, query, collections, tasks
from src.api.middleware import logging as logging_middleware
from src.api.middleware import rate_limit as rate_limit_middleware


# Configure logging
configure_logging()
logger = get_logger(__name__)
config = get_config()


# Create FastAPI application
app = FastAPI(
    title=config.app.app_name,
    version=config.app.api_version,
    description="A simple RAG (Retrieval-Augmented Generation) system using local LLMs",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add logging middleware
logging_middleware.add_logging_middleware(app)

# Add rate limiting middleware (if enabled)
if config.security.rate_limit_enabled:
    rate_limit_middleware.add_rate_limit_middleware(app)
    logger.info("Rate limiting middleware enabled")
else:
    logger.info("Rate limiting middleware disabled")

# Include routers
app.include_router(health.router)
app.include_router(documents.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")
app.include_router(collections.router, prefix="/api/v1")
app.include_router(tasks.router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """Run on application startup."""
    logger.info("Application starting up", app_name=config.app.app_name, version=config.app.api_version)
    
    # Initialize necessary services
    # Services are lazily loaded via dependencies, so no explicit init needed


@app.on_event("shutdown")
async def shutdown_event():
    """Run on application shutdown."""
    logger.info("Application shutting down")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for unhandled exceptions.
    
    Args:
        request: Incoming request
        exc: Exception that occurred
        
    Returns:
        JSON response with error details
    """
    logger.error(
        "Unhandled exception",
        path=request.url.path,
        method=request.method,
        exception_type=type(exc).__name__,
        error=str(exc)
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error_code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "details": {
                "exception": type(exc).__name__,
                "message": str(exc)
            }
        }
    )


@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint with API information.
    
    Returns:
        API metadata
    """
    return {
        "name": config.app.app_name,
        "version": config.app.api_version,
        "description": "RAG System API",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    logger.info(
        "Starting server",
        host=config.app.app_host,
        port=config.app.app_port,
        debug=config.app.app_debug
    )
    
    uvicorn.run(
        "src.api.main:app",
        host=config.app.app_host,
        port=config.app.app_port,
        reload=config.app.app_debug,
        log_level="info" if not config.app.app_debug else "debug"
    )
