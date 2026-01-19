"""
Health Check Endpoints

Monitor service health and readiness.
"""

from fastapi import APIRouter, Depends
from datetime import datetime

from app.config import settings, Settings
from app.models.schemas import HealthResponse
from app.core.vector_store import get_vector_store
from app.core.embeddings import get_embedder
from app.api.deps import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/health", tags=["Health"])


@router.get(
    "",
    response_model=HealthResponse,
    summary="Health Check",
    description="Check if the service is running and all dependencies are available"
)
async def health_check(
    settings: Settings = Depends(get_settings)
) -> HealthResponse:
    """
    Health check endpoint.
    
    Returns:
        HealthResponse with service status
    """
    services_status = {}
    
    # Check vector store
    try:
        vector_store = get_vector_store()
        info = vector_store.get_collection_info()
        services_status["vector_store"] = "connected"
        services_status["vector_count"] = info["vectors_count"]
    except Exception as e:
        logger.error(f"Vector store health check failed: {e}")
        services_status["vector_store"] = "error"
    
    # Check embeddings
    try:
        embedder = get_embedder()
        info = embedder.get_model_info()
        services_status["embeddings"] = "ready"
        services_status["embedding_model"] = info["model_name"]
    except Exception as e:
        logger.error(f"Embeddings health check failed: {e}")
        services_status["embeddings"] = "error"
    
    # Check LLM (just verify config is present)
    if settings.OPENROUTER_API_KEY:
        services_status["llm"] = "configured"
    else:
        services_status["llm"] = "not_configured"
    
    # Overall status
    overall_status = "healthy" if all(
        v not in ["error", "not_configured"] 
        for k, v in services_status.items() 
        if k in ["vector_store", "embeddings", "llm"]
    ) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow().isoformat(),
        services=services_status
    )


@router.get(
    "/ready",
    summary="Readiness Check",
    description="Check if the service is ready to accept requests"
)
async def readiness_check() -> dict:
    """
    Readiness check for Kubernetes/container orchestration.
    
    Returns:
        Simple ready status
    """
    return {"status": "ready"}


@router.get(
    "/live",
    summary="Liveness Check",
    description="Check if the service is alive"
)
async def liveness_check() -> dict:
    """
    Liveness check for Kubernetes/container orchestration.
    
    Returns:
        Simple alive status
    """
    return {"status": "alive"}
