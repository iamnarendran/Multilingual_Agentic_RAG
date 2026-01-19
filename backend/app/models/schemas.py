"""
Pydantic Schemas for Request/Response Validation

All API endpoints use these models for input validation and response formatting.
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

from app.models.enums import QueryType, DocumentType, LanguageCode


# =============================================================================
# QUERY SCHEMAS
# =============================================================================

class QueryRequest(BaseModel):
    """Request model for query endpoint"""
    
    query: str = Field(
        ...,
        description="User query in any supported language",
        min_length=1,
        max_length=1000,
        example="What is the capital of India?"
    )
    
    language: Optional[str] = Field(
        "auto",
        description="Query language (auto-detect if not specified)",
        example="en"
    )
    
    filters: Optional[Dict[str, Any]] = Field(
        None,
        description="Optional metadata filters for search",
        example={"category": "geography"}
    )
    
    top_k: Optional[int] = Field(
        5,
        description="Number of source documents to retrieve",
        ge=1,
        le=20
    )
    
    include_sources: bool = Field(
        True,
        description="Include source documents in response"
    )
    
    @validator("query")
    def validate_query(cls, v):
        """Validate query is not empty or just whitespace"""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


class Citation(BaseModel):
    """Citation information"""
    doc_id: str
    position: int
    text: str


class Source(BaseModel):
    """Source document information"""
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class QueryResponse(BaseModel):
    """Response model for query endpoint"""
    
    answer: str = Field(
        ...,
        description="Generated answer with citations"
    )
    
    confidence: float = Field(
        ...,
        description="Confidence score (0-1)",
        ge=0.0,
        le=1.0
    )
    
    language: str = Field(
        ...,
        description="Detected/used language",
        example="en"
    )
    
    query_type: str = Field(
        ...,
        description="Classified query type",
        example="SIMPLE_QA"
    )
    
    citations: List[Citation] = Field(
        default_factory=list,
        description="List of citations in the answer"
    )
    
    sources: Optional[List[Source]] = Field(
        None,
        description="Source documents used (if requested)"
    )
    
    metadata: Dict[str, Any] = Field(
        ...,
        description="Processing metadata (time, cost, etc.)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "India's capital is New Delhi [Doc ID: 123].",
                "confidence": 0.95,
                "language": "en",
                "query_type": "SIMPLE_QA",
                "citations": [
                    {"doc_id": "123", "position": 25, "text": "[Doc ID: 123]"}
                ],
                "metadata": {
                    "processing_time": 2.5,
                    "total_cost": 0.0045,
                    "documents_analyzed": 3
                }
            }
        }


# =============================================================================
# DOCUMENT SCHEMAS
# =============================================================================

class DocumentUploadResponse(BaseModel):
    """Response model for document upload"""
    
    document_id: str = Field(
        ...,
        description="Unique document identifier"
    )
    
    filename: str = Field(
        ...,
        description="Original filename"
    )
    
    chunks_created: int = Field(
        ...,
        description="Number of chunks created from document"
    )
    
    language: str = Field(
        ...,
        description="Detected document language"
    )
    
    message: str = Field(
        ...,
        description="Status message",
        example="Document uploaded and processed successfully"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "document_id": "abc123",
                "filename": "report.pdf",
                "chunks_created": 15,
                "language": "en",
                "message": "Document uploaded and processed successfully"
            }
        }


class DocumentInfo(BaseModel):
    """Document information"""
    
    document_id: str
    filename: str
    language: str
    chunks_count: int
    uploaded_at: str
    file_type: str
    size_bytes: Optional[int] = None


class DocumentListResponse(BaseModel):
    """Response model for listing documents"""
    
    documents: List[DocumentInfo]
    total: int
    page: int = 1
    page_size: int = 50
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "document_id": "abc123",
                        "filename": "report.pdf",
                        "language": "en",
                        "chunks_count": 15,
                        "uploaded_at": "2024-01-15T10:30:00",
                        "file_type": "pdf"
                    }
                ],
                "total": 1,
                "page": 1,
                "page_size": 50
            }
        }


# =============================================================================
# HEALTH & SYSTEM SCHEMAS
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response"""
    
    status: str = Field(
        ...,
        description="Service status",
        example="healthy"
    )
    
    version: str = Field(
        ...,
        description="API version",
        example="1.0.0"
    )
    
    timestamp: str = Field(
        ...,
        description="Current server time"
    )
    
    services: Dict[str, str] = Field(
        ...,
        description="Status of dependent services"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "timestamp": "2024-01-15T10:30:00Z",
                "services": {
                    "vector_store": "connected",
                    "embeddings": "ready",
                    "llm": "available"
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response model"""
    
    error: str = Field(
        ...,
        description="Error type"
    )
    
    message: str = Field(
        ...,
        description="Error message"
    )
    
    details: Optional[Any] = Field(
        None,
        description="Additional error details"
    )
    
    timestamp: str = Field(
        ...,
        description="Error timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Query cannot be empty",
                "details": None,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }


# =============================================================================
# STATISTICS SCHEMAS
# =============================================================================

class AgentStats(BaseModel):
    """Statistics for a single agent"""
    agent_name: str
    calls: int
    total_tokens: int
    total_cost: float
    total_time: float
    errors: int


class SystemStats(BaseModel):
    """Overall system statistics"""
    
    total_queries: int
    total_documents: int
    total_cost: float
    avg_processing_time: float
    agents: List[AgentStats]
