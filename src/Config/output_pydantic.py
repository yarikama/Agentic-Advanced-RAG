from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

__all__ = ["UserQueryClassification", "Queries", "QueriesIdentification", "RefinedRetrievalData", "RankedRetrievalData", "AuditMetric", "AuditResult", "UpdateCondition"]

# Pydantic Models For Task Outputs
# For Classifier
class UserQueryClassification(BaseModel):
    needs_retrieval: bool
    justification: str

# For Query Processor
class Queries(BaseModel):
    original_query: str
    transformed_query: Optional[str] = None
    decomposed_queries: Optional[List[str]] = None
    
# For Classifier (Classification)
class QueriesIdentification(BaseModel):
    queries: List[str]
    collection_name: List[Optional[str]]
            
# For Retriever (Retrieval)
class RefinedRetrievalData(BaseModel):
    content: List[str]
    metadata: List[Dict[str, Any]]
    
# For Reranker
class RankedRetrievalData(BaseModel):
    ranked_content: List[str]
    ranked_metadata: List[Dict[str, Any]]
    relevance_scores: List[float]
    
# For Response Auditor
class AuditMetric(BaseModel):
    name: str
    score: float = Field(..., ge=0, le=1)
    comment: Optional[str] = None

class AuditResult(BaseModel):
    metrics: List[AuditMetric]
    overall_score: float = Field(..., ge=0, le=1)
    restart_required: bool
    additional_comments: Optional[str] = None
    
class UpdateCondition(BaseModel):
    is_database_updated: bool
    reason: str
    

# class RetrievalData(BaseModel):
#     query: str
#     needs_retrieval: bool
#     collection_name: Optional[str] = None
#     retrieved_metadata: Optional[Dict[str, Any]] = None
#     retrieved_content: Optional[str] = None