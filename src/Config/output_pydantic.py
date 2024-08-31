from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any




# Pydantic Models For Task Outputs
# For Classifier
class UserQueryClassificationResult(BaseModel):
    needs_retrieval: bool
    justification: str

# For Query Processor
class QueriesProcessResult(BaseModel):
    original_query: str
    transformed_queries: Optional[List[str]] = None
    decomposed_queries: Optional[List[str]] = None
    
# For Classifier (Classification)
class SubQueriesClassificationResult(BaseModel):
    queries: List[str]
    collection_name: List[Optional[str]]

class TopicSearchingEntity(BaseModel):
    description: str
    score: int
    example_sentence: str
    
class TopicSearchingResult(BaseModel):
    topics: List[TopicSearchingEntity]
            
class TopicRerankingResult(BaseModel):
    relevant_scores: List[int]
            
# For Retriever (Retrieval)
class RetrievalResult(BaseModel):
    content: List[str]
    metadata: List[Dict[str, Any]]
    
# For Reranker
class RerankingResult(BaseModel):
    ranked_content: List[str]
    ranked_metadata: List[Dict[str, Any]]
    relevance_scores: List[int]
    
# For Response Auditor
class AuditMetric(BaseModel):
    name: str
    score: float = Field(..., ge=0, le=1)
    comment: Optional[str] = None

class ResponseAuditResult(BaseModel):
    metrics: List[AuditMetric]
    overall_score: float = Field(..., ge=0, le=1)
    restart_required: bool
    additional_comments: Optional[str] = None