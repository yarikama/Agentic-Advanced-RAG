from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Pydantic Models For Task Outputs
class UserQueryClassificationResult(BaseModel):
    needs_retrieval: bool
    justification: str

class QueriesProcessResult(BaseModel):
    original_query: str
    transformed_queries: Optional[List[str]] = None
    decomposed_queries: Optional[List[str]] = None
    
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
    
class SortedTopicRerankingResult(BaseModel):
    ranked_topics: List[TopicSearchingEntity]
    relevance_scores: List[int]
            
class RetrievalResult(BaseModel):
    content: List[str]
    metadata: List[Dict[str, Any]]
    
class RerankingResult(BaseModel):
    relevance_scores: List[int]
    
class SortedRerankingResult(BaseModel):
    ranked_content: List[str]
    ranked_metadata: List[Dict[str, Any]]
    relevance_scores: List[int]
    
class AuditMetric(BaseModel):
    name: str
    score: int = Field(..., ge=0, le=100)
    comment: Optional[str] = None

class ResponseAuditResult(BaseModel):
    metrics: List[AuditMetric]
    overall_score: int = Field(..., ge=0, le=100)
    restart_required: bool
    additional_comments: Optional[str] = None