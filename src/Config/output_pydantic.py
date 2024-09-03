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

class TopicRerankingResult(BaseModel):
    relevant_scores: List[int]
    
class TopicSearchingResult(BaseModel):
    community_summaries: List[str]
    possible_answers: List[str]
            
class TopicResult(BaseModel):
    communities_with_scores : Dict[str, int]
    communities_summaries: List[str]
    possible_answers: List[str]
                
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