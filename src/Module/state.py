from Config.output_pydantic import *
from typing import Optional, TypedDict, List
from pydantic import BaseModel, Field

class OverallState(BaseModel):
    # Input
    user_query: str
    specific_collection: Optional[str] = None
    
    # pydantic models
    user_query_classification_result: Optional[UserQueryClassificationResult] = None
    queries_process_result: Optional[QueriesProcessResult] = None
    sub_queries_classification_result: Optional[SubQueriesClassificationResult] = None
    topic_searching_result: Optional[TopicSearchingResult] = None
    retrieval_result: Optional[RetrievalResult] = None
    rerank_result: Optional[RerankingResult] = None
    response_audit_result: Optional[ResponseAuditResult] = None
        
    # Output
    generation_result: str = None
    repeat_times: int = 0

class SingleState(BaseModel):
    user_query: str
    specific_collection: Optional[str] = None
    
    # Output
    retrived_data: Optional[List[dict]] = None
    reranked_data: Optional[List[dict]] = None
    result = str