from pydantic import BaseModel
from typing import Optional
from LLMMA.output_pydantic import *

class OverallState(BaseModel):
    # Input
    user_query: str
    specific_collection: Optional[str] = None
    
    # pydantic models
    user_query_classification: Optional[UserQueryClassification] = None
    queries: Optional[Queries] = None
    queries_identification_list: Optional[QueriesIdentificationList] = None
    refined_retrieval_data: Optional[RefinedRetrievalData] = None
    ranked_retrieval_data: Optional[RankedRetrievalData] = None
    audit_result: Optional[AuditResult] = None
    
    # Output
    result: str = None
    repeat_times: int = 0