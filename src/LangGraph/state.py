from LLMMA.output_pydantic import *
from typing import Optional, TypedDict
from pydantic import BaseModel, Field

class OverallState(TypedDict):
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
    update_database: Optional[bool] = None