from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Annotated
import pandas as pd




# Pydantic Models For Task Outputs
class UserQueryClassificationResult(BaseModel):
    """
    Represents the result of classifying a user query.

    Attributes:
        needs_retrieval (bool): Indicates whether retrieval is needed.
        justification (str): Explanation for the classification decision.
    """
    needs_retrieval: bool = Field(..., description="Indicates whether retrieval is needed")
    justification: str = Field(..., description="Explanation for the classification decision")

class QueryProcessResult(BaseModel):
    """
    Represents the result of processing queries.

    Attributes:
        original_query (str): The original user query.
        transformed_queries (Optional[List[str]]): List of transformed queries, if any.
        decomposed_queries (Optional[List[str]]): List of decomposed queries, if any.
    """
    original_query: str = Field(..., description="The original user query")
    transformed_queries: Optional[List[str]] = Field(None, description="List of transformed queries, if any")
    decomposed_queries: Optional[List[str]] = Field(None, description="List of decomposed queries, if any")
    
class SubQueriesClassificationResult(BaseModel):
    """
    Represents the classification result for sub-queries.

    Attributes:
        queries (List[str]): List of sub-queries.
        collection_name (List[Optional[str]]): Corresponding collection names for each sub-query.
    """
    queries: List[str]
    collection_name: List[Optional[str]]

class TopicRerankingResult(BaseModel):
    """
    Represents the result of topic reranking.

    Attributes:
        relevant_scores (List[int]): List of relevance scores for topics.
    """
    relevant_scores: List[int]
    
class TopicSearchingResult(BaseModel):
    """
    Represents the result of topic searching.

    Attributes:
        communities_summaries (List[str]): Summaries of relevant communities.
        possible_answers (List[str]): List of possible answers.
    """
    communities_summaries: List[str]
    possible_answers: List[str]
            
class SearchTopicsAndHyDEResult(BaseModel):
    """
    Represents the comprehensive result of topic analysis.

    Attributes:
        communities_with_scores (Dict[str, Dict[str, Any]]): Communities with their scores and details.
        communities_summaries (List[str]): Summaries of relevant communities.
        possible_answers (List[str]): List of possible answers.
    """
    communities_with_scores: Dict[str, Dict[str, Any]]
    communities_summaries: List[str]
    possible_answers: List[str]
                
class DetailedSearchResult(BaseModel):
    """
    Represents the result of detailed search.

    Attributes:
        sorted_retrieved_data (List[Any]): List of detailed search results, including both retrieved data and community information.
    """
    sorted_retrieved_data: List[Any] = Field(..., description="List of detailed search results, including both retrieved data and community information")


class RetrievalResult(BaseModel):
    """
    Represents the result of content retrieval.

    Attributes:
        content (List[str]): Retrieved content.
        metadata (List[Dict[str, Any]]): Metadata associated with the retrieved content.
    """
    content: List[str]
    metadata: List[Dict[str, Any]]
    
class RerankingResult(BaseModel):
    """
    Represents the result of content reranking.

    Attributes:
        relevance_scores (List[int]): Relevance scores for the reranked content.
    """
    relevance_scores: List[int]
    
class SortedRerankingResult(BaseModel):
    """
    Represents the sorted result of content reranking.

    Attributes:
        ranked_content (List[str]): Reranked content in order of relevance.
        ranked_metadata (List[Dict[str, Any]]): Metadata for the reranked content.
        relevance_scores (List[int]): Relevance scores for the reranked content.
    """
    ranked_content: List[str]
    ranked_metadata: List[Dict[str, Any]]
    relevance_scores: List[int]
    
class AuditMetric(BaseModel):
    """
    Represents a single audit metric.

    Attributes:
        name (str): Name of the metric.
        score (int): Score of the metric (0-100).
        comment (Optional[str]): Optional comment on the metric.
    """
    name: str
    score: int = Field(..., ge=0, le=100)
    comment: Optional[str] = None

class ResponseAuditResult(BaseModel):
    """
    Represents the result of a response audit.

    Attributes:
        metrics (List[AuditMetric]): List of audit metrics.
        overall_score (int): Overall audit score (0-100).
        restart_required (bool): Indicates if a restart is required.
        additional_comments (Optional[str]): Additional comments on the audit.
    """
    metrics: List[AuditMetric]
    overall_score: int = Field(..., ge=0, le=100)
    restart_required: bool
    additional_comments: Optional[str] = None



def concat_answers(list1: List[str], new_answer:List[str]) -> List[str]:
    """
    Concatenates two lists of strings.

    Args:
        list1 (List[str]): First list of strings.
        new_answer (str): New answer to be added.
        
    Returns:
        List[str]: A new list containing all elements from list1 and the new_answer.
    """
    return list1 + new_answer

def concat_contexts(list1: List[List[Any]], new_context: List[List[Any]]) -> List[List[Any]]:
    """
    Concatenates two lists of lists of any type.

    Args:
        list1 (List[List[Any]]): First list of lists of any type.
        new_context (List[Any]): New context to be added. 
        
    Returns:
        List[List[Any]]: A new list containing all elements from list1 and the new_context.
    """
    return list1 + new_context


# Pydantic Models For State In 
class OverallState(BaseModel):
    """
    Represents the overall state of the system, including input, intermediate results, and output.

    Attributes:
        user_query (str): The user's input query.
        specific_collection (Optional[str]): Name of a specific collection, if any.
        
        user_query_classification_result (Optional[UserQueryClassificationResult]): Result of user query classification.
        query_process_result (Optional[QueryProcessResult]): Result of query processing.
        sub_queries_classification_result (Optional[SubQueriesClassificationResult]): Result of sub-queries classification.
        search_topics_and_hyde_result (Optional[SearchTopicsAndHyDEResult]): Result of topic searching.
        detailed_search_result (Optional[DetailedSearchResult]): Result of detailed search.
        information_organization_result (str): Result of information organization.
        retrieval_result (Optional[RetrievalResult]): Result of content retrieval.
        rerank_result (Optional[RerankingResult]): Result of content reranking.
        response_audit_result (Optional[ResponseAuditResult]): Result of response auditing.
        generation_result (str): The final generated result.
        repeat_times (int): Number of repetitions.
        
        dataset (Optional[pd.DataFrame]): The dataset used for processing.
        all_results (Annotated[List[str], concat_answers]): List of all results.
        all_contexts (Annotated[List[List[Any]], concat_contexts]): List of all contexts.
    """
    # Input
    user_query: Optional[str] = Field(None, description="The user's input query")
    specific_collection: Optional[str] = Field(None, description="Name of a specific collection, if any")
    
    # pydantic models
    user_query_classification_result: Optional[UserQueryClassificationResult] = Field(None, description="Result of user query classification")
    query_process_result: Optional[QueryProcessResult] = Field(None, description="Result of query processing")
    sub_queries_classification_result: Optional[SubQueriesClassificationResult] = None
    search_topics_and_hyde_result: Optional[SearchTopicsAndHyDEResult] = None
    detailed_search_result: Optional[DetailedSearchResult] = None
    information_organization_result: Optional[str] = Field(None, description="Result of information organization")
    response_audit_result: Optional[ResponseAuditResult] = None
        
    # Output
    generation_result: Optional[str] = Field(None, description="The final generated result")
    repeat_times: int = Field(0, description="Number of repetitions", ge=0)

    # Batch Dataset Processing
    dataset_queries: Optional[List[str]] = Field(None, description="List of queries from dataset")
    all_results: Optional[Annotated[List[str], concat_answers]] = []
    all_contexts: Optional[Annotated[List[List[Any]], concat_contexts]] = [[]]

class SingleState(BaseModel):
    """
    Represents a single state in the system.

    Attributes:
        user_query (str): The user's input query.
        specific_collection (Optional[str]): Name of a specific collection, if any.
        retrieved_data (Optional[List[dict]]): Retrieved data, if any.
        reranked_data (Optional[List[dict]]): Reranked data, if any.
        result (str): The result for this single state.
    """
    user_query: str
    specific_collection: Optional[str] = None
    
    # Output
    retrieved_data: Optional[List[dict]] = None
    reranked_data: Optional[List[dict]] = None
    result: str