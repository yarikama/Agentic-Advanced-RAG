from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Annotated, Tuple
import pandas as pd
import config.constants as const

# Pydantic Models For LLM Structure Outputs
class HyDEOutput(BaseModel):
    """
    Represents the output of the HyDE model.

    Attributes:
        possible_answers (List[str]): List of possible answers.
    """
    possible_answers: List[str]
    

# Pydantic Models For Task Outputs
class UserQueryClassificationResult(BaseModel):
    """
    Represents the result of classifying a user query.

    Attributes:
        needs_retrieval (bool): Indicates whether retrieval is needed.
        domain_range_score (int): Score for the domain range.
        justification (str): Explanation for the classification decision.
    """
    needs_retrieval: bool = Field(..., description="Indicates whether retrieval is needed")
    domain_range_score: int = Field(..., description="Score for the domain range", ge=0, le=100)
    justification: str = Field(..., description="Explanation for the classification decision")
    relevant_keywords: List[str] = Field(..., description="List of relevant keywords")


class GlobalMappingResult(BaseModel):
    """
    Represents the result of global mapping.

    Attributes:
        communities_summaries (List[str]): Summaries of relevant communities.
        possible_answers (List[str]): List of possible answers.
    """
    communities_summaries: Optional[List[str]] = Field([], description="Summaries of relevant communities")
    possible_answers: Optional[List[str]] = Field([], description="List of possible answers")


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

    
class GlobalTopicSearchingResult(BaseModel):
    """
    Represents the result of topic searching.

    Attributes:
        communities_summaries (List[str]): Summaries of relevant communities.
        possible_answers (List[str]): List of possible answers.
    """
    communities_summaries: List[str]
    possible_answers: List[str]

            
class LocalTopicSearchingResult(BaseModel):
    """
    Represents the result of local topic analysis.

    Attributes:
        information_summaries (List[str]): Summaries of relevant information.
        possible_answers (List[str]): List of possible answers.
    """
    information_summaries: List[str]
    possible_answers: List[str]
    

class GlobalTopicSearchingAndHyDEResult(BaseModel):
    """
    Represents the result of global topic analysis.

    Attributes:
        communities_with_scores (List[Tuple[str, int]]): Communities with their scores.
        communities_summaries (List[str]): Summaries of relevant communities.
        possible_answers (List[str]): List of possible answers.
    """
    communities_with_scores: List[Tuple[str, int]]
    communities_summaries: List[str]
    possible_answers: List[str]
                
                
class LocalTopicSearchingAndHyDEResult(BaseModel):
    """
    Represents the result of local topic analysis.

    Attributes:
        information_with_scores (List[Tuple[str, int]]): Information with their scores.
        information_summaries (List[str]): Summaries of relevant information.
        possible_answers (List[str]): List of possible answers.
    """
    information_with_scores: List[Tuple[str, int]]
    information_summaries: List[str]
    possible_answers: List[str]
                
class DetailedSearchResult(BaseModel):
    """
    Represents the result of detailed search.

    Attributes:
        all_sorted_retrieved_data (List[Any]): List of detailed search results, including both retrieved data and community information.
        all_detail_data (List[str]): List of detailed search results.
    """
    all_sorted_retrieved_data: List[Any] = Field(..., description="List of detailed search results, including both retrieved data and community information")
    all_detail_data: List[str] = Field(..., description="List of detailed search results")
    all_topic_data: List[str] = Field(..., description="List of topic search results")

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


# Pydantic Models For State
class MappingState(BaseModel):
    """
    Represents the state for reranking.

    Attributes:
        - user_query (str): The user's input query.
        - sub_queries (List[str]): List of sub-queries.
        - batch_size (int): Number of communities or information to rerank.
        - batch_data (List[str]): List of communities or information to rerank.
    """
    user_query: str
    sub_queries: Optional[List[str]]
    batch_data: List[str]
    
    
class InformationOrganizationResult(BaseModel):
    """
    Represents the result of information organization.

    Attributes:
        organized_information (List[str]): List of organized information.
    """
    organized_information: List[str]


def concate_or_reset(input_list: List[Tuple[str, int]], new_list: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    if new_list == const.RESET_LIST:
        return []
    else:
        return input_list + new_list

def concate_mapping_result(input_mapping_result: Dict[str, List], new_mapping_result: Dict[str, List]):
    if new_mapping_result == const.RESET_LIST:
        return None
    if not input_mapping_result and not new_mapping_result:
        return None
    if input_mapping_result is None:
        return new_mapping_result
    if new_mapping_result is None:
        return input_mapping_result
    return {
        "communities_summaries": input_mapping_result.get("communities_summaries", []) + new_mapping_result.get("communities_summaries", []), 
        "possible_answers": input_mapping_result.get("possible_answers", []) + new_mapping_result.get("possible_answers", [])
    }

def concate_twoD_list(list_of_lists: List[List[str]], new_list: List[str]) -> List[List[str]]:
    list_of_lists.append(new_list)
    return list_of_lists

def append_information_context(list_of_lists: List[List[str]], new_list: List[str]) -> List[List[str]]:
    if new_list:
        return list_of_lists + [new_list]
    return list_of_lists + [[]]

def append_vectordatabase_context(list_of_lists: List[List[Dict]], new_list: List[str]) -> List[List[Dict]]:
    if new_list:
        new_list = [{"text": i} for i in new_list]
        return list_of_lists + [new_list]
    return list_of_lists + [[]]

def append_graphdatabase_context(list_of_lists: List[List[str]], new_list: List[str]) -> List[List[str]]:
    if new_list:
        return list_of_lists + [new_list]
    return list_of_lists + [[]]

def append_generation_result(list_of_results: List[str], new_result: str) -> List[str]:
    if new_result:
        return list_of_results + [new_result]
    return list_of_results + [""]


class UnitQueryState(BaseModel):
    """
    Attributes:
        query_id (Optional[str]): The id of the query.
        user_query (Optional[str]): The user's input query.
        specific_collection (Optional[str]): Name of a specific collection, if any.
        
        user_query_classification_result (Optional[UserQueryClassificationResult]): Result of user query classification.
        query_process_result (Optional[QueryProcessResult]): Result of query processing.
        sub_queries_classification_result (Optional[SubQueriesClassificationResult]): Result of sub-queries classification.
        
        global_topic_searching_and_hyde_result (Optional[GlobalTopicSearchingAndHyDEResult]): Result of topic searching.
        local_topic_searching_and_hyde_result (Optional[LocalTopicSearchingAndHyDEResult]): Result of topic searching.
        detailed_search_result (Optional[DetailedSearchResult]): Result of detailed search.
        information_organization_result (Optional[InformationOrganizationResult]): Result of information organization.
        response_audit_result (Optional[ResponseAuditResult]): Result of response auditing.
        
        retrieval_data (Optional[List[str]]): Retrieved data.
        
        global_mapping_result (Annotated[List[Tuple[str, int]], concate_or_reset]): Result of global reranking.
        local_mapping_result (Annotated[List[Tuple[str, int]], concate_or_reset]): Result of local reranking.
        detail_mapping_result (Annotated[List[Tuple[str, int]], concate_or_reset]): Result of detail reranking.
        
        generation_result (Optional[str]): The final generated result.
    """
    # Settings
    query_id:                                   Optional[int]                                        = Field(None, description="The id of the query")
    user_query:                                 Optional[str]                                        = Field(None, description="The user's input query")
    specific_collection:                        Optional[str]                                        = Field(None, description="Name of a specific collection, if any")
    # pydantic models        
    user_query_classification_result:           Optional[UserQueryClassificationResult]              = Field(None, description="Result of user query classification")
    query_process_result:                       Optional[QueryProcessResult]                         = Field(None, description="Result of query processing")
    sub_queries_classification_result:          Optional[SubQueriesClassificationResult]             = Field(None, description="Result of sub-queries classification")
    global_topic_searching_and_hyde_result:     Optional[GlobalTopicSearchingAndHyDEResult]          = Field(None, description="Result of topic searching")
    local_topic_searching_and_hyde_result:      Optional[LocalTopicSearchingAndHyDEResult]           = Field(None, description="Result of topic searching")
    detailed_search_result:                     Optional[DetailedSearchResult]                       = Field(None, description="Result of detailed search")
    information_organization_result:            Optional[InformationOrganizationResult]              = Field(None, description="Result of information organization")
    response_audit_result:                      Optional[ResponseAuditResult]                        = Field(None, description="Result of response auditing")
    # Retrieval
    retrieval_data:                             Optional[List[str]]                                  = Field(None, description="Retrieved data")
    possible_answers:                           Optional[List[str]]                                  = Field(None, description="Possible answers")
    # Reranking
    local_reranking_result:                     Annotated[List[Tuple[str, int]], concate_or_reset]   = Field([],   description="Result of local reranking")
    detail_reranking_result:                    Annotated[List[Tuple[str, int]], concate_or_reset]   = Field([],   description="Result of detail reranking")
    # Global Mapping
    global_mapping_result:                      Annotated[Dict[str, List], concate_mapping_result]   = Field(None, description="Result of global reranking")
    # Output
    generation_result:                          Optional[str]                                        = Field(None, description="The final generated result")
    repeat_times:                               Optional[int]                                        = Field(0,    description="Number of repetitions", ge=0)
    
    
    
class QueriesState(BaseModel):
    """
    Attributes:
        dataset_queries (Optional[List[str]]): List of queries from dataset.
        user_query (Optional[str]): The user's input query.
        generation_result (Optional[str]): The final generated result.
        information_organization_result (Optional[InformationOrganizationResult]): Result of information organization.
        detailed_search_result (Optional[DetailedSearchResult]): Result of detailed search.
        all_generation_results (Annotated[List[str], append_generation_result]): List of all results.
        all_information_contexts (Annotated[List[List[str]], append_information_context]): List of all contexts.
        all_vectordatabase_contexts (Annotated[List[List[Dict]], append_vectordatabase_context]): List of all contexts for multihop queries.
        all_graphdatabase_contexts (Annotated[List[List[Dict]], append_graphdatabase_context]): List of all contexts for multihop queries.
    """
    # Batch Dataset Processing       
    dataset_queries:                            Optional[List[str]]                                        = Field([],   description="List of queries from dataset")
    user_query:                                 Optional[str]                                              = Field(None, description="The user's input query")
    specific_collection:                        Optional[str]                                              = Field(None, description="Name of a specific collection, if any")
    
    generation_result:                          Optional[str]                                              = Field(None, description="The final generated result")
    information_organization_result:            Optional[InformationOrganizationResult]                    = Field(None, description="Result of information organization")
    detailed_search_result:                     Optional[DetailedSearchResult]                             = Field(None, description="Result of detailed search")
    possible_answers:                           Optional[List[str]]                                        = Field(None, description="List of possible answers")
    
    all_generation_results:                     Annotated[List[str],        append_generation_result]      = Field([],   description="List of all results")
    all_information_contexts:                   Annotated[List[List[str]],  append_information_context]    = Field([[]], description="List of all contexts")
    all_vectordatabase_contexts:                Annotated[List[List[Dict]], append_vectordatabase_context] = Field([[]], description="List of all contexts for multihop queries")
    all_graphdatabase_contexts:                 Annotated[List[List[str]],  append_graphdatabase_context]  = Field([[]], description="List of all contexts for multihop queries")
    all_possible_answers:                       Annotated[List[List[str]],  append_information_context]    = Field([[]], description="List of all possible answers")
    

def retrieval_data_concate(input_list: List[List[Dict]], new_list: List[str]) -> List[List[Dict]]:
    if new_list:
        new_list = [{"text": i} for i in new_list]
        return input_list + [new_list]
    return input_list + [[{"text": ""}]]

def information_organization_result_concate(input_list: List[List[str]], new_information_organization_result: InformationOrganizationResult) -> List[List[str]]:
    if new_information_organization_result:
        return input_list + [new_information_organization_result.organized_information]
    return input_list + [[]]

class QueriesParallelState(BaseModel):
    """
    Attributes:
        batches (List[Dict[str, Any]]): List of batches.
    """
    dataset_queries:                            List[str]                                                  = Field([],   description="List of queries from dataset")
    dataset_specific_collection:                Optional[str]                                              = Field(None, description="Name of a specific collection, if any")
    batches:                                    Optional[List[Dict[str, Any]]]
    
    query_id:                                   Annotated[List[int], lambda x, y: x + [y]]                                              = Field(None, description="The id of the query")
    information_organization_result:            Annotated[List[List[str]], information_organization_result_concate]                   = Field([[]], description="Result of information organization")
    retrieval_data:                             Annotated[List[List[Dict]], retrieval_data_concate]                                   = Field([[]], description="Retrieved data")
    generation_result:                          Annotated[List[str], lambda x, y: x + [y]]                                            = Field([], description="The final generated result")

    
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