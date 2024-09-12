from MultiAgent import *
from SingleAgent import *
# from Frontend import *
from Utils import *
from Config.rag_config import RAGConfig
from Config.output_pydantic import *
from Config.constants import *
from pandas import DataFrame
import json
import pandas as pd

class NodesModularRAG():
    def __init__(self):
        self.retriever = Retriever()
        self.rag_system = MultiAgent_RAG()
        self.batch_size = NODE_BATCH_SIZE
        self.global_retrieval_level = NODE_RETRIEVAL_LEVEL
        
    def prepare_batch_input_for_reranking(self, 
                                        input_list: List[str], 
                                        sub_queries: List[str], 
                                        user_query: str
                                        ) -> List[Dict[str, Any]]:
        """
        This function is used to prepare the batch input for the reranking.
        args:
            input_list: List[str]: the list of data to be reranked
            sub_queries: List[str]: the list of sub queries
            user_query: str: the user query
        returns:
            batches: List[Dict[str, Any]]: the batch input for the reranking
                - user_query: str: the user query
                - sub_queries: List[str]: the list of sub queries
                - batch_data: List[str]: the list of data to be reranked
                - batch_size: int: the size of the batch
        """    
        batches = []
        for i in range(0, len(input_list), self.batch_size):
            batch_input = input_list[i:i+self.batch_size]
            batches.append({
                "user_query": user_query,
                "sub_queries": sub_queries,
                "batch_data": batch_input,
                "batch_size": len(batch_input)
            })
        return batches
    
    def sort_data_desc_and_filter_by_score(self, data: List[str], scores: List[int])-> Tuple[List[str], List[Tuple[str, int]]]:
        """
        This function is used to sort the data in descending order and filter the data by the score.
        args:
            data: List[str]: the list of data to be sorted
            scores: List[int]: the list of scores
        returns:
            sorted_data: List[str]: the list of data sorted in descending order
            sorted_data_with_scores: List[Tuple[str, int]]: the list of data with scores in descending order
        """
        assert len(data) == len(scores), "The number of data and scores must match, len(data) = %d, len(scores) = %d" % (len(data), len(scores))
        filtered_data_with_scores = [(data, score) for data, score in zip(data, scores) if score > 0]
        sorted_data_with_scores = sorted(filtered_data_with_scores, key=lambda x: x[1], reverse=True)
        sorted_data = [data for data, _ in sorted_data_with_scores]
        return sorted_data, sorted_data_with_scores
    
    def update_next_query_node(self, state: OverallState):
        index_of_current_result = len(state.all_results)
        print("index = ", index_of_current_result)
        print("total = ", len(state.dataset_queries))
        return {
            "user_query": state.dataset_queries[index_of_current_result]
        }
        
    def user_query_classification_node(self, state: OverallState):
        return {  
            "user_query_classification_result": self.rag_system.user_query_classification_run(user_query=state.user_query)
        } 
    
    def query_process_node(self, state: OverallState):
        return {  
            "query_process_result": self.rag_system.query_process_run(user_query=state.user_query)
        } 
    
    def sub_queries_classification_node(self, state: OverallState):
        if state.specific_collection is None:
            return {
                "sub_queries_classification_result": self.rag_system.sub_queries_classification_without_specification_run(user_query=state.user_query)
            }
        else:
            return {
                "sub_queries_classification_result": self.rag_system.sub_queries_classification_with_specification_run(user_query=state.user_query, specific_collection=state.specific_collection)
            }
            
    def global_topic_searching_and_hyde_node(self, state: OverallState):
        # Iterate all community
        all_communities = self.retriever.global_retrieve(self.global_retrieval_level)["communities"]
        
        # Assign the community to the agents
        batch_inputs = self.prepare_batch_input_for_reranking(
            input_list=all_communities,
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
            user_query=state.user_query
        )
            
        all_scores = self.rag_system.global_topic_reranking_run_batch_async(node_batch_inputs=batch_inputs).relevant_scores        
        if sum(all_scores) == 0:
            return {
                "global_topic_searching_and_hyde_result": GlobalTopicSearchingAndHyDEResult(
                    communities_with_scores = {},
                    communities_summaries = [],
                    possible_answers = [],
                )
            }        
            
        sorted_communities, sorted_communities_with_scores = self.sort_data_desc_and_filter_by_score(all_communities, all_scores)
        # ------------------------------ Topic Searching ------------------------------
        topic_searching_results = self.rag_system.global_topic_searching_run(
            user_query=state.user_query, 
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
            data=sorted_communities,
        )
        communities_summaries = topic_searching_results.communities_summaries
        possible_answers = topic_searching_results.possible_answers
        return {
            "global_topic_searching_and_hyde_result": GlobalTopicSearchingAndHyDEResult(
                communities_with_scores = sorted_communities_with_scores,
                communities_summaries = communities_summaries,
                possible_answers  = possible_answers,
            )
        } 
        
    def local_topic_searching_and_hyde_node(self, state: OverallState):
        # judge the top_k, top_community, top_inside_relations from the score
        
        # get the results from the local retriever
        results = self.retriever.local_retrieve([state.user_query])
        chunks = results["text_mapping"]
        communities = results["report_mapping"]
        outside_relations = results["outside_relations"]
        inside_relations = results["inside_relations"]
        entities = results["entities"]
        
        # use the results to run the topic reranking
        all_information = list(set(chunks + communities + outside_relations + inside_relations + entities))
        
        print("all_information = ", all_information)
        
        # Batch the information
        batch_inputs = self.prepare_batch_input_for_reranking(
            input_list=all_information,
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
            user_query=state.user_query
        )
        
        all_scores = self.rag_system.local_topic_reranking_run_batch_async(node_batch_inputs=batch_inputs).relevant_scores
        print("all_scores = ", all_scores)
        # If the score is 0, return an empty result
        if sum(all_scores) == 0:
            return {
                "local_topic_searching_and_hyde_result": LocalTopicSearchingAndHyDEResult(
                    information_with_scores = [],
                    information_summaries = [],
                    possible_answers = [],
                )
            }
        
        # Filter the information with the score
        sorted_information, sorted_information_with_scores = self.sort_data_desc_and_filter_by_score(all_information, all_scores)
        
        # after reranking, do topic searching for HyDE
        local_topic_searching_results = self.rag_system.local_topic_searching_run(
            user_query=state.user_query,
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
            data=sorted_information
        )
        # return the result (decide the result for generations)
        information_summaries = local_topic_searching_results.information_summaries
        possible_answers = local_topic_searching_results.possible_answers
        return {
            "local_topic_searching_and_hyde_result": LocalTopicSearchingAndHyDEResult(
                information_with_scores = sorted_information_with_scores,
                information_summaries = information_summaries,
                possible_answers = possible_answers,
            )
        }
        
    def detailed_search_node(self, state: OverallState):
        all_data_with_scores = []
        all_queries = [state.user_query]
        
        if state.sub_queries_classification_result and state.sub_queries_classification_result.queries:
            all_queries.extend(state.sub_queries_classification_result.queries)

        if state.local_topic_searching_and_hyde_result:
            all_queries.extend(state.local_topic_searching_and_hyde_result.possible_answers)
            all_data_with_scores.extend(state.local_topic_searching_and_hyde_result.information_with_scores)
            
        if state.global_topic_searching_and_hyde_result:
            all_queries.extend(state.global_topic_searching_and_hyde_result.possible_answers)
            all_data_with_scores.extend(state.global_topic_searching_and_hyde_result.communities_with_scores)
            
        retrieved_data = self.retriever.hybrid_retrieve(state.specific_collection, all_queries, 10)

        batch_inputs = self.prepare_batch_input_for_reranking(
            input_list=retrieved_data,
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
            user_query=state.user_query
        )
        
        retrieved_scores = self.rag_system.reranking_run_batch_async(node_batch_inputs=batch_inputs).relevance_scores
        
        assert len(retrieved_data) == len(retrieved_scores), "The number of data and scores must match, len(data) = %d, len(scores) = %d" % (len(retrieved_data), len(retrieved_scores))        
        retrieved_data_with_scores = list(zip(retrieved_data, retrieved_scores))
        all_data_with_scores = sorted(all_data_with_scores + retrieved_data_with_scores, key=lambda x: x[1], reverse=True)
        sorted_data = [data for data, _ in all_data_with_scores]
        return {
            "detailed_search_result": DetailedSearchResult(sorted_retrieved_data=sorted_data)
        }
        
    def information_organization_node(self, state: OverallState):
        return {
            "information_organization_result": self.rag_system.information_organization_run(
                user_query=state.user_query,
                sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
                retrieved_data=state.detailed_search_result.sorted_retrieved_data if state.detailed_search_result else [],
            )
        }
         
    def generation_node(self, state: OverallState):        
        return {
            "generation_result": self.rag_system.generation_run(
                user_query=state.user_query,
                sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else []
            )
        }
        
    def store_result_for_ragas_node(self, state: OverallState):
        new_answer = state.generation_result
        new_context = state.information_organization_result if state.information_organization_result else []
        return {
            "user_query": None,
            "user_query_classification_result": None,
            "query_process_result": None,
            "sub_queries_classification_result": None,
            "global_topic_searching_and_hyde_result": None,
            "local_topic_searching_and_hyde_result": None,
            "detailed_search_result": None,
            "information_organization_result": None,
            "response_audit_result": None,
            "generation_result": None,
            "repeat_times": 0,
            "all_results": [new_answer],
            "all_contexts": [new_context],
        }
        
    def repeat_count_node(self, state: OverallState):
        repeat_times = state["repeat_times"]
        if repeat_times is None:
            repeat_times = 0
        return {
            "repeat_times": repeat_times + 1
        }

    def database_update_node(self, state: OverallState):
        return self.rag_system.database_update_run()
        
    # Conditional Nodes
    def is_retrieval_needed_cnode(self, state: OverallState):
        if not state.user_query_classification_result.needs_retrieval:
            return "retrieval_not_needed"
        if state.user_query_classification_result.domain_range_score >= 70:
            return "retrieval_needed_for_global_topic_searching"
        else:
            return "retrieval_needed_for_local_topic_searching"
        
    def is_information_organization_needed_cnode(self, state: OverallState):
        if state.detailed_search_result and state.detailed_search_result.sorted_retrieved_data:
            return "information_organization_needed"
        else:
            return "information_organization_not_needed"
    
    def is_restart_needed_cnode(self, state: OverallState):
        if state.response_audit_result.restart_required and state.repeat_times < 3:
            return "restart_needed"
        else:
            return "restart_not_needed"
                
    def is_dataset_unfinished_cnode(self, state: OverallState):
        if len(state.dataset_queries) > len(state.all_results):
            return "dataset_unfinished"
        else:
            return "dataset_finished"
    
    def is_global_local_cnode(self, state: OverallState):
        if state.user_query_classification_result.domain_range_score >= 70:
            return "global retriever"
        else:
            return "local retriever"
    
class NodesMultiAgentRAG():
    def __init__(self, 
                user_query: str, 
                specific_collection: str, 
                rag_config: RAGConfig,
                ):
        
        self.rag_system = MultiAgent_RAG(rag_config)
        self.rag_system.tasks.update_tasks(user_query, specific_collection)
    
    def overall_node(self, state: OverallState):
        return self.rag_system.overall_run()
    
class NodesSingleAgentRAG():
    def __init__(self, 
                user_query: str, 
                specific_collection: str, 
                rag_config: RAGConfig,
                ):
        
        self.user_query = user_query
        self.specific_collection = specific_collection
        self.rag_system = SingleAgent(rag_config)
        
    def run_node(self, state: SingleState):
        return self.rag_system.run(self.user_query, self.specific_collection)