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
            
    def search_topics_and_hyde_node(self, state: OverallState):
        # Iterate all community
        all_communities = self.retriever.global_retrieve(self.global_retrieval_level) 
        all_communities = all_communities.values.tolist()
        
                
        # Assign the community to the agents
        batches = []
        for i in range(0, len(all_communities), self.batch_size):
            batch_communities = {f"community_{j}": community for j, community in enumerate(all_communities[i:i + self.batch_size], start=i)}
            batch_json = json.dumps(batch_communities, ensure_ascii=False)
            batches.append({
                "user_query": state.user_query,
                "sub_queries": state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
                "batch_communities": batch_json,
                "batch_size": len(batch_communities),
            })
            
        score_results = self.rag_system.topic_reranking_run_batch_async(node_batch_inputs=batches)
        
        assert len(score_results.relevant_scores) == len(all_communities), "The number of scores must match the number of communities"
        
        if sum(score_results.relevant_scores) == 0:
            return {
                "search_topics_and_hyde_result": SearchTopicsAndHyDEResult(
                    communities_with_scores = {},
                    communities_summaries = [],
                    possible_answers = [],
                )
            }        
        filtered_communities = [(community, score) for community, score in zip(all_communities, score_results.relevant_scores) if score > 0]
        sorted_communities = sorted(filtered_communities, key=lambda x: x[1], reverse=True)        
        sorted_communities_only = [community for community, _ in sorted_communities]
        
        
        # ------------------------------ Topic Searching ------------------------------
        topic_searching_results = self.rag_system.topic_searching_run(
            community_information=sorted_communities_only, 
            user_query=state.user_query, 
            sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else []
        )
        
        # Create a dictionary with community information and scores
        communities_with_scores = {f"community_{i}": {"info": community, "score": score} 
                                   for i, (community, score) in enumerate(sorted_communities, start=1)}
        
        return {
            "search_topics_and_hyde_result": SearchTopicsAndHyDEResult(
                communities_with_scores = communities_with_scores,
                communities_summaries = topic_searching_results.communities_summaries,
                possible_answers = topic_searching_results.possible_answers,
            )
        } 
        
    def detailed_search_node(self, state: OverallState):
        all_queries = []
        
        if state.sub_queries_classification_result and state.sub_queries_classification_result.queries:
            all_queries.extend(state.sub_queries_classification_result.queries)
        
        if state.search_topics_and_hyde_result and state.search_topics_and_hyde_result.possible_answers:
            all_queries.extend(state.search_topics_and_hyde_result.possible_answers)

        if len(all_queries) == 0:
            return {
                "detailed_search_result": DetailedSearchResult(sorted_retrieved_data=[])
            }

        dedup_retrieved_data = self.retriever.hybrid_retrieve(state.specific_collection, all_queries, 10)

        rerank_inputs = []
        for i in range(0, len(dedup_retrieved_data), self.batch_size):
            batch = dedup_retrieved_data[i:i+self.batch_size]
            rerank_inputs.append({
                "user_query": state.user_query,
                "batch_retrieved_data": json.dumps(batch, ensure_ascii=False),
                "sub_queries": state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
                "batch_size": len(batch)
            })
        
        score_results = self.rag_system.reranking_run_batch_async(node_batch_inputs=rerank_inputs)
        
        assert len(score_results.relevance_scores) == len(dedup_retrieved_data), "The number of scores must match the number of retrieved data"
        
        if sum(score_results.relevance_scores) == 0:
            return {
                "detailed_search_result": DetailedSearchResult(sorted_retrieved_data=[])
            }

        dedup_data_with_scores = [(data, score) for data, score in zip(dedup_retrieved_data, score_results.relevance_scores) if score > 0]
        community_data = [(community_info["info"], community_info["score"]) for community_info in state.search_topics_and_hyde_result.communities_with_scores.values()]
        all_data = dedup_data_with_scores + community_data
        sorted_results = sorted(all_data, key=lambda x: x[1], reverse=True)
        final_results = [item[0] for item in sorted_results]
        
        return {
            "detailed_search_result": DetailedSearchResult(sorted_retrieved_data=final_results)
        }
        
    def information_organization_node(self, state: OverallState):
        return {
            "information_organization_result": self.rag_system.information_organization_run(
                user_query=state.user_query,
                sub_queries=state.sub_queries_classification_result.queries if state.sub_queries_classification_result else [],
                retrieved_data=state.detailed_search_result.sorted_retrieved_data if state.detailed_search_result else [],
                community_information=state.search_topics_and_hyde_result.communities_summaries if state.search_topics_and_hyde_result else []
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
        new_context = state.detailed_search_result.sorted_retrieved_data if state.detailed_search_result else []

        return {
            "all_results": [new_answer],
            "all_contexts": [new_context]
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
        if state.user_query_classification_result.needs_retrieval:
            return "retrieval_needed"
        else:
            return "retrieval_not_needed"
        
    def is_information_organization_needed_cnode(self, state: OverallState):
        if (state.search_topics_and_hyde_result and state.search_topics_and_hyde_result.communities_with_scores) or (state.detailed_search_result and state.detailed_search_result.sorted_retrieved_data):
            return "information_organization_needed"
        else:
            return "information_organization_not_needed"
    
    def is_restart_needed_cnode(self, state: OverallState):
        if state.response_audit_result.restart_required and state.repeat_times < 3:
            return "restart_needed"
        else:
            return "restart_not_needed"
                
    def is_dataset_unfinished_cnode(self, state: OverallState):
        print("this is the end of the workflow")
        print("len(state.dataset_queries) = ", len(state.dataset_queries))
        print("len(state.all_results) = ", len(state.all_results))
        if len(state.dataset_queries) > len(state.all_results):
            return "dataset_unfinished"
        else:
            return "dataset_finished"
    
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