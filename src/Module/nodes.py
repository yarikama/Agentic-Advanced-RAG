from MultiAgent import *
from SingleAgent import *
# from Frontend import *
from Utils import *
from Config.rag_config import RAGConfig
from Config.output_pydantic import *
from pandas import DataFrame
import json
import pandas as pd

class NodesModularRAG():
    def __init__(self):
        self.retriever = Retriever()
        self.global_retrive_level = 0
        self.batch_size = 5
        self.rag_system = MultiAgent_RAG()
        
    def user_query_classification_node(self, state: OverallState):
        return {  
            "user_query_classification_result": self.rag_system.user_query_classification_run(user_query=state.user_query)
        } 
    
    def query_process_node(self, state: OverallState):
        return {  
            "queries_process_result": self.rag_system.query_process_run(user_query=state.user_query)
        } 
    
    def sub_query_classification_node(self, state: OverallState):
        if state.specific_collection is None:
            return {
                "sub_queries_classification_result": self.rag_system.sub_queries_classification_without_specification_run(user_query=state.user_query)
            }
        else:
            return {
                "sub_queries_classification_result": self.rag_system.sub_queries_classification_with_specification_run(user_query=state.user_query, specific_collection=state.specific_collection)
            }
            
    def topic_search_node(self, state: OverallState):
        # Iterate all community
        all_communities = self.retriever.retrieve_all_communities(self.global_retrive_level) 
        all_communities = all_communities.values.tolist()
        
                
        # Assign the community to the agents
        batches = []
        for i in range(0, len(all_communities), self.batch_size):
            batch_communities = {f"community_{j}": community for j, community in enumerate(all_communities[i:i + self.batch_size], start=i)}
            batch_json = json.dumps(batch_communities, ensure_ascii=False)
            batches.append({
                "batch_communities": batch_json,
                "user_query": state.user_query,
                "batch_size": len(batch_communities)
            })
            
        score_results = self.rag_system.topic_reranking_run_batch_async(node_batch_inputs=batches)

        if sum(score_results.relevant_scores) == 0:
            return {
                "topic_result": TopicResult(
                    communities_with_scores = {},
                    communities_summaries = [],
                    possible_answers = [],
                )
            }
        
        # Combine community information and scores
        filtered_communities = [(community, score) for community, score in zip(all_communities, score_results.relevant_scores) if score > 0]
        sorted_communities = sorted(filtered_communities, key=lambda x: x[1], reverse=True)
        
        # Extract only sorted community information
        sorted_communities_only = [community for community, _ in sorted_communities]
        
        topic_searching_results = self.rag_system.topic_searching_run(community_information=sorted_communities_only, user_query=state.user_query)
        
        # Create a dictionary with community information and scores
        communities_with_scores = {f"community_{i}": {"info": community, "score": score} 
                                   for i, (community, score) in enumerate(sorted_communities, start=1)}
        
        return {
            "topic_result": TopicResult(
                communities_with_scores = communities_with_scores,
                communities_summaries = topic_searching_results.communities_summaries,
                possible_answers = topic_searching_results.possible_answers,
            )
        } 
        
    def detailed_search_node(self, state: OverallState):
        all_queries = []
        
        if state.sub_queries_classification_result and state.sub_queries_classification_result.queries:
            all_queries.extend(state.sub_queries_classification_result.queries)
        
        if state.topic_result and state.topic_result.possible_answers:
            all_queries.extend(state.topic_result.possible_answers)

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
                "batch_size": len(batch)
            })
        
        score_results = self.rag_system.reranking_run_batch_async(node_batch_inputs=rerank_inputs)
        
        if sum(score_results.relevance_scores) == 0:
            return {
                "detailed_search_result": DetailedSearchResult(sorted_retrieved_data=[])
            }

        dedup_data_with_scores = [(data, score) for data, score in zip(dedup_retrieved_data, score_results.relevance_scores) if score > 0]
        community_data = [(community_info["info"], community_info["score"]) for community_info in state.topic_result.communities_with_scores.values()]
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
                retrieved_data=state.detailed_search_result.sorted_retrieved_data if state.detailed_search_result else [],
                community_information=state.topic_result.communities_summaries if state.topic_result else []
            )
        }
         
    def generation_node(self, state: OverallState):        
        return {
            "generation_result": self.rag_system.generation_run(user_query=state.user_query)
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
        if (state.topic_result and state.topic_result.communities_with_scores) or (state.detailed_search_result and state.detailed_search_result.sorted_retrieved_data):
            return "information_organization_needed"
        else:
            return "information_organization_not_needed"
    
    def is_restart_needed_cnode(self, state: OverallState):
        if state.response_audit_result.restart_required and state.repeat_times < 3:
            return "restart_needed"
        else:
            return "restart_not_needed"
                
        
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