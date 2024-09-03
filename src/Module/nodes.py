from MultiAgent import *
from SingleAgent import *
from .state import OverallState, SingleState
from Frontend import *
from Utils import *
from Config.rag_config import RAGConfig
from Config.output_pydantic import TopicResult 
from pandas import DataFrame
import json

class NodesModularRAG():
    def __init__(self, rag_config: RAGConfig):
        self.rag_system = MultiAgent_RAG(rag_config)
        self.retriever = Retriever()
        self.global_retrive_level = 1
        self.batch_size = 5

    def user_input_node(self, state: OverallState):
        state.user_query = "How old is Alice?" 
        state.specific_collection = "alice"
        return {
            "user_query": state.user_query,
            "specific_collection": state.specific_collection
        }
    
    def user_query_classification_node(self, state: OverallState):
        return self.rag_system.user_query_classification_run(user_query=state.user_query)
    
    def query_process_node(self, state: OverallState):
        return self.rag_system.query_process_run(user_query=state.user_query)
    
    def sub_query_classification_node(self, state: OverallState):
        if state.specific_collection is None:
            return self.rag_system.sub_queries_classification_without_specification_run(user_query=state.user_query)
        else:
            return self.rag_system.sub_queries_classification_with_specification_run(user_query=state.user_query, 
                                                                                    specific_collection=state.specific_collection)
    
    def topic_search_node(self, state: OverallState):
        # Iterate all community
        all_communities = self.retriever.retrieve_all_communities(self.global_retrive_level) 
        all_communities = all_communities.values.tolist()
        
                
        # Assign the community to the agents
        batches = []
        for i in range(0, len(all_communities), self.batch_size):
            batch_communities = {f"community_{j}": community 
                                for j, community in enumerate(all_communities[i:i + self.batch_size], start=i)}
            batch_json = json.dumps(batch_communities, ensure_ascii=False)
            batches.append({
                "batch_communities": batch_json,
                "user_query": state.user_query,
                "batch_size": len(batch_communities)
            })
            
        score_results = self.rag_system.topic_reranking_run_batch_async(node_batch_inputs=batches)
        
        filtered_communities = [(community, score) for community, score in zip(all_communities, score_results) if score > 0]
        sorted_communities = sorted(filtered_communities, key=lambda x: x[1], reverse=True)
        sorted_communities_only = [community for community, _ in sorted_communities]
        
        searching_results = self.rag_system.topic_searching_run(communitiy_information=sorted_communities_only)
        
        # Make Summarization and Hypothesis Answers
        return {
            "topic_result": TopicResult(
                communities_with_scores = {f"community_{i}": score for i, (community, score) in enumerate(sorted_communities, start=1)},
                communities_summaries = searching_results.community_summaries,
                possible_answers = searching_results.possible_answers
            )
        } 
        
    def detailed_search_node(self, state: OverallState):
        # do detailed search from the community hypothesis answers or from the topic
        
        # rerank the retrieved data and community information
        
        # aggregate the data retrieved and the community information by function instead of by LLM
        
        # summarize all the data information for generating the final response 
        
        pass
    
    def all_data_summarization_node(self, state: OverallState):
        pass
        
    
        
        
    def retrieval_node(self, state: OverallState):
        pass
   
    def rerank_node(self, state: OverallState):
        pass
         
    def generation_node(self, state: OverallState):
        return self.rag_system.generation_run()
        
    def repeat_count_node(self, state: OverallState):
        repeat_times = state["repeat_times"]
        if repeat_times is None:
            repeat_times = 0
        return {
            "repeat_times": repeat_times+1
        }

    def database_update_node(self, state: OverallState):
        return self.rag_system.database_update_run()
        
        
    # Conditional Nodes
    def is_retrieval_needed(self, state: OverallState):
        if state.user_query_classification_result.needs_retrieval:
            return "retrieval_needed"
        else:
            return "retrieval_not_needed"
        
    def is_restart_needed(self, state: OverallState):
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