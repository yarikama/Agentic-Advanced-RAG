from MultiAgent import *
from SingleAgent import *
from .state import OverallState, SingleState
from Frontend import *
from Utils import *
from Config.rag_config import RAGConfig
class NodesModularRAG():
    def __init__(self, 
                user_query: str, 
                specific_collection: str, 
                rag_config: RAGConfig,
                ):
        
        self.rag_system = LLMMA_RAG_System(rag_config)
        self.rag_system.tasks.update_tasks(user_query, specific_collection)
    
    
    # Action Nodes
    def user_query_classification_node(self, state: OverallState):
        return self.rag_system.user_query_classification_run()
    
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
        if state["user_query_classification"].needs_retrieval:
            return "retrieval_needed"
        else:
            return "retrieval_not_needed"
        
    def is_restart_needed(self, state: OverallState):
        if state["audit_result"].restart_required and state["repeat_times"] < 3:
            return "restart_needed"
        else:
            return "restart_not_needed"
                
        
class NodesMultiAgentRAG():
    def __init__(self, 
                user_query: str, 
                specific_collection: str, 
                rag_config: RAGConfig,
                ):
        
        self.rag_system = LLMMA_RAG_System(rag_config)
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