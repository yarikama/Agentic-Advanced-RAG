from MultiAgent import *
from SingleAgent import *
from .state import OverallState, SingleState

class NodesModularRAG():
    def __init__(self, user_query: str, specific_collection: str):
        self.rag_system = LLMMA_RAG_System()
        self.rag_system.tasks.update_task(user_query, specific_collection)
    
    # Action Nodes
    def user_query_classification_node(self, state: OverallState):
        return self.rag_system.user_query_classification_run()
    
    def retrieval_and_generation_node(self, state: OverallState):
        return self.rag_system.retrieval_and_generation_run()
         
    def generation_node(self, state: OverallState):
        return self.rag_system.generation_run()
        
    def repeat_count_node(self, state: OverallState):
        return {
            "repeat_times": state["repeat_times"] + 1
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
    def __init__(self, user_query: str, specific_collection: str):
        self.rag_system = LLMMA_RAG_System()
        self.rag_system.tasks.update_task(user_query, specific_collection)
    
    def overall_node(self, state: OverallState):
        return self.rag_system.overall_run()
    
class NodesSingleAgentRAG():
    def __init__(self, user_query: str, specific_collection: str):
        self.user_query = user_query
        self.specific_collection = specific_collection
        self.rag_system = SingleAgent()
        
    def run_node(self, state: SingleState):
        return self.rag_system.run(self.user_query, self.specific_collection)