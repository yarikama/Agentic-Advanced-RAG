from LLMMA import *
from .state import OverallState
from langgraph.graph import END

class Nodes:
    def __init__(self, query: str, collection: str):
        self.rag_system = LLMMA_RAG_System()
        self.rag_system.tasks.update_task(query, collection)
    
    # Action Nodes
    def user_query_classification_node(self, state: OverallState):
        result = self.rag_system.user_query_classification_run()
        return OverallState(**{
            **state.model_dump(),
            "user_query_classification": result
        })
        
    def retrieval_and_generation_node(self, state: OverallState):
        retrieval_result, generation_result = self.rag_system.retrieval_and_generation_run()
        return OverallState(**{
            **state.model_dump(),
            "refined_retrieval_data": retrieval_result,
            "result": generation_result
        })
    
    def generation_node(self, state: OverallState):
        generation_result = self.rag_system.generation_run()
        return OverallState(**{
            **state.model_dump(),
            "result": generation_result
        })
        
    def database_update_node(self, state: OverallState):
        self.rag_system.database_update_run()
        return END
        
    # Conditional Nodes
    def is_retrieval_needed(self, state: OverallState):
        if state.user_query_classification.needs_retrieval:
            return "retrieval_needed"
        else:
            return "retrieval_not_needed"
        
    def is_restart_needed(self, state: OverallState):
        if state.audit_result.restart_required or state.repeat_times >= 3:
            return "restart_needed", OverallState(**{
                **state.model_dump(),
                "repeat_times": state.repeat_times + 1
            })
        else:
            return "restart_not_needed", state