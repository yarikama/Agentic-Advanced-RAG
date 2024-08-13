from langgraph.graph import StateGraph
from .state import OverallState
from .nodes import Nodes

class WorkFlow():
    def __init__(self, query: str, collection: str):
        # Create the workflow(graph)
        workflow = StateGraph(OverallState)
        
        # Add nodes to the workflow
        nodes = Nodes(query, collection)
        workflow.add_node("user_query_classification", nodes.user_query_classification_node)
        workflow.add_node("is_retrieval_needed", nodes.is_retrieval_needed)
        workflow.add_node("retrieval_and_generation", nodes.retrieval_and_generation_node)
        workflow.add_node("generation", nodes.generation_node)
        workflow.add_node("is_restart_needed", nodes.is_restart_needed)
        workflow.add_node("database_update", nodes.database_update_node)

        # Set starting node
        workflow.set_entry_point("user_query_classification")
        
        # Add edges to the workflow
        workflow.add_edge('user_query_classification', 'is_retrieval_needed')
        workflow.add_conditional_edges(
            "retrieval_needed?",
            nodes.is_retrieval_needed,
            {
                "retrieval_needed": "retrieval_and_generation",
                "retreival_not_needed": "generation",
            }
        )
        workflow.add_edge("retrieval_and_generation", "is_restart_needed")
        workflow.add_edge("generation", "is_restart_needed")
        workflow.add_conditional_edges(
            "restart_needed?",
            nodes.is_restart_needed,
            {
                "restart_needed": "user_query_classification",
                "restart_not_needed": "database_update",
            }
        )
        
        
        # Compile
        self.app = workflow.compile()