from langgraph.graph import StateGraph
from .state import OverallState
from .nodes import Nodes

class WorkFlow():
    def __init__(self, query: str, collection: str):
        # Create the workflow(graph)
        workflow = StateGraph(OverallState)
        
        # Add nodes to the workflow
        nodes = Nodes(query, collection)
        workflow.add_node("user_query_classification_node", nodes.user_query_classification_node)
        workflow.add_node("retrieval_and_generation_node", nodes.retrieval_and_generation_node)
        workflow.add_node("generation_node", nodes.generation_node)
        workflow.add_node("repeat_count_node", nodes.repeat_count_node)
        workflow.add_node("database_update_node", nodes.database_update_node)

        # Set starting node and finish node
        workflow.set_entry_point("user_query_classification_node")
        workflow.set_finish_point("database_update_node")
        
        # Add edges to the workflow
        workflow.add_conditional_edges(
            "user_query_classification_node",
            nodes.is_retrieval_needed,
            {
                "retrieval_needed": "retrieval_and_generation_node",
                "retrieval_not_needed": "generation_node",
            }
        )
        workflow.add_edge("retrieval_and_generation_node", "repeat_count_node")
        workflow.add_edge("generation_node", "repeat_count_node")
        workflow.add_conditional_edges(
            "repeat_count_node",
            nodes.is_restart_needed,
            {
                "restart_needed": "user_query_classification_node",
                "restart_not_needed": "database_update_node",
            }
        )
        
        # Compile
        self.app = workflow.compile()