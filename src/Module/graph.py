from langgraph.graph import StateGraph
from .state import OverallState, SingleState
from .nodes import NodesModularRAG, NodesMultiAgentRAG, NodesSingleAgentRAG
from Utils import *
from Frontend import *
from Config.rag_config import RAGConfig

class WorkFlowModularRAG():
    def __init__(self, 
                query: str, 
                collection: str, 
                rag_config: RAGConfig,
                ):
        # Create the workflow(graph)
        workflow = StateGraph(OverallState)
        
        # Add nodes to the workflow
        nodes = NodesModularRAG(query, collection, rag_config)
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
        
class WorkFlowMultiAgentRAG():
    def __init__(self, 
                query: str, 
                collection: str, 
                rag_config: RAGConfig,
                ):        
        # Create the workflow(graph)
        workflow = StateGraph(OverallState)
        
        # Add nodes to the workflow
        nodes = NodesMultiAgentRAG(query, collection, rag_config)
        workflow.add_node("overall_node", nodes.overall_node)
        workflow.set_entry_point("overall_node")
        workflow.set_finish_point("overall_node")
        
        # Compile
        self.app = workflow.compile()
        
class WorkFlowSingleAgentRAG():
    def __init__(self, 
                query: str, 
                collection: str, 
                rag_config: RAGConfig,
                ):        
        # Create the workflow(graph)
        workflow = StateGraph(SingleState)
        
        # Add nodes to the workflow
        nodes = NodesSingleAgentRAG(query, collection, rag_config)
        workflow.add_node("run_node", nodes.run_node)
        workflow.set_entry_point("run_node")
        workflow.set_finish_point("run_node")
        
        # Compile
        self.app = workflow.compile()
        