from langgraph.graph import StateGraph
from .nodes import NodesModularRAG, NodesMultiAgentRAG, NodesSingleAgentRAG
from Utils import *
# from Frontend import *
from Config.rag_config import RAGConfig
from Config.output_pydantic import *

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
        workflow.add_node("retrieval_node", nodes.retrieval_node)
        workflow.add_node("rerank_node", nodes.rerank_node)
        workflow.add_node("generation_node", nodes.generation_node)
        workflow.add_node("repeat_count_node", nodes.repeat_count_node)
        workflow.add_node("database_update_node", nodes.database_update_node)

        # Set starting node
        workflow.set_entry_point("user_query_classification_node")
        
        # Add edges to the workflow
        workflow.add_conditional_edges(
            "user_query_classification_node",
            nodes.is_retrieval_needed,
            {
                "retrieval_needed": "retrieval_node",
                "retrieval_not_needed": "generation_node",
            }
        )
        workflow.add_edge("retrieval_node", "rerank_node")
        workflow.add_edge("rerank_node", "generation_node")
        workflow.add_edge("generation_node", "repeat_count_node")
        workflow.add_conditional_edges(
            "repeat_count_node",
            nodes.is_restart_needed,
            {
                "restart_needed": "user_query_classification_node",
                "restart_not_needed": "database_update_node",
            }
        )

        # Set finish node
        workflow.set_finish_point("database_update_node")
        # Compile
        self.graph = workflow.compile()
        
class WorkFlowModularHybridRAG():
    def __init__(self):
        # Create the workflow(graph)
        workflow = StateGraph(OverallState)
        
        # Add nodes class
        nodes = NodesModularRAG()
        
        # Add nodes into the workflow
        workflow.add_node("user_query_classification_node", nodes.user_query_classification_node)
        workflow.add_node("query_process_node", nodes.query_process_node)
        workflow.add_node("sub_query_classification_node", nodes.sub_query_classification_node)
        workflow.add_node("topic_search_node", nodes.topic_search_node)
        workflow.add_node("detailed_search_node", nodes.detailed_search_node)
        workflow.add_node("information_organization_node", nodes.information_organization_node)
        workflow.add_node("generation_node", nodes.generation_node)
        
        # Draw the workflow
        workflow.set_entry_point("user_query_classification_node")
        workflow.add_conditional_edges(
            "user_query_classification_node",
            nodes.is_retrieval_needed_cnode,
            {
                "retrieval_needed": "query_process_node",
                "retrieval_not_needed": "generation_node",
            }
        )
        workflow.add_edge("query_process_node", "sub_query_classification_node")
        workflow.add_edge("sub_query_classification_node", "topic_search_node")
        workflow.add_edge("topic_search_node", "detailed_search_node")
        workflow.add_conditional_edges(
            "detailed_search_node",
            nodes.is_information_organization_needed_cnode,
            {
                "information_organization_needed": "information_organization_node",
                "information_organization_not_needed": "generation_node",
            }
        )
        workflow.add_edge("information_organization_node", "generation_node")
        workflow.set_finish_point("generation_node")
        
        # Compile
        self.graph = workflow.compile()
        
        
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
        self.graph = workflow.compile()
        
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
        self.graph = workflow.compile()
        
        
if __name__ == "__main__":
    workflow = WorkFlowModularHybridRAG()
    print(workflow)