from langgraph.graph import StateGraph, END
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
    """
    Initialize the workflow with 
    {
        dataset_queries: [str],
    }
    """
    def __init__(self):
        # Create the workflow(graph)
        workflow = StateGraph(OverallState)
        
        # Add nodes class
        nodes = NodesModularRAG()
        
        # Add nodes into the workflow
        workflow.add_node("update_next_query_node", nodes.update_next_query_node)
        workflow.add_node("user_query_classification_node", nodes.user_query_classification_node)
        workflow.add_node("retrieve_global_data_node", nodes.retrieve_global_data_node)
        workflow.add_node("retrieve_local_data_node", nodes.retrieve_local_data_node)
        workflow.add_node("retrieve_detail_data_node", nodes.retrieve_detail_data_node)
        workflow.add_node("global_mapping_node", nodes.global_mapping_node)
        workflow.add_node("local_mapping_node", nodes.local_mapping_node)
        workflow.add_node("detail_mapping_node", nodes.detail_mapping_node)
        workflow.add_node("global_reducing_node", nodes.global_reducing_node)
        workflow.add_node("local_reducing_node", nodes.local_reducing_node)
        workflow.add_node("detail_reducing_node", nodes.detail_reducing_node)
        workflow.add_node("information_organization_node", nodes.information_organization_node)
        workflow.add_node("generation_node", nodes.generation_node)
        workflow.add_node("store_result_for_ragas_node", nodes.store_result_for_ragas_node)

        # Draw the workflow
        workflow.set_entry_point("update_next_query_node")
        workflow.add_edge("update_next_query_node", "user_query_classification_node")
        workflow.add_conditional_edges(
            "user_query_classification_node",
            nodes.is_retrieval_needed_cnode,
            {
                "retrieval_not_needed": "generation_node",
                "retrieval_needed_for_global_topic_searching": "retrieve_global_data_node",
                "retrieval_needed_for_local_topic_searching": "retrieve_local_data_node",
            }
        )
        workflow.add_conditional_edges("retrieve_global_data_node", nodes.dispatch_global_mapping_cnode, ["global_mapping_node"])
        workflow.add_conditional_edges("retrieve_local_data_node", nodes.dispatch_local_mapping_cnode, ["local_mapping_node"])
        workflow.add_edge("global_mapping_node", "global_reducing_node")
        workflow.add_edge("local_mapping_node", "local_reducing_node")
        workflow.add_edge("global_reducing_node", "retrieve_detail_data_node")
        workflow.add_edge("local_reducing_node", "retrieve_detail_data_node")
        workflow.add_conditional_edges("retrieve_detail_data_node", nodes.dispatch_detail_mapping_cnode, ["detail_mapping_node"])
        workflow.add_edge("detail_mapping_node", "detail_reducing_node")
        workflow.add_edge("detail_reducing_node", "information_organization_node")
        workflow.add_edge("information_organization_node", "generation_node")
        workflow.add_edge("generation_node", "store_result_for_ragas_node")
        workflow.add_conditional_edges(
            "store_result_for_ragas_node",
            nodes.is_dataset_unfinished_cnode,
            {
                "dataset_unfinished": "update_next_query_node",
                "dataset_finished": END,
            }
        )
        # Compile
        self.graph = workflow.compile()
        
        
class WorkFlowModularHybridRAG_Unit_Function_Test():
    def __init__(self):
        # Create the workflow(graph)
        workflow = StateGraph(OverallState)
        
        # Add nodes class
        nodes = NodesModularRAG()
        
        # Add nodes into the workflow
        workflow.add_node("user_query_classification_node", nodes.user_query_classification_node)
        workflow.add_node("retrieve_global_data_node", nodes.retrieve_global_data_node)
        workflow.add_node("retrieve_local_data_node", nodes.retrieve_local_data_node)
        workflow.add_node("retrieve_detail_data_node", nodes.retrieve_detail_data_node)
        workflow.add_node("global_mapping_node", nodes.global_mapping_node)
        workflow.add_node("local_mapping_node", nodes.local_mapping_node)
        workflow.add_node("detail_mapping_node", nodes.detail_mapping_node)
        workflow.add_node("global_reducing_node", nodes.global_reducing_node)
        workflow.add_node("local_reducing_node", nodes.local_reducing_node)
        workflow.add_node("detail_reducing_node", nodes.detail_reducing_node)
        workflow.add_node("information_organization_node", nodes.information_organization_node)
        workflow.add_node("generation_node", nodes.generation_node)

        # Draw the workflow
        workflow.set_entry_point("user_query_classification_node")
        workflow.set_finish_point("generation_node")
        workflow.add_conditional_edges(
            "user_query_classification_node",
            nodes.is_retrieval_needed_cnode,
            {
                "retrieval_not_needed": "generation_node",
                "retrieval_needed_for_global_topic_searching": "retrieve_global_data_node",
                "retrieval_needed_for_local_topic_searching": "retrieve_local_data_node",
            }
        )
        workflow.add_conditional_edges("retrieve_global_data_node", nodes.dispatch_global_mapping_cnode, ["global_mapping_node"])
        workflow.add_conditional_edges("retrieve_local_data_node", nodes.dispatch_local_mapping_cnode, ["local_mapping_node"])
        workflow.add_edge("global_mapping_node", "global_reducing_node")
        workflow.add_edge("local_mapping_node", "local_reducing_node")
        workflow.add_edge("global_reducing_node", "retrieve_detail_data_node")
        workflow.add_edge("local_reducing_node", "retrieve_detail_data_node")
        workflow.add_conditional_edges("retrieve_detail_data_node", nodes.dispatch_detail_mapping_cnode, ["detail_mapping_node"])
        workflow.add_edge("detail_mapping_node", "detail_reducing_node")
        workflow.add_edge("detail_reducing_node", "information_organization_node")
        workflow.add_edge("information_organization_node", "generation_node")
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