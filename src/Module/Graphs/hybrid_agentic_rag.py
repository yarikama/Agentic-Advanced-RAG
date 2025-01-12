from langgraph.graph import StateGraph
from src.Module.nodes import *
from src.Utils import *
from Config.output_pydantic import *

                
class HybridAgenticRAG:
    def __init__(self):
        # Create the workflow(graph)
        workflow = StateGraph(UnitQueryState)
        
        # Add nodes class
        nodes = NodesModularRAG()
        
        # Add nodes into the workflow
        workflow.add_node("user_query_classification_node", nodes.user_query_classification_node)
        
        workflow.add_node("global_retrieval_node", nodes.global_retrieval_node)
        workflow.add_node("local_retrieval_node", nodes.local_retrieval_node)
        workflow.add_node("detail_retrieval_node", nodes.detail_retrieval_node)
        
        workflow.add_node("local_reranking_node", nodes.local_reranking_node)
        workflow.add_node("detail_reranking_node", nodes.detail_reranking_node)
        workflow.add_node("local_reducing_node", nodes.local_reducing_node)
        workflow.add_node("detail_reducing_node", nodes.detail_reducing_node)
            
        workflow.add_node("global_mapping_node", nodes.global_mapping_node)
        workflow.add_node("global_reducing_node", nodes.global_reducing_node)
        
        workflow.add_node("local_community_retrieval_node", nodes.local_community_retrieval_node)
        workflow.add_node("local_hyde_node", nodes.local_hyde_node)

        workflow.add_node("information_organization_node", nodes.information_organization_node)
        workflow.add_node("generation_node", nodes.generation_node)
        
        # Draw the workflow
        workflow.set_entry_point("user_query_classification_node")
        workflow.add_conditional_edges(
            "user_query_classification_node",
            nodes.is_retrieval_needed_edges,
            {
                "retrieval_not_needed": "generation_node",
                "retrieval_needed_for_global_topic_searching": "global_retrieval_node",
                "retrieval_needed_for_local_topic_searching": "local_retrieval_node",
            }
        )
        workflow.add_conditional_edges("global_retrieval_node", nodes.global_mapping_dispatch_edges, ["global_mapping_node"])
        workflow.add_edge("local_retrieval_node", "local_reranking_node")
        workflow.add_edge("global_mapping_node", "global_reducing_node")
        workflow.add_edge("global_reducing_node", "detail_retrieval_node")
        workflow.add_edge("local_reranking_node", "local_reducing_node")
        workflow.add_conditional_edges("local_reducing_node", 
            nodes.is_local_retrieval_empty_edges, 
            {
                "local_retrieval_not_empty": "detail_retrieval_node",
                "local_retrieval_empty": "local_community_retrieval_node",
            }
        )
        workflow.add_conditional_edges("local_community_retrieval_node", 
            nodes.is_local_retrieval_empty_edges, 
            {
                "local_retrieval_not_empty": "detail_retrieval_node",
                "local_retrieval_empty": "local_hyde_node",
            }
        )
        workflow.add_edge("local_hyde_node", "detail_retrieval_node")
        workflow.add_edge("detail_retrieval_node", "detail_reranking_node")
        workflow.add_edge("detail_reranking_node", "detail_reducing_node")
        workflow.add_edge("detail_reducing_node", "information_organization_node")
        workflow.add_edge("information_organization_node", "generation_node")

        # Compile
        self.graph = workflow.compile()
        
if __name__ == "__main__":
    workflow = HybridAgenticRAG()
    print(workflow)