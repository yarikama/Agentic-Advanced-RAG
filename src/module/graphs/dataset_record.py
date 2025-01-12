from langgraph.graph import StateGraph, END
from src.module.nodes import *
from src.utils import *
from config.output_pydantic import *
from src.module.graphs.hybrid_agentic_rag import HybridAgenticRAG

class DatasetRecord:
    def __init__(self):
        # Create the workflow(graph)
        workflow = StateGraph(QueriesState)
        
        # Add nodes class
        nodes = NodesDataset()
        
        # Add subgraph
        unit_query_subgraph = HybridAgenticRAG().graph
        
        # Add nodes into the workflow
        workflow.add_node("update_next_query_node", nodes.update_next_query_node)
        workflow.add_node("unit_query_subgraph", unit_query_subgraph)
        workflow.add_node("store_results_node", nodes.store_results_node)

        # Draw the workflow
        workflow.set_entry_point("update_next_query_node")
        workflow.add_edge("update_next_query_node", "unit_query_subgraph")
        workflow.add_edge("unit_query_subgraph", "store_results_node")
        workflow.add_conditional_edges(
            "store_results_node",
            nodes.is_dataset_unfinished_edges,
            {
                "dataset_unfinished": "update_next_query_node",
                "dataset_finished": END,
            }
        )
        # Compile
        self.graph = workflow.compile()
        
if __name__ == "__main__":
    workflow = DatasetRecord()
    print(workflow)