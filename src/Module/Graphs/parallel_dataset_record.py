from langgraph.graph import StateGraph
from src.Module.nodes import *
from src.Module.Graphs.hybrid_agentic_rag import HybridAgenticRAG
from src.Utils import *
from Config.output_pydantic import *


class ParallelDatasetRecord:
    def __init__(self):
        workflow = StateGraph(QueriesParallelState)
                              
        nodes = NodesParallelDataset()
                              
        unit_query_subgraph = HybridAgenticRAG().graph
                              
        workflow.add_node("prepare_batches_node", nodes.prepare_batches)
        workflow.add_node("unit_query_subgraph", unit_query_subgraph)
        workflow.add_node("sort_results_by_query_id", nodes.sort_results_by_query_id)

        workflow.set_entry_point("prepare_batches_node")
        workflow.set_finish_point("sort_results_by_query_id")
        workflow.add_conditional_edges("prepare_batches_node", nodes.dispatch_dataset_edges, ["unit_query_subgraph"])
        workflow.add_edge("unit_query_subgraph", "sort_results_by_query_id")
        self.graph = workflow.compile()
        
if __name__ == "__main__":
    workflow = ParallelDatasetRecord()
    print(workflow)