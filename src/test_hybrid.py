from Module import *


if __name__ == "__main__":
    workflow = WorkFlowModularHybridRAG()
    results = workflow.graph.stream({"repeat_times": 0})
    for result in results:
        print(result)