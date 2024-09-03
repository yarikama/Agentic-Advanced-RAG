from Utils import *
from pandas import DataFrame
from MultiAgent import MultiAgent_RAG
from MultiAgent import MultiAgent_RAG
import json

retriever = Retriever()
communities_2 = retriever.retrieve_all_communities(2)
communities_2 = communities_2.values.tolist()
communities_dict = {f"community_{i}": community for i, community in enumerate(communities_2)}
communities_json = json.dumps(communities_dict, ensure_ascii=False, indent=2)

print(communities_json)

batch_size = 10
user_query = "What are the movies about love in this dataset?"

batches = []
for i in range(0, len(communities_2), batch_size):
    batch_communities = {f"community_{j}": community 
                         for j, community in enumerate(communities_2[i:i + batch_size], start=i)}
    batch_json = json.dumps(batch_communities, ensure_ascii=False)
    batches.append({
        "batch_communities": batch_json,
        "user_query": user_query,
        "batch_size": len(batch_communities)
    })

MA = MultiAgent_RAG()
results = MA.topic_reranking_run_batch_async(node_batch_inputs=batches)
print("len(results):", len(results))
print("len(communities_2):", len(communities_2))
assert len(communities_2) == len(results)

