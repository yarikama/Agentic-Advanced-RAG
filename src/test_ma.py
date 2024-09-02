from Utils import *
from pandas import DataFrame
from MultiAgent import MultiAgent_RAG
from MultiAgent import MultiAgent_RAG


retriever = Retriever()
communities_2 = retriever.retrieve_all_communities(0)
communities_2 = communities_2.values.tolist()
batch_size = 5
# Assign the community to the agents
batches = [{"batch_communities": communities_2[i:i + batch_size], "user_query": "What are the movies about love in this dataset?"} for i in range(0, len(communities_2), batch_size)] 
print(len(batches))

MARAG = MultiAgent_RAG()
results = MARAG.topic_reranking_run_batch_async(node_batch_inputs=batches)
print(results)
print(len(communities_2))
print(len(results))