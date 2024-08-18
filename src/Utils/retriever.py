import os
from openai import OpenAI
from dotenv import load_dotenv
from .embedder import Embedder
from Config import constants as const
from pymilvus import AnnSearchRequest
from .vector_database import VectorDatabase
from typing import List, Dict, Any, Union, Optional
load_dotenv()

class Retriever:
    def __init__(self, vectordatabase: Optional[VectorDatabase] = None, embedder: Optional[Embedder] = None):
        self.embedder = embedder if embedder else Embedder()
        self.vectordatabase = vectordatabase if vectordatabase else VectorDatabase()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.openai_api_key)
        print("Retriever initialized")

    def generate_hypothetical_document(self, query: str) -> str:
        response = self.client.chat.completions.create(
            model=const.MODEL_NAME,  
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates a hypothetical document based on a given query."},
                {"role": "user", "content": f"Generate a short, relevant document that could answer the query in less than 60 words: {query}"}
            ]
        )
        return response.choices[0].message.content
    
    def dense_search_request(self, dense_query_vectors: Union[List[float], List[List[float]]], field_name: str, top_k: int = const.TOP_K) -> List[Dict[str, Any]]:
        
        dense_search_param = {
            "data": dense_query_vectors,
            "anns_field": field_name,
            "param": {
                "metric_type": "COSINE",
                "params": {"ef": 100}
            },
            "limit": top_k
        }

        dense_search_request = AnnSearchRequest(**dense_search_param)
        return dense_search_request 

    def sparse_search_request(self, sparse_query_vectors: Union[List[float], List[List[float]]], field_name: str, top_k: int = const.TOP_K) -> List[Dict[str, Any]]:
        
        sparse_search_param = {
            "data": sparse_query_vectors,
            "anns_field": field_name,
            "param": {
                "metric_type": "IP",
                "params": {"drop_ratio_search": 0.1}                
            },
            "limit": top_k
        }

        sparse_search_request = AnnSearchRequest(**sparse_search_param)
        return sparse_search_request
    
    def hybrid_search_request(self, dense_query_vectors: Union[List[float], List[List[float]]], sparse_query_vectors: Union[List[float], List[List[float]]]):
        dense_search_request = self.dense_search_request(dense_query_vectors, "dense_vector")
        sparse_search_request = self.sparse_search_request(sparse_query_vectors, "sparse_vector")
        hybrid_search_requests = [dense_search_request, sparse_search_request]
        return hybrid_search_requests

    def hybrid_retrieve(self, collection_name: str, query_text: str, top_k: int = const.TOP_K, alpha: float = const.ALPHA) -> List[Dict[str, Any]]:
        # HyDE
        hypothetical_doc = self.generate_hypothetical_document(query_text)
        print(f"hypothetical document generated.")
        dense_query = self.embedder.embed_dense(hypothetical_doc)
        sparse_query = self.embedder.embed_sparse(collection_name, hypothetical_doc)
        hybrid_search_requests_HyDE = self.hybrid_search_request(dense_query, sparse_query)
        results = self.vectordatabase.hybrid_search(collection_name, hybrid_search_requests_HyDE, "weighted", [1 - alpha, alpha], max(top_k - int(top_k / 2), 1))
        print(f"""hybrid search with HyDE (dense search weight of {1 - alpha} and sparse search weight of {alpha})""")
        # Original
        dense_query = self.embedder.embed_dense(query_text)
        sparse_query = self.embedder.embed_sparse(collection_name, query_text)
        hybrid_search_requests_origin = self.hybrid_search_request(dense_query, sparse_query)
        # results = self.vectordatabase.hybrid_search(collection_name, hybrid_search_requests_origin, "weighted", [1 - alpha, alpha], top_k)
        results.extend(self.vectordatabase.hybrid_search(collection_name, hybrid_search_requests_origin, "weighted", [1 - alpha, alpha], max(1, int(top_k / 2))))
        print(f"""hybrid search with original (dense search weight of {1 - alpha} and sparse search weight of {alpha})""")
        # print(f"Successfully retrieved {len(results)} results")
        
        return results
    
    def dense_retrieve(self, collection_name: str, query_text: str, top_k: int = const.TOP_K) -> List[Dict[str, Any]]:
        dense_query = self.embedder.embed_dense(query_text)
        dense_search_request = self.dense_search_request(dense_query, "dense_vector")
        results = self.vectordatabase.search(collection_name, dense_search_request, top_k)
        return results
        
if __name__ == "__main__":
    vectordatabase = VectorDatabase()
    embedder = Embedder()
    retriever = Retriever(vectordatabase, embedder)
    print(retriever.dense_retrieve("RAG", "What is RAG?"))
    
    