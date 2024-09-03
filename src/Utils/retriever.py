import os
from openai import OpenAI
from dotenv import load_dotenv
from .embedder import Embedder
from Config import constants as const
from pymilvus import AnnSearchRequest
from .vector_database import VectorDatabase
from .knowledge_graph_database import KnowledgeGraphDatabase
from typing import List, Dict, Any, Union, Optional, Set
load_dotenv()

class Retriever:
    def __init__(self, 
                vectordatabase: Optional[VectorDatabase] = None,
                graphdatabase: Optional[KnowledgeGraphDatabase] = None, 
                embedder: Optional[Embedder] = None,):
        
        self.embedder = embedder if embedder else Embedder()
        self.vectordatabase = vectordatabase if vectordatabase else VectorDatabase()
        self.graphdatabase = graphdatabase if graphdatabase else KnowledgeGraphDatabase()
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
    
    def dense_search_request(self, 
                            dense_query_vectors: Union[List[float], List[List[float]]], 
                            field_name: str, top_k: int = const.TOP_K) -> List[Dict[str, Any]]:
        
        if const.IS_GPU_INDEX:
            dense_search_param = {
                "data": dense_query_vectors,
                "anns_field": field_name,
                "param": {
                    "metric_type": "L2",
                    "params": {
                        "itopk_size": 128,
                        "search_width": 4,
                        "min_iterations": 0,
                        "max_iterations": 0,
                        "team_size": 0
                    },
                },
                "limit": top_k
            }
        else:
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

    def sparse_search_request(self, 
                              sparse_query_vectors: Union[List[float], List[List[float]]], 
                              field_name: str, top_k: int = const.TOP_K) -> List[Dict[str, Any]]:
        
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

    def hybrid_retrieve(self, 
                        collection_name: str, 
                        query_texts: List[str], 
                        top_k: int = const.TOP_K, 
                        alpha: float = const.ALPHA, 
                        isHyDE: bool = False) -> List[List[Dict[str, Any]]]:
        """
        this is for similar multiple queries searching, the result is a deduplicated list of documents
        """
        dense_queries = self.embedder.embed_dense(query_texts)
        sparse_queries = self.embedder.embed_sparse(collection_name, query_texts)
        
        if isHyDE:
            hypothetical_docs = [self.generate_hypothetical_document(query) for query in query_texts]
            hyde_dense_queries = self.embedder.embed_dense(hypothetical_docs)
            hyde_sparse_queries = self.embedder.embed_sparse(collection_name, hypothetical_docs)            
            dense_queries = hyde_dense_queries + dense_queries
            sparse_queries = hyde_sparse_queries + sparse_queries
        
        hybrid_search_requests = self.hybrid_search_request(dense_queries, sparse_queries)        
        batch_results = self.vectordatabase.hybrid_search(collection_name, hybrid_search_requests, "weighted", [1 - alpha, alpha], top_k * len(query_texts))
        
        seen_contents = set()
        unique_results = []
        
        for query_results in batch_results:
            for result in query_results:
                content = result['content']
                if content not in seen_contents:
                    seen_contents.add(content)
                    unique_results.append(result)
        
        return unique_results 
    
    def dense_retrieve(self, collection_name: str, query_text: str, top_k: int = const.TOP_K) -> List[Dict[str, Any]]:
        dense_query = self.embedder.embed_dense(query_text)
        dense_search_request = self.dense_search_request(dense_query, "dense_vector")
        results = self.vectordatabase.search(collection_name, dense_search_request, top_k)
        return results
    
    def retrieve_all_communities(self, level: int) -> str:
        community_data = self.graphdatabase.db_query(
        """
        MATCH (c:__Community__)
        WHERE c.level = $level
        RETURN c.full_content AS output
        """,
            params={"level": level},
        )
        return community_data
    