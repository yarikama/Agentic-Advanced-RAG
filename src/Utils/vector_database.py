import pandas as pd
from . import constants as const
from typing import List, Dict, Any
from pymilvus import WeightedRanker, RRFRanker, connections, FieldSchema, CollectionSchema, DataType, Collection, MilvusClient

class VectorDatabase:
    def __init__(self, 
                host: str = const.MILVUS_HOST, 
                port: str = const.MILVUS_PORT, 
                database_name: str = const.MILVUS_DATABASE_NAME
                ):
        
        self.host = host
        self.port = port
        self.database_name = database_name
        self.client = self.connect()
        print("VectorDatabase initialized.")
        
    def connect(self):
        connections.connect(
            host=self.host, 
            port=self.port, 
            db_name=self.database_name
        )
        print(f"Connected to Milvus at {self.host}:{self.port} with database {self.database_name}.")
        return MilvusClient(uri = f"http://{self.host}:{self.port}")
        
    def disconnect(self):
        connections.disconnect("default")
        print("Disconnected from Milvus.")

    def get_collection(self, collection_name: str):
        if self.client.has_collection(collection_name):
            return Collection(collection_name)
        raise ValueError(f"Collection {collection_name} does not exist.")

    def list_collections(self):
        return self.client.list_collections()

    def create_collection(self, 
                        collection_name: str, 
                        dense_dim: int = const.EMBEDDING_DENSE_DIM
                        ):

        if self.client.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} already exists.")

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=512)
        ]
        
        schema = CollectionSchema(
            fields, 
            description=f"Hybrid collection for dense and sparse vector embeddings of {collection_name}.",
            enable_dynamic_fields=True,
        )
        
        collection = Collection(
            name=collection_name, 
            schema=schema, 
            using=self.database_name
        )
        
        print(f"Successfully created collection {collection_name} with dense dimension {dense_dim} and sparse embeddings.")

        dense_index_params = {
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 200}
        } 
        
        sparse_index_params = {
            "metric_type": "IP",
            "index_type": "SPARSE_INVERTED_INDEX",
            "params": {"drop_ratio_build": 0.2}
        }
        
        collection.create_index("dense_vector", dense_index_params)
        collection.create_index("sparse_vector", sparse_index_params)
        print(f"Successfully created indexes for collection {collection_name}.")

    def create_collection_from_dataframe(self, 
                                        collection_name: str, 
                                        df: pd.DataFrame, 
                                        dense_dim: int = const.EMBEDDING_DENSE_DIM
                                        ):
        
        if self.client.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} already exists.")
        pass

    def load_collection(self, collection_name: str):
        self.client.load_collection(collection_name=collection_name)
        print(f"Collection {collection_name} loaded into memory.")

    def release_collection(self, collection_name: str):
        collection = self.get_collection(collection_name)
        collection.release()
        print(f"Collection {collection_name} released from memory.")

    def drop_collection(self, collection_name: str):
        collection = self.get_collection(collection_name)
        collection.drop()
        print(f"Collection {collection_name} dropped.")

    def insert_data(self, 
                    collection_name: str, 
                    data: List[Dict[str, any]]
                    ) -> None:
        
        collection = self.get_collection(collection_name)
        collection.insert(data)
        print("Inserted data into Milvus.")

    def get_content_from_hits(self, 
                            hits: List[Dict[str, Any]]
                            ) -> List[Dict[str, Any]]:
        return [
            {
                "content": hit.entity.get("content"),
                "metadata": hit.entity.get("metadata"),
                # "distance": hit.distance,
                # "id": hit.id
            }
            for hit in hits
        ]

    def hybrid_search(self, 
                    collection_name: str, 
                    search_requests, 
                    rerank_type: str = const.RERANK_TYPE,
                    weights: List[float] = const.RERANK_WEIGHTS,    
                    top_k: int = const.TOP_K
                    ) -> List[Dict[str, Any]]:

        collection = self.get_collection(collection_name)
        collection.load()

        rerank = WeightedRanker(*weights) if rerank_type == "weighted" else RRFRanker()
        
        milvus_results = collection.hybrid_search(
            search_requests, 
            rerank, 
            limit=top_k, 
            output_fields=["content", "metadata"]
        )
        
        contents = [self.get_content_from_hits(hits) for hits in milvus_results]

        return contents
    
    def search(self, 
               collection_name: str,
               search_request,
               top_k: int = const.TOP_K
               ) -> List[Dict[str, Any]]:
        
        collection = self.get_collection(collection_name)
        collection.load()
        
        search_params = {
            "data": search_request.data,
            "anns_field": search_request.anns_field,
            "param": search_request.param,
            "limit": top_k,
            "output_fields": ["content", "metadata"]
        }
        
        milvus_results = collection.search(**search_params)        
        
        contents = [self.get_content_from_hits(hits) for hits in milvus_results]

        return contents


if __name__ == "__main__":
    vectordatabase = VectorDatabase()
    vectordatabase.create_collection("test_collection123", dense_dim=1024)
    vectordatabase.load_collection("test_collection123")
    print(vectordatabase.list_collections())
    vectordatabase.drop_collection("test_collection123")
    vectordatabase.disconnect()