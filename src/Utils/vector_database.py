import pandas as pd
from Config import constants as const
from typing import List, Dict, Any
from pymilvus import WeightedRanker, RRFRanker, connections, FieldSchema, CollectionSchema, DataType, Collection, MilvusClient, AnnSearchRequest
import warnings

class VectorDatabase:
    _instance = None

    def __new__(
        cls, 
        host: str = const.MILVUS_HOST, 
        port: str = const.MILVUS_PORT, 
        database_name: str = const.MILVUS_DATABASE_NAME
    ) -> 'VectorDatabase':
        """
        This is a singleton class for the vector database.
        """
        if cls._instance is None:
            cls._instance = super(VectorDatabase, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self, 
        host: str = const.MILVUS_HOST, 
        port: str = const.MILVUS_PORT, 
        database_name: str = const.MILVUS_DATABASE_NAME
    ):
        if self._initialized:
            return
        self.host = host
        self.port = port
        self.database_name = database_name
        self.client = self.connect()
        self._initialized = True
        self.loaded_collections = set()  
        print("VectorDatabase initialized.")
        
    def connect(self) -> MilvusClient:
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

    def create_collection(
        self, 
        collection_name: str, 
        dense_dim: int = const.EMBEDDING_DENSE_DIM,
        is_document_mapping: bool = const.IS_DOCUMENT_MAPPING
    ):
        if self.client.has_collection(collection_name):
            warnings.warn(f"Collection {collection_name} already exists, skipping creation.")
            return

        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=dense_dim),
            FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
            FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=1024)
        ]
        
        print(f"Creating collection {collection_name} with fields: {fields}")
        
        if is_document_mapping:
            fields.append(FieldSchema(name="doc_id", dtype=DataType.STRING, max_length=512))
                                      
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
        
        print(f"Successfully created collection {collection_name} with dense embeddings (dense dim: {dense_dim}) and sparse embeddings.")

        if const.IS_GPU_INDEX:
            dense_index_params = {
                "metric_type": "L2",
                "index_type": "GPU_CAGRA",
                "params": {
                    'intermediate_graph_degree': 64,
                    'graph_degree': 32
                }
            }
        else:
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

    def create_collection_from_dataframe(
        self, 
        collection_name: str, 
        df: pd.DataFrame, 
        dense_dim: int = const.EMBEDDING_DENSE_DIM
    ):
        if self.client.has_collection(collection_name):
            raise ValueError(f"Collection {collection_name} already exists.")
        pass

    def load_collection(self, collection_name: str):
        collection = self.get_collection(collection_name)
        if collection_name in self.loaded_collections:
            print(f"Collection {collection_name} is already loaded.")
            return collection
        collection.load()
        self.loaded_collections.add(collection_name)
        print(f"Collection {collection_name} loaded into memory.")
        return collection

    def release_collection(self, collection_name: str):
        collection = self.get_collection(collection_name)
        collection.release()
        self.loaded_collections.discard(collection_name)  # 從已加載集合中移除
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

    def hybrid_search(
        self, 
        collection_name: str, 
        search_requests, 
        rerank_type: str = const.RERANK_TYPE,
        weights: List[float] = const.RERANK_WEIGHTS,    
        top_k: int = const.TOP_K,
    ) -> List[Dict[str, Any]]:
        """
        This function is used to perform a hybrid search on the vector database.
        It uses the WeightedRanker or RRFRanker to rerank the search results.
        
        Args:
            collection_name: str -> The name of the collection
            search_requests: List[Dict[str, Any]] -> The search requests
            rerank_type: str -> The type of reranking to use
            weights: List[float] -> The weights to use for the reranking
            top_k: int -> The number of the top results
        Returns:
            List[Dict[str, Any]] -> The search results
                - content: str -> The content of the document
                - metadata: Dict[str, Any] -> The metadata of the document
        """
        collection = self.load_collection(collection_name)
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
        
        collection = self.load_collection(collection_name)
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
    
    def get_all_entities(self, 
                         collection_name: str):
        """
        Retrieves all vectors from the specified collection and returns them as a pandas DataFrame + write into .parquet.
        Parameters:
            collection_name (str): The name of the collection to retrieve entities from.
        Returns:
            pd.DataFrame: A DataFrame containing all entities from the specified collection.
        """
        
        collection = self.load_collection(collection_name)
        iterator = collection.query_iterator(batch_size=16384,
                                             expr="id > 0",
                                             output_fields=["id","dense_vector","content","metadata"],)  

        all_data = []
        while True:
            result = iterator.next()
            if not result:
                iterator.close()
                break
            
            for hit in result:               
                record = {
                    "id": hit['id'],  # ID
                    "dense_vector": hit['dense_vector'],  
                    # "sparse_vector": hit['sparse_vector'],  
                    "content": hit['content'],  
                    "metadata": hit['metadata']
                }
                all_data.append(record)
        df = pd.DataFrame(all_data)
        df = df.sort_values(by='id').reset_index(drop=True)
        
        file_name = f'.parquet/{collection_name}_all_entities.parquet'
        df.to_parquet(file_name, index=False)
        
        return df
    
    def Kmeans_clustering(self, 
                          collection_name: str,
                          num_k,
                          batch_size,
                          limit):
            
        file_path = f'../tests/{collection_name}_all_entities.parquet'
        #read json get vector
        vectors = []
           
        df = pd.read_parquet(file_path)
        vector = df['dense_vector']
        for v in vector:
            v = v.tolist()
            vectors.append(v)
        vectors_array = np.array(vectors)
        
        #do kmeans
        kmeans = MiniBatchKMeans(n_clusters=num_k, random_state=0, batch_size=batch_size)
        kmeans.fit(vectors_array)
        cluster_centers = kmeans.cluster_centers_
        cluster_centers = cluster_centers.tolist()
        
        result = []
        for center in cluster_centers:
            #change format to search_format
            data = []
            center = np.array(center)
            data.append(center)
            
            #search center
            search_req = {
                "data": data,
                "anns_field": "dense_vector",
                "param": {"metric_type": "COSINE", "params": {}},
                "limit": limit
            }
            search_req = AnnSearchRequest(**search_req)
        
            res = self.search(
                collection_name=collection_name,
                search_request=search_req,
                top_k=limit
            )
            result.append(res)
            print(res)
    
        return result 
    
    def LSH_clustering(self,
                       num_layers, 
                       hash_size, 
                       table_num, 
                       sampling_ratio,
                       collection_name: str,
                       search_limit):
        
        collection = self.load_collection(collection_name)
        file_path = f'../tests/{collection_name}_all_entities.parquet'
        vectors = []
        df = pd.read_parquet(file_path)
        vector = df['dense_vector']
        for v in vector:
            v = v.tolist()
            vectors.append(v)
        vectors_array = np.array(vectors)
        
        # do LSH
        def lsh_layer(vectors, current_layer):

            if current_layer > num_layers:
                return [vectors]  # break

            dim = len(vectors[0])
            lsh = LSHash(hash_size=hash_size, input_dim=dim, num_hashtables=table_num)
            
            # store
            for ix, v in enumerate(vectors):
                lsh.index(v, extra_data=str(ix))
            
            next_buckets = []
            for table in lsh.hash_tables:
                for hash_value, bucket in table.storage.items():
                    bucket_vectors = [item[0] for item in bucket]
                    if len(bucket_vectors) > 0:
                        next_buckets.extend(lsh_layer(bucket_vectors, current_layer + 1))
            return next_buckets

        # start LSH
        final_buckets = lsh_layer(vectors_array, 1)
        
        # last layer
        representative_bucket_vectors = []
        representative_vectors = []
        for bucket in final_buckets:
            num_vectors_to_sample = max(1, int(len(bucket) * sampling_ratio))
            indices_to_sample = np.random.choice(len(bucket), num_vectors_to_sample, replace=False)
            sampled_vectors = [bucket[i] for i in indices_to_sample]
            representative_bucket_vectors.append(sampled_vectors)
            representative_vectors.extend(sampled_vectors)
        print(f"\n The vectors number by LSH-{num_layers} clustering : {len(representative_vectors)}")
        print(f" The buckets number by LSH-{num_layers} clustering : {len(representative_bucket_vectors)}")
            
        # search result
        result = []
        with open('search_results.txt', 'w') as file:
            for bucket in representative_bucket_vectors:
                meta_data = []
                for v in bucket:
                    data = []
                    v = np.array(v)
                    data.append(v)
                    
                    #search center 
                    search_req = {
                        "data": data,
                        "anns_field": "dense_vector",
                        "param": {"metric_type": "COSINE","params": {"ef": 100}},
                        "limit": search_limit
                    }
                    search_req = AnnSearchRequest(**search_req)
                    res = self.search(
                        collection_name=collection_name,
                        search_request=search_req,
                        top_k=search_limit
                    )
                    result.extend(res)
                    meta_data.append(res[0][0]['metadata'])
    
                    file.write(f"{res[0][0]['content']}\n [{res[0][0]['metadata']}]\n")
                file.write("\n\n\n\n==================================================================\n\n\n\n")
                
                # #count meta_data num
                # meta_data_count = Counter(meta_data)
                # for word, count in meta_data_count.items():
                #     print(f" {word}: {count}")
                # print("\n\n")
                  
        return result, representative_vectors, representative_bucket_vectors
    
    def GMM_clustering(self,
                       umap_componemts, 
                       gmm_components, 
                       sampling_ratio,
                       collection_name: str,
                       search_limit):
        
        collection = self.load_collection(collection_name)
        file_path = f'../tests/{collection_name}_all_entities.parquet'
        vectors = []
        df = pd.read_parquet(file_path)
        vector = df['dense_vector']
        for v in vector:
            v = v.tolist()
            vectors.append(v)
        vectors_array = np.array(vectors)
        
        # umap
        reducer = umap.UMAP(metric='cosine',n_components=umap_componemts)
        X_umap = reducer.fit_transform(vectors_array)
        print(f" Umap shape: {X_umap.shape}")  
        
        # GMM
        gmm = GaussianMixture(n_components=gmm_components)
        labels = gmm.fit_predict(X_umap)
        
        # sample
        representative_bucket_vectors = []
        representative_vectors = []
        for cluster in np.unique(labels):
            cluster_indices = np.where(labels == cluster)[0]
            sample_size = int(sampling_ratio * len(cluster_indices))
            sample_indices = np.random.choice(cluster_indices, sample_size, replace=False)
            representative_bucket_vectors.append(vectors_array[sample_indices])
            representative_vectors.extend(vectors_array[sample_indices])
            print(f"Cluster {cluster}: {len(sample_indices)}")
            
        # search result
        result = []
        with open('search_results.txt', 'w') as file:
            for bucket in representative_bucket_vectors:
                meta_data = []
                for v in bucket:
                    data = []
                    v = np.array(v)
                    data.append(v)
                    
                    #search center 
                    search_req = {
                        "data": data,
                        "anns_field": "dense_vector",
                        "param": {"metric_type": "COSINE","params": {"ef": 100}},
                        "limit": search_limit
                    }
                    search_req = AnnSearchRequest(**search_req)
                    res = self.search(
                        collection_name=collection_name,
                        search_request=search_req,
                        top_k=search_limit
                    )
                    
                    result.extend(res)
                    meta_data.append(res[0][0]['metadata'])
    
                    file.write(f"{res[0][0]['content']}\n [{res[0][0]['metadata']}]\n")
                file.write("\n\n\n\n==================================================================\n\n\n\n")
                
                # #count meta_data num
                # meta_data_count = Counter(meta_data)
                # for word, count in meta_data_count.items():
                #     print(f" {word}: {count}")
                # print("\n\n")
                  
        return result, representative_vectors, representative_bucket_vectors


if __name__ == "__main__":
    vectordatabase = VectorDatabase()
    vectordatabase.create_collection("test_collection123", dense_dim=1024)
    vectordatabase.load_collection("test_collection123")
    print(vectordatabase.list_collections())
    vectordatabase.drop_collection("test_collection123")
    vectordatabase.disconnect()

