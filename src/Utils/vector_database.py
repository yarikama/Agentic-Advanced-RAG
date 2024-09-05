import pandas as pd
import os
import numpy as np
import umap.umap_ as umap
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from lshashpy3 import LSHash
from collections import Counter
from . import constants as const
from typing import List, Dict, Any
from pymilvus import WeightedRanker, RRFRanker, connections, FieldSchema, CollectionSchema, DataType, Collection, MilvusClient, AnnSearchRequest

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
    
    def get_all_entities(self, 
                         collection_name: str):
        """
        Retrieves all vectors from the specified collection and returns them as a pandas DataFrame + write into .parquet.
        Parameters:
            collection_name (str): The name of the collection to retrieve entities from.
        Returns:
            pd.DataFrame: A DataFrame containing all entities from the specified collection.
        """
        
        collection = self.get_collection(collection_name)
        collection.load()
        
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
        
        file_name = f'../tests/{collection_name}_all_entities.parquet'
        df.to_parquet(file_name, index=False)
        
        return df
    
    def Clustering_result_output(self,
                                 collection_name: str,
                                 labels,
                                 sampling_ratio,
                                 vectors_array,
                                 method
                                 ):
        """
        Result output : including sorting, sampling, searching, and writing to a text file.
        Args:
            collection_name (str): The name of the collection.
            labels: The labels(list) of the clusters.
            sampling_ratio: The ratio used for sampling.
            vectors_array: The array of all collection vectors.
            method: The clustering method.
        Returns:
            result: The final searching result for vectors ("content", "metadata").
            vectors : Only vectors (1-dim list).
            cluster_vectors : The vectors in each cluster list (2-dim list).
        """
        collection = self.get_collection(collection_name)
        collection.load()
        print(f"\n # Collection name : {collection_name}")
        print(f" # Clustering method : {method} ")
        print(f" # Number of clustering : {len(np.unique(labels))} ")
        
        # sample
        cluster_vectors = []
        vectors = []
        count = 0
        for cluster in np.unique(labels):
            cluster_indices = np.where(labels == cluster)[0]
            sample_size = int(sampling_ratio * len(cluster_indices))
            sample_indices = np.random.choice(cluster_indices, sample_size, replace=False)
            cluster_vectors.append(vectors_array[sample_indices])
            vectors.extend(vectors_array[sample_indices])
            if count <= 9:
                print(f"   - Cluster {cluster}: {len(sample_indices)}")
                count += 1
            elif count == 10:
                print('   - ...')
                count += 1  
                        
        # search result
        result = []
        ouput_file = f'.sparse_graph_txt/{method}_search_results.txt'
        with open(ouput_file, 'w') as file:
            for bucket in cluster_vectors:
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
                        "limit": 1
                    }
                    search_req = AnnSearchRequest(**search_req)
                    res = self.search(
                        collection_name=collection_name,
                        search_request=search_req,
                        top_k=1
                    )
                    
                    result.extend(res)
                    meta_data.append(res[0][0]['metadata'])
    
                    file.write(f"{res[0][0]['content']}\n [{res[0][0]['metadata']}]\n")
                
        return result, vectors, cluster_vectors
        
    def Kmeans_clustering(self,
                          umap_componemts=10, 
                          k_components=200, 
                          kmeans_batch_size=1000,
                          sampling_ratio=0.1,
                          collection_name: str=''):
        """
        Perform Kmeans clustering on a collection.
        The results of content and metadata are saved in a text file.
        Args:
            umap_componemts (int): The number of components for UMAP dimensionality reduction.
            k_components (int): The number of components for Kmeans clustering.
            kmeans_batch_size (int): The batch size for MiniBatchKmeans.
            sampling_ratio (float): The ratio of samples to be used for clustering.
            collection_name (str): The name of the collection.
        Returns:
            result: The final searching result for vectors ("content", "metadata").
            vectors : Only vectors (1-dim list).
            cluster_vectors : The vectors in each cluster list (2-dim list).
        """
            
        file_path = f'../tests/{collection_name}_all_entities.parquet'
        temp_vectors = []   
        df = pd.read_parquet(file_path)
        vector = df['dense_vector']
        for v in vector:
            v = v.tolist()
            temp_vectors.append(v)
        vectors_array = np.array(temp_vectors)
        
        # umap
        reducer = umap.UMAP(metric='cosine',n_components=umap_componemts)
        X_umap = reducer.fit_transform(vectors_array) 
        
        #do kmeans
        kmeans = MiniBatchKMeans(n_clusters=k_components, random_state=0, batch_size=kmeans_batch_size)
        labels = kmeans.fit_predict(X_umap)
        
        result, vectors, cluster_vectors = self.Clustering_result_output(collection_name=collection_name, 
                                                                         labels=labels, 
                                                                         sampling_ratio=sampling_ratio, 
                                                                         vectors_array=vectors_array,
                                                                         method='Kmeans')
                  
        return result, vectors, cluster_vectors
    
    def LSH_clustering(self,
                       num_layers=1, 
                       hash_size=8 , 
                       table_num=1 , 
                       sampling_ratio=0.1, 
                       collection_name: str=''):
        """
        Perform LSH clustering on a collection.
        The results of content and metadata are saved in a text file.
        Args:
            num_layers (int): The number of layers in the LSH structure.
            hash_size (int): The size of the hash functions (num of clustering 2^k) used in LSH.
            table_num (int): The number of hash tables used in LSH.
            sampling_ratio (float): The ratio of samples to be used for clustering.
            collection_name (str): The name of the collection.
        Returns:
            result: The final searching result for vectors ("content", "metadata").
            vectors : Only vectors (1-dim list).
            cluster_vectors : The vectors in each cluster list (2-dim list).
        """
        
        file_path = f'../tests/{collection_name}_all_entities.parquet'
        temp_vectors = []
        df = pd.read_parquet(file_path)
        vector = df['dense_vector']
        for v in vector:
            v = v.tolist()
            temp_vectors.append(v)
        vectors_array = np.array(temp_vectors)
        
        # do LSH
        labels = []
        global labels_counter
        labels_counter = 0
        def lsh_layer(vectors, current_layer):
            global labels_counter

            if current_layer > num_layers:
                return vectors  # break

            dim = len(vectors[0])
            lsh = LSHash(hash_size=hash_size, input_dim=dim, num_hashtables=table_num)
            
            # store
            for ix, v in enumerate(vectors):
                lsh.index(v, extra_data=str(ix))
            
            next_buckets = []
            for table in lsh.hash_tables:
                for hash_value, bucket in table.storage.items():
                    bucket_vectors = [list(item)[0] for item in bucket]
                    if len(bucket_vectors) > 0:
                        labels.extend([labels_counter] * len(bucket_vectors))
                        labels_counter += 1
                        next_buckets.extend(lsh_layer(bucket_vectors, current_layer + 1))            
            return next_buckets
        
        # start LSH
        final_buckets = lsh_layer(vectors_array, 1)
        final_buckets = np.array(final_buckets)
    
        result, vectors, cluster_vectors = self.Clustering_result_output(collection_name=collection_name, 
                                                                         labels=labels, 
                                                                         sampling_ratio=sampling_ratio, 
                                                                         vectors_array=final_buckets,
                                                                         method='LSH')
                  
        return result, vectors, cluster_vectors
    
    def GMM_clustering(self,
                       umap_componemts=10, 
                       gmm_components=200, 
                       sampling_ratio=0.1,
                       collection_name: str=''):
        """
        Perform GMM clustering on a collection.
        The results of content and metadata are saved in a text file.
        Args:
            umap_componemts (int): The number of components for UMAP dimensionality reduction.
            gmm_components (int): The number of components for Gaussian Mixture Model.
            sampling_ratio (float): The ratio of samples to be used for clustering.
            collection_name (str): The name of the collection.
        Returns:
            result: The final searching result for vectors ("content", "metadata").
            vectors : Only vectors (1-dim list).
            cluster_vectors : The vectors in each cluster list (2-dim list).
        """
        
        file_path = f'../tests/{collection_name}_all_entities.parquet'
        temp_vectors = []
        df = pd.read_parquet(file_path)
        vector = df['dense_vector']
        for v in vector:
            v = v.tolist()
            temp_vectors.append(v)
        vectors_array = np.array(temp_vectors)
        
        # umap
        reducer = umap.UMAP(metric='cosine',n_components=umap_componemts)
        X_umap = reducer.fit_transform(vectors_array) 
        
        # GMM
        gmm = GaussianMixture(n_components=gmm_components)
        labels = gmm.fit_predict(X_umap)
        
        result, vectors, cluster_vectors = self.Clustering_result_output(collection_name=collection_name, 
                                                                         labels=labels, 
                                                                         sampling_ratio=sampling_ratio, 
                                                                         vectors_array=vectors_array,
                                                                         method='GMM')
                  
        return result, vectors, cluster_vectors
    
    def DBSCAN_clustering(self,
                          umap_componemts=10,
                          eps=0.0001,
                          min_samples=40,
                          sampling_ratio=0.1,
                          collection_name: str=''):
        """
        Perform DBSCAN clustering on a collection.
        The results of content and metadata are saved in a text file.
        Args:
            umap_componemts (int): The number of components for UMAP dimensionality reduction.
            eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood.
            min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.
            sampling_ratio (float): The ratio of samples to be used for clustering.
            collection_name (str): The name of the collection.
        Returns:
            result: The final searching result for vectors ("content", "metadata").
            vectors : Only vectors (1-dim list).
            cluster_vectors : The vectors in each cluster list (2-dim list).
        """
        
        file_path = f'../tests/{collection_name}_all_entities.parquet'
        temp_vectors = []
        df = pd.read_parquet(file_path)
        vector = df['dense_vector']
        for v in vector:
            v = v.tolist()
            temp_vectors.append(v)
        vectors_array = np.array(temp_vectors)
        
        # umap
        reducer = umap.UMAP(metric='cosine',n_components=umap_componemts)
        X_umap = reducer.fit_transform(vectors_array) 
        
        # GMM
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = dbscan.fit_predict(X_umap)
        
        result, vectors, cluster_vectors = self.Clustering_result_output(collection_name=collection_name, 
                                                                         labels=labels, 
                                                                         sampling_ratio=sampling_ratio, 
                                                                         vectors_array=vectors_array,
                                                                         method='DBSCAN')
                  
        return result, vectors, cluster_vectors


if __name__ == "__main__":
    vectordatabase = VectorDatabase()
    vectordatabase.create_collection("test_collection123", dense_dim=1024)
    vectordatabase.load_collection("test_collection123")
    print(vectordatabase.list_collections())
    vectordatabase.drop_collection("test_collection123")
    vectordatabase.disconnect()