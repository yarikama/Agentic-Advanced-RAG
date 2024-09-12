import pandas as pd
import numpy as np
import umap.umap_ as umap
from .vector_database import VectorDatabase
from sklearn.cluster import MiniBatchKMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import pairwise_distances
from hdbscan import HDBSCAN
from lshashpy3 import LSHash
from typing import Optional

class Clustering:
    def __init__(self,
                 vectordatabase: Optional[VectorDatabase] = None) -> None:
        
        self.vectordatabase = vectordatabase if vectordatabase else VectorDatabase()
        print("Clustering initialized")
        
    def clustering_result_output(self,
                                 collection_name: str,
                                 labels,
                                 sampling_ratio,
                                 data,
                                 method
                                 ):
        """
        Result output : including sorting, sampling, searching, and writing to a text file.
        Args:
            collection_name (str): The name of the collection.
            labels: The labels(list) of the clusters.
            sampling_ratio: The ratio used for sampling.
            data:  All vectors and contents.
            method: The clustering method.
        Returns:
            result: The final searching result for vectors ("content", "metadata").
            vectors : Only vectors (1-dim list).
            cluster_vectors : The vectors in each cluster list (2-dim list).
        """
        collection = self.vectordatabase.get_collection(collection_name)
        collection.load()
        print(f"\n # Collection name : {collection_name}")
        print(f" # Clustering method : {method} ")
        print(f" # Number of clustering : {len(np.unique(labels))} ")
        
        # sample
        cluster_vectors = []
        vectors = []
        result_df = pd.DataFrame(columns=['dense_vector','content', 'metadata'])
        count = 0
        ouput_file = f'.sparse_graph_txt/{method}_search_results.txt'
        
        with open(ouput_file, 'w') as file:
            for cluster in np.unique(labels):
                cluster_indices = np.where(labels == cluster)[0]
                sample_size = int(sampling_ratio * len(cluster_indices))
                sample_indices = np.random.choice(cluster_indices, sample_size, replace=False)
                cluster_vectors.append(data.loc[sample_indices,'dense_vector'].values)
                result_df = pd.concat([result_df,data.loc[sample_indices]], ignore_index=True)
                vectors.extend(data.loc[sample_indices,'dense_vector'])
                if count <= 9:
                    print(f"   - Cluster {cluster}: {len(sample_indices)}")
                    count += 1
                elif count == 10:
                    print('   - ...')
                    count += 1
                
                for index in sample_indices:
                    content = data.loc[index, 'content']
                    metadata = data.loc[index, 'metadata']
                    file.write(f"{content}\n [{metadata}]\n")
                
        print(f" # Number of sampling vectors : {len(vectors)} ") 
                        
        return result_df, vectors, cluster_vectors
    
    def load_collection(self, 
                        collection_name: str):
        
        file_path = f'.parquet/{collection_name}_all_entities.parquet'
        temp_vectors = []   
        df = pd.read_parquet(file_path)
        data = df[['dense_vector', 'content','metadata']]
        for v in data['dense_vector']:
            v = v.tolist()
            temp_vectors.append(v)
        vectors_array = np.array(temp_vectors)
        return data, vectors_array
        
    def kmeans_clustering(self,
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
        
        data, vectors_array = self.load_collection(collection_name)    
        
        # umap
        reducer = umap.UMAP(metric='cosine',n_components=umap_componemts)
        X_umap = reducer.fit_transform(vectors_array) 

        #do kmeans
        kmeans = MiniBatchKMeans(n_clusters=k_components, random_state=0, batch_size=kmeans_batch_size)
        labels = kmeans.fit_predict(X_umap)
        
        result, vectors, cluster_vectors = self.clustering_result_output(collection_name=collection_name, 
                                                                         labels=labels, 
                                                                         sampling_ratio=sampling_ratio, 
                                                                         data=data,
                                                                         method='Kmeans')
                  
        return result, vectors, cluster_vectors

    def lsh_clustering(self,
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
        
        data, vectors_array = self.load_collection(collection_name)
        
        # do LSH
        labels = []
        global labels_counter
        labels_counter = 0
        def lsh_layer(vectors, current_layer, original_indices):
            global labels_counter

            if current_layer > num_layers:
                return vectors, original_indices  # break

            dim = len(vectors[0])
            lsh = LSHash(hash_size=hash_size, input_dim=dim, num_hashtables=table_num)
            
            # store
            for ix, v in enumerate(vectors):
                lsh.index(v, extra_data=str(original_indices[ix]))
            
            next_buckets = []
            next_indices = []
            for table in lsh.hash_tables:
                for hash_value, bucket in table.storage.items():
                    bucket_vectors = [list(item)[0] for item in bucket]
                    bucket_indices = [int(item[1]) for item in bucket]
                    if len(bucket_vectors) > 0:
                        labels.extend([labels_counter]*len(bucket_vectors))
                        labels_counter += 1
                        next_buckets.extend(bucket_vectors)
                        next_indices.extend(bucket_indices)
            return lsh_layer(next_buckets, current_layer + 1, next_indices)
        
        # start LSH
        original_indices = list(range(len(vectors_array)))
        final_buckets, final_indices = lsh_layer(vectors_array, 1, original_indices)
        final_buckets = np.array(final_buckets)
        
        sorted_data = data.iloc[final_indices].reset_index(drop=True)

        result, vectors, cluster_vectors = self.clustering_result_output(collection_name=collection_name, 
                                                                         labels=labels, 
                                                                         sampling_ratio=sampling_ratio, 
                                                                         data=sorted_data,
                                                                         method='LSH')
                  
        return result, vectors, cluster_vectors
    
    def gmm_clustering(self,
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
        
        data, vectors_array = self.load_collection(collection_name)
        
        # umap
        reducer = umap.UMAP(metric='cosine',n_components=umap_componemts)
        X_umap = reducer.fit_transform(vectors_array) 
        
        # GMM
        gmm = GaussianMixture(n_components=gmm_components)
        labels = gmm.fit_predict(X_umap)
        
        result, vectors, cluster_vectors = self.clustering_result_output(collection_name=collection_name, 
                                                                         labels=labels, 
                                                                         sampling_ratio=sampling_ratio, 
                                                                         data=data,
                                                                         method='GMM')
                  
        return result, vectors, cluster_vectors
    
    def dbscan_clustering(self,
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
        
        data, vectors_array = self.load_collection(collection_name)
        
        # umap
        reducer = umap.UMAP(metric='cosine',n_components=umap_componemts)
        X_umap = reducer.fit_transform(vectors_array) 
        
        # dbscan
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = dbscan.fit_predict(X_umap)
        
        result, vectors, cluster_vectors = self.clustering_result_output(collection_name=collection_name, 
                                                                         labels=labels, 
                                                                         sampling_ratio=sampling_ratio, 
                                                                         data=data,
                                                                         method='DBSCAN')
                  
        return result, vectors, cluster_vectors
    
    def hdbscan_clustering(self,
                           umap_componemts=10,
                           min_cluster_size=50,
                           sampling_ratio=0.1,
                           collection_name: str=''):
        """
        Perform HDBSCAN clustering on a collection.
        The results of content and metadata are saved in a text file.
        Args:
            umap_componemts (int): The number of components for UMAP dimensionality reduction.
            min_cluster_size (int): The minimum number of samples in a cluster.
            sampling_ratio (float): The ratio of samples to be used for clustering.
            collection_name (str): The name of the collection.
        Returns:
            result: The final searching result for vectors ("content", "metadata").
            vectors : Only vectors (1-dim list).
            cluster_vectors : The vectors in each cluster list (2-dim list).
        """
        
        data, vectors_array = self.load_collection(collection_name)
        
        # umap
        reducer = umap.UMAP(metric='cosine',n_components=umap_componemts)
        X_umap = reducer.fit_transform(vectors_array) 
        
        # hdbscan
        # distance = pairwise_distances(X_umap, metric='cosine')
        hdbscan = HDBSCAN(min_cluster_size=min_cluster_size) #, metric='precomputed'
        # labels = hdbscan.fit_predict(distance.astype('float64'))
        labels = hdbscan.fit_predict(X_umap)
        
        result, vectors, cluster_vectors = self.clustering_result_output(collection_name=collection_name, 
                                                                         labels=labels, 
                                                                         sampling_ratio=sampling_ratio, 
                                                                         data=data,
                                                                         method='HDBSCAN')
                  
        return result, vectors, cluster_vectors
    
if __name__ == '__main__':
    cluster = Clustering()
    cluster.Kmeans_clustering(collection_name='narrative_test_cpu')