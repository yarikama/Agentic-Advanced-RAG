import pandas as pd
import numpy as np
import gc
import umap.umap_ as umap
from sklearn.cluster import MiniBatchKMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics.pairwise import pairwise_distances
from hdbscan import HDBSCAN
from lshashpy3 import LSHash
import matplotlib.pyplot as plt

class Clustering:
    def __init__(self) -> None:
        
        print("Clustering initialized")
        
    def clustering_result_output(
        self,
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
            result_df: The final searching result for vectors ("content", "metadata").
            vectors : Only vectors (1-dim list).
            cluster_vectors : The vectors in each cluster list (2-dim list).
        """
        
        print(f"\n # Collection name : {collection_name}")
        print(f" # Clustering method : {method} ")
        
        # sample
        len_cluster = {}
        vectors = []
        cluster_vectors = []
        result_df = pd.DataFrame(columns=['dense_vector','content', 'metadata'])
        ouput_file = f'.sparse_graph_txt/{method}_{collection_name}_results.txt'
        
        with open(ouput_file, 'w') as file:
            for cluster in np.unique(labels):
                cluster_indices = np.where(labels == cluster)[0]
                sample_size = int(sampling_ratio * len(cluster_indices))
                sample_indices = np.random.choice(cluster_indices, sample_size, replace=False)
                cluster_vectors.append(data.loc[sample_indices,'dense_vector'])
                result_df = pd.concat([result_df,data.loc[sample_indices]], ignore_index=True)
                vectors.extend(data.loc[sample_indices,'dense_vector'])
                len_cluster[cluster] = len(sample_indices)
                
                for index in sample_indices:
                    content = data.loc[index, 'content']
                    metadata = data.loc[index, 'metadata']
                    file.write(f"{content}\n [{metadata}]\n")           
        
        print(f" # Number of sampling vectors : {len(vectors)} ")
        print(f" # Number of clustering : {len(np.unique(labels))} ")

        # Plot cluster sizes as a histogram
        cluster_sizes = list(len_cluster.values())
        plt.figure(figsize=(10, 6))
        plt.hist(cluster_sizes, bins=30, color='blue', edgecolor='black')
        plt.title('Cluster Size Distribution')
        plt.xlabel('Cluster Size')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
                        
        return result_df, vectors, cluster_vectors
    
    def clustering_spliting_output(self,
                                   collection_name: str,
                                   labels,
                                   num_splits,
                                   data,
                                   method
                                   ):
        """
        Result output: including sorting, sampling, searching, and writing to multiple text files.
        Args:
            collection_name (str): The name of the collection.
            labels: The labels (list) of the clusters.
            sampling_ratio: The ratio used for sampling.
            num_splits: The number of splits for the output files.
            data: All vectors and contents.
            method: The clustering method.
        Returns:
            result_df: The final searching result for vectors ("content", "metadata").
            vectors: Only vectors (1-dim list).
            cluster_vectors: The vectors in each cluster list (2-dim list).
        """
        
        print(f"\n # Collection name : {collection_name}")
        print(f" # Clustering method : {method} ")
        
        # sample
        len_cluster = {}
        vectors = []
        cluster_vectors = []
        result_df = pd.DataFrame(columns=['dense_vector','content', 'metadata'])
        output_files = [f'.sparse_graph_txt/{method}/{collection_name}_results_part{i}.txt' for i in range(1, num_splits+1)]
        file_handles = [open(output_file, 'w') for output_file in output_files]
        file_counts = [0] * num_splits
        
        try:
            for cluster in np.unique(labels):
                cluster_indices = np.where(labels == cluster)[0]
                np.random.shuffle(cluster_indices)
                split_indices = np.array_split(cluster_indices, num_splits)
                
                cluster_vectors.append(data.loc[cluster_indices,'dense_vector'])
                result_df = pd.concat([result_df,data.loc[cluster_indices]], ignore_index=True)
                vectors.extend(data.loc[cluster_indices,'dense_vector'])
                len_cluster[cluster] = len(cluster_indices)

                for part, indices in enumerate(split_indices):       
                    for index in indices:
                        content = data.loc[index, 'content']
                        metadata = data.loc[index, 'metadata']
                        file_handles[part].write(f"{content}\n [{metadata}]\n")
                        file_counts[part] += 1
        
        finally:
            for file_handle in file_handles:
                file_handle.close()
        
        print(f" # Number of sampling vectors : {len(vectors)} ")
        print(f" # Number of clustering : {len(np.unique(labels))} ")
        for i, count in enumerate(file_counts):
            print(f" # Number of entries in file part {i+1}: {count}")

        # Plot cluster sizes as a histogram
        cluster_sizes = list(len_cluster.values())
        plt.figure(figsize=(10, 6))
        plt.hist(cluster_sizes, bins=30, color='blue', edgecolor='black')
        plt.title('Cluster Size Distribution')
        plt.xlabel('Cluster Size')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
                        
        return result_df, vectors, cluster_vectors
    
    def load_collection(self, 
                        collection_name: str):
        
        file_path = f'.parquet/{collection_name}_all_entities.parquet'
        df = pd.read_parquet(file_path)
        df = df.sort_values(by='id').reset_index(drop=True)
        data = df[['id','dense_vector','content', 'metadata']]
        print(" Load collection vectors done...")
        del df
        gc.collect()
        
        return data
    
    def umap_data_output(self,
                            collection_name: str,
                            umap_componemts=10):
        """
        Perform UMAP dimensionality reduction on a collection.
        Args:
            data: The vectors in the collection.
            umap_componemts (int): The number of components for UMAP dimensionality reduction.
        Returns:
            X_umap: The reduced vectors.
        """
        data = self.load_collection(collection_name)
        reducer = umap.UMAP(metric='cosine', n_components=umap_componemts)
        X_umap = reducer.fit_transform(data['dense_vector'].values.tolist())
        df = pd.DataFrame({'id': data['id'], 'umap_vector': list(X_umap)})
        output_file = f'.parquet/umap/{collection_name}_umap{umap_componemts}_vectors.parquet'
        df.to_parquet(output_file, index=False)
        print(" UMAP dimensionality reduction done and saved to parquet...")
        gc.collect()
        
        return X_umap
    
    def umap_data_read(self, 
                       collection_name: str, 
                       umap_componemts=10):
        
        file_path = f'.parquet/umap/{collection_name}_umap{umap_componemts}_vectors.parquet'
        df = pd.read_parquet(file_path)
        df = df.sort_values(by='id').reset_index(drop=True)
        data = df[['id','umap_vector']]
        X_umap = np.array([v for v in data['umap_vector']])
        print(" Load UMAP vectors done...")
        del df,data
        gc.collect()
        
        return X_umap
        
    def kmeans_clustering(self,
                          umap_componemts=10, 
                          k_components=2000, 
                          kmeans_batch_size=10000,
                          sampling_ratio=0.1,
                          num_splits=5,
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
        
        data = self.load_collection(collection_name)    
        
        # umap
        X_umap = self.umap_data_read(collection_name, umap_componemts)

        #do kmeans
        num_batches = int(np.ceil(len(X_umap) / kmeans_batch_size))
        kmeans = MiniBatchKMeans(n_clusters=k_components, random_state=0, batch_size=kmeans_batch_size)
        labels = kmeans.fit_predict(X_umap)
        
        for i in range(num_batches):
            start_idx = i * kmeans_batch_size
            end_idx = min((i + 1) * kmeans_batch_size, len(X_umap))
            batch_vectors = X_umap[start_idx:end_idx]

            # 增量訓練 MiniBatchKMeans 模型
            kmeans.partial_fit(batch_vectors)
            gc.collect()
        labels = kmeans.predict(X_umap)
                
        # result, vectors, cluster_vectors = self.clustering_result_output(collection_name=collection_name, 
        #                                                                  labels=labels, 
        #                                                                  sampling_ratio=sampling_ratio, 
        #                                                                  data=data,
        #                                                                  method='Kmeans')
        result, vectors, cluster_vectors = self.clustering_spliting_output(collection_name=collection_name, 
                                                                           labels=labels, 
                                                                           num_splits=num_splits, 
                                                                           data=data,
                                                                           method='Kmeans')
                  
        return result, vectors, cluster_vectors

    def lsh_clustering(self,
                       num_layers=1, 
                       hash_size=11 , 
                       table_num=1 , 
                       sampling_ratio=0.1,
                       num_splits=5,
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
        
        data = self.load_collection(collection_name)
        
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
        original_indices = list(range(len(data)))
        final_buckets, final_indices = lsh_layer(data['dense_vector'].values.tolist(), 1, original_indices)
        final_buckets = np.array(final_buckets)
        
        sorted_data = data.iloc[final_indices].reset_index(drop=True)

        # result, vectors, cluster_vectors = self.clustering_result_output(collection_name=collection_name, 
        #                                                                  labels=labels, 
        #                                                                  sampling_ratio=sampling_ratio, 
        #                                                                  data=sorted_data,
        #                                                                  method='LSH')
        result, vectors, cluster_vectors = self.clustering_spliting_output(collection_name=collection_name, 
                                                                           labels=labels, 
                                                                           num_splits=num_splits, 
                                                                           data=data,
                                                                           method='LSH')
                  
        return result, vectors, cluster_vectors
    
    def gmm_clustering(self,
                       umap_componemts=10, 
                       gmm_components=2000, 
                       sampling_ratio=0.1,
                       num_splits=5,
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
        
        data = self.load_collection(collection_name)
        
        # umap
        X_umap = self.umap_data_read(collection_name, umap_componemts)
        
        # GMM
        gmm = GaussianMixture(n_components=gmm_components)
        labels = gmm.fit_predict(X_umap)
        
        # result, vectors, cluster_vectors = self.clustering_result_output(collection_name=collection_name, 
        #                                                                  labels=labels, 
        #                                                                  sampling_ratio=sampling_ratio, 
        #                                                                  data=data,
        #                                                                  method='GMM')
        result, vectors, cluster_vectors = self.clustering_spliting_output(collection_name=collection_name, 
                                                                           labels=labels, 
                                                                           num_splits=num_splits, 
                                                                           data=data,
                                                                           method='GMM')
                  
        return result, vectors, cluster_vectors
    
    def dbscan_clustering(self,
                          umap_componemts=10,
                          eps=0.0001,
                          min_samples=40,
                          sampling_ratio=0.1,
                          num_splits=5,
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
        
        data = self.load_collection(collection_name)
        
        # umap
        X_umap = self.umap_data_read(collection_name, umap_componemts)
        
        # dbscan
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = dbscan.fit_predict(X_umap)
        
        # result, vectors, cluster_vectors = self.clustering_result_output(collection_name=collection_name, 
        #                                                                  labels=labels, 
        #                                                                  sampling_ratio=sampling_ratio, 
        #                                                                  data=data,
        #                                                                  method='DBSCAN')
        result, vectors, cluster_vectors = self.clustering_spliting_output(collection_name=collection_name, 
                                                                           labels=labels, 
                                                                           num_splits=num_splits, 
                                                                           data=data,
                                                                           method='DBSCAN')
                  
        return result, vectors, cluster_vectors
    
    def hdbscan_clustering(self,
                           umap_componemts=10,
                           min_cluster_size=50,
                           sampling_ratio=0.1,
                           num_splits=5,
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
        
        data = self.load_collection(collection_name)
        
        # umap
        X_umap = self.umap_data_read(collection_name, umap_componemts)
        
        # hdbscan
        # distance = pairwise_distances(X_umap, metric='cosine')
        hdbscan = HDBSCAN(min_cluster_size=min_cluster_size) #, metric='precomputed'
        # labels = hdbscan.fit_predict(distance.astype('float64'))
        labels = hdbscan.fit_predict(X_umap)
        
        # result, vectors, cluster_vectors = self.clustering_result_output(collection_name=collection_name, 
        #                                                                  labels=labels, 
        #                                                                  sampling_ratio=sampling_ratio, 
        #                                                                  data=data,
        #                                                                  method='HDBSCAN')
        result, vectors, cluster_vectors = self.clustering_spliting_output(collection_name=collection_name, 
                                                                           labels=labels, 
                                                                           num_splits=num_splits, 
                                                                           data=data,
                                                                           method='HDBSCAN')
                  
        return result, vectors, cluster_vectors
    
    def agglomerative_clustering(self,
                                 umap_componemts=10,
                                 n_clusters=200,
                                 sampling_ratio=0.1,
                                 num_splits=5,
                                 collection_name: str=''):
        """
        Perform Agglomerative clustering on a collection.
        The results of content and metadata are saved in a text file.
        Args:
            umap_componemts (int): The number of components for UMAP dimensionality reduction.
            n_clusters (int): The number of clusters to form.
            sampling_ratio (float): The ratio of samples to be used for clustering.
            collection_name (str): The name of the collection.
        Returns:
            result: The final searching result for vectors ("content", "metadata").
            vectors : Only vectors (1-dim list).
            cluster_vectors : The vectors in each cluster list (2-dim list).
        """
        
        data = self.load_collection(collection_name)
        
        # umap
        X_umap = self.umap_data_read(collection_name, umap_componemts)
        
        # Agglomerative
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agg.fit_predict(X_umap)
        
        # result, vectors, cluster_vectors = self.clustering_result_output(collection_name=collection_name, 
        #                                                                  labels=labels, 
        #                                                                  sampling_ratio=sampling_ratio, 
        #                                                                  data=data,
        #                                                                  method='Agglomerative')
        result, vectors, cluster_vectors = self.clustering_spliting_output(collection_name=collection_name, 
                                                                           labels=labels, 
                                                                           num_splits=num_splits, 
                                                                           data=data,
                                                                           method='Agglomerative')
                  
        return result, vectors, cluster_vectors
    
if __name__ == '__main__':
    
    cluster = Clustering()
    cluster.Kmeans_clustering(collection_name='narrative_test_cpu')