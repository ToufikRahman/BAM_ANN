import make_chunks
import read_DEEP
import os
import create_index
import query_processing
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
import evaluation
import search_index
import minibatchKmeans
import heapq 
import save_results
from concurrent.futures import ProcessPoolExecutor, as_completed

class BAM:
    def __init__(self, data_path, m=48, ef=400, chunk_size=1000000, total_data_size=10000000, num_clusters=10):
        self._m = m
        self._ef = ef
        name = data_path.split('/')[-1][:-5]
        self._name = f"{name}_BAM_B"
        self._chunk_path = f"{self._name}/chunks"
        self._cluster_path = f"{self._name}/clusters"
        self._centroid_path = f"{self._name}/centroids"
        # self._index_path = f"{self._name}/indexes_M{m}_ef{ef}"
        self._model_path = f"{self._name}/model"
        self.chunk_size = chunk_size
        self.total_data_size = total_data_size
        self.num_clusters = num_clusters

    def build_index(self, data_path, index_path, index_num):
        if not os.path.exists(self._name):
            os.makedirs(self._name)
        os.makedirs(self._chunk_path, exist_ok=True)
        os.makedirs(self._cluster_path, exist_ok=True)
        os.makedirs(self._centroid_path, exist_ok=True)
        os.makedirs(index_path, exist_ok=True)

        param_list = [self._m, self._ef]
        index_name = f'hnsw_M{self._m}_E{self._ef}.bin'

        if len(os.listdir(self._chunk_path)) == 0:
            t = time.time()
            make_chunks.create_chunk('deep', data_path, self._chunk_path, chunk_size=self.chunk_size, total_data_size=self.total_data_size)
            print("Chunking time:", time.time() - t)

        if len(os.listdir(self._cluster_path)) == 0:
            t = time.time()
            minibatchKmeans.cluster(self._cluster_path, self._chunk_path, self._centroid_path, self._model_path, num_clusters=self.num_clusters, batch_size=1000000)
            print("Total Clustering time:", time.time() - t)

        if 'BAM_IDX.bin' not in os.listdir(index_path):
            t = time.time()
            create_index.build_index(self._cluster_path, index_num, index_path, index_name, param_list)
            print("Total Indexing time:", time.time() - t)

    def search(self, bam_idx, all_query, dim, gt_path, index_num, k=100, num_centroids_per_query=5):
        index_name = f'hnsw_M{self._m}_E{self._ef}.bin'
        query_dict = query_processing.group_query_multi(all_query, self._centroid_path, num_centroids_per_query)

        # Initialize predicted data
        predicted_data = [[(-1, np.inf)] * k for _ in range(len(all_query))]

        # Sequential processing (no parallelization)
        for key, value in query_dict.items():
            local_predicted_data = process_key_value(
                key, value, bam_idx, index_name, index_num, k, all_query
            )
            # Merge results
            for v, new_data in local_predicted_data:
                predicted_data[v] = heapq.nsmallest(k, predicted_data[v] + new_data, key=lambda x: x[1])

        # Convert results to arrays
        predicted_array = np.array([[item[0] for item in query_data] for query_data in predicted_data])
        predicted_distances = np.array([[item[1] for item in query_data] for query_data in predicted_data])

        return predicted_array, predicted_distances


    def print_results(self, predicted_array, ground_truth, total_time, k):
        recall = round(evaluation.recall(predicted_array, ground_truth), 3)
        qps = round(len(ground_truth) / total_time, 3)

        print("Recall:", recall)
        print(f"Total time for {len(ground_truth)} queries: {total_time}")
        print(f"Average time per query: {total_time / len(ground_truth)}")
        print(f"QPS: {qps}")

        parameters = {
            "Name:": self._name,
            "M:": self._m,
            "ef:": self._ef,
            "gt_length:": len(ground_truth),
            "K:": k,
            "Clusters:": "dyn",
            "recall:": recall,
            "QPS:": qps,
        }
        # save_results.save_results(parameters)


def process_key_value(key, value, bam_idx, index_name, index_num, k, all_query):
    # print(f"Loading index for: {key}")
    name = key.split('.')[0]
    idx_key = f'{name}_{index_name}'    
    print(f"Loading index for: {idx_key}")
    index = bam_idx[idx_key]

    local_predicted_data = []
    queries = [all_query[v] for v in value]
    batch_labels, batch_distances = search_index.search(index, index_num, queries, k_neighbors=k)

    for v, labels, distances in zip(value, batch_labels, batch_distances):
        new_data = list(zip(labels, distances))
        local_predicted_data.append((v, new_data))

    return local_predicted_data

