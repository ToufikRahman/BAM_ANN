import numpy as np
from sklearn.cluster import MiniBatchKMeans
import os
import gc
import joblib  # For saving and loading the model

def load_chunk(chunk_name, chunk_folder_path):
    file_path = os.path.join(chunk_folder_path, chunk_name)
    return np.load(file_path)

def append_to_cluster_file(cluster_path, data):
    if os.path.exists(cluster_path):
        existing_data = np.load(cluster_path, mmap_mode='r+')
        combined_data = np.vstack((existing_data, data))
        np.save(cluster_path, combined_data)
    else:
        np.save(cluster_path, data)

def check_and_load_model(model_path):
    if os.path.exists(model_path):
        print("Model found. Loading...")
        return joblib.load(model_path)
    else:
        print("No existing model. Creating a new one.")
        return None

def cluster(cluster_folder_path, chunk_folder_path, centroid_path, model_path, num_clusters=10, batch_size=1000000):
    # Check if model exists
    chunk_names = os.listdir(chunk_folder_path)
    kmeans = check_and_load_model(model_path)
    if not kmeans:
        kmeans = MiniBatchKMeans(max_iter = 300, n_clusters=num_clusters, random_state=42, batch_size=batch_size)
        print('Clustering started...')
        # Load data and fit model incrementally
        for chunk_file in chunk_names:
            try:
                data = load_chunk(chunk_file, chunk_folder_path)
                print(f'Partial fitting {chunk_file}')
                kmeans.partial_fit(data[:, :-1])  
                del data
                gc.collect()
            except Exception as e:
                print(f'Failed processing {chunk_file}: {str(e)}')
                continue
        # Save the trained model
        joblib.dump(kmeans, model_path)
        print(f"Model saved at {model_path}")
    kmeans = check_and_load_model(model_path)
    # Process each chunk and save clustered data in bulk
    for chunk_file in chunk_names:
        print(f'Clustering {chunk_file}')
        data = load_chunk(chunk_file, chunk_folder_path)
        indices = kmeans.predict(data[:, :-1])
        
        # Organize data by cluster
        clustered_data = {i: [] for i in range(num_clusters)}
        for idx, point in zip(indices, data):
            clustered_data[idx].append(point)
        
        # Save data for each cluster
        for idx, points in clustered_data.items():
            if points:  # Only save if there are points in this cluster
                cluster_filename = f'cluster{idx}.npy'
                cluster_full_path = os.path.join(cluster_folder_path, cluster_filename)
                append_to_cluster_file(cluster_full_path, np.array(points))
        
        del data
        gc.collect()

    # Save centroids

    centroid_filename = "centroid_info.txt"
    centroid_file_path = os.path.join(centroid_path, centroid_filename)
    centroid_info = {}
    for i, centroid in enumerate(kmeans.cluster_centers_):
        cluster_filename = f'cluster{i}.npy'
        # cluster_full_path = os.path.join(cluster_folder_path, cluster_filename)
        # np.save(cluster_full_path, np.array(cluster))
        # print(f'Save cluster: {cluster_filename}')
        centroid_info[cluster_filename] = kmeans.cluster_centers_[i].tolist()

    # Write centroid info to a text file
    with open(centroid_file_path, 'w') as f:
        for filename, centroid in centroid_info.items():
            f.write(f'{filename} {centroid}\n')