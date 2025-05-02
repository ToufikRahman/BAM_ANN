import read_DEEP
import numpy as np
from sklearn.cluster import KMeans
import os
import time


def create_clusters(data_folder, cluster_output, centroid_output, clusters_per_chunk):


    # List all the data files in the data folder
    data_files = [f for f in os.listdir(data_folder) if f.endswith('.fbin')]

    counter = 0
    all_clusters_info = []  # List to store cluster information

    t = time.time()

    for data_file in data_files:

        file_index = int(data_file.split('.')[0].replace('chunk', ''))

        # Load the data
        data, _ = read_DEEP.read_fbin(os.path.join(data_folder, data_file))

        # print('Shape of chunk ', file_index, ': ', data.shape)

        true_start_idx = file_index * data.shape[0]

        # Create the column vector starting with true ids
        column_vector = np.arange(true_start_idx, data.shape[0] + true_start_idx).reshape((-1, 1))

        # Append the true ID column vector to the data array
        data_with_id = np.hstack((data, column_vector))

        # Perform K-means clustering
        kmeans = KMeans(clusters_per_chunk)  
        clusters = kmeans.fit_predict(data)

        # print('Clustering chunk ', file_index, ' : Done')

        for cluster_id in range(kmeans.n_clusters):
            cluster_filename = f"{data_file}_cluster{cluster_id}.npy"
            cluster_data = data_with_id[clusters == cluster_id]
            np.save(os.path.join(cluster_output, cluster_filename), cluster_data)

            # Calculate the centroid of the cluster
            cluster_centroid = np.mean(cluster_data[:, :-1], axis=0)  # Exclude the last column (ID)

            # Store cluster information
            all_clusters_info.append({
                'cluster_filename': cluster_filename,
                'centroid': cluster_centroid.tolist(),
                'cluster_size': len(cluster_data)
            })

        print('Save clusters for chunk ', file_index, ' : Done')
        counter += 1

    # Save all cluster information to a text file
    output_text_file = f'{centroid_output}/all_clusters_info.txt'
    with open(output_text_file, 'w') as f:
        for info in all_clusters_info:
            f.write(f"Cluster_Filename: {info['cluster_filename']}\n")
            f.write(f"Centroid: {info['centroid']}\n")
            f.write(f"Cluster_Size: {info['cluster_size']}\n")

    print('Total time for clustering: ', time.time() - t)
