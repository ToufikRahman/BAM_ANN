import numpy as np
import os
import ast
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist

def get_cluster_info(centroid_path):
    # Initialize the dictionary to store cluster information
    all_cluster_info_dict = {}

    # Iterate through each text file in the folder
    for filename in os.listdir(centroid_path):
        if filename.endswith('all_clusters_info.txt'):
            filepath = os.path.join(centroid_path, filename)

            # Read the text file
            with open(filepath, 'r') as f:
                lines = f.readlines()

            # Process the lines to extract cluster information
            # current_cluster_info = {}

            current_cluster_filename = None
            current_centroid = None
            current_cluster_size = None

            for line in lines:
                words = line.strip().split()

                if line == '\n':
                    continue

                if words[0] == 'Cluster_Filename:':
                    current_cluster_filename = words[1]
                elif words[0] == 'Centroid:':
                    current_centroid_str = ' '.join(words[1:])
                    current_centroid = ast.literal_eval(current_centroid_str)
                    
                elif words[0] == 'Cluster_Size:':
                    current_cluster_size = int(words[1])
                    

                # Add the cluster information to the overall dictionary
                key = current_cluster_filename
                all_cluster_info_dict[key] = [current_centroid, current_cluster_size]

    return all_cluster_info_dict


def merge(centroid_path, clusters_path, out_path, max_size_per_cluster):
    all_cluster_info_dict = get_cluster_info(centroid_path)

    file_names = []
    centroids = []
    sizes = []

    for key, value in all_cluster_info_dict.items():
        file_names.append(key)
        centroids.append(value[0])
        sizes.append(value[1])

    centroids_array = np.array(centroids)
    distances = cdist(centroids_array, centroids_array)
    closest_indices = np.argsort(distances, axis=1)[:, 1:(centroids_array.shape[0])]

    # print(closest_indices)

    merged_clusters = []
    merged_sizes = []

    visited = set()

    for i in range(len(file_names)):
        if i in visited:
            continue
       
        merged_cluster = [i]
        merged_size = sizes[i]

        for j in closest_indices[i]:
            if j not in visited and merged_size + sizes[j] <= max_size_per_cluster:
                merged_cluster.append(j)
                merged_size += sizes[j]
                visited.add(j)
        
        for idx in merged_cluster:
            visited.add(idx)
       
        merged_clusters.append(merged_cluster)
        merged_sizes.append(merged_size)

    # tsne = TSNE(n_components=2,random_state=42)
    # centroids_tsne = tsne.fit_transform(centroids_array)
    # for cluster_idx in merged_clusters:
    #     x = centroids_tsne[cluster_idx,0]
    #     y = centroids_tsne[cluster_idx,1]
    #     plt.scatter(x,y,label=f'cluster_indices[0]',alpha=0.7)
    # plt.title('t-SNE for merged clusters')
    # plt.xlabel('t-SNE 1')
    # plt.ylabel('t-SNE 2')
    # plt.legend()
    # plt.show()

    # print(f'Merged clusters: {merged_clusters} \n Merged sizes: {merged_sizes}')

    # print(file_names[0],'  ',file_names[24], '  ' ,file_names[7])
    c = 0
    all_clusters_info = []
    for cluster_indices in merged_clusters:

        # print(cluster_indices)
        
        file_list = []
        for file in cluster_indices:
            file_list.append(file_names[file])
        # print(file_list)
        merged_data = np.concatenate([np.load(os.path.join(clusters_path,f)) for f in file_list])
        # print(merged_data.shape)
        name = f'merged_{c}.npy'
        # Calculate the centroid of the cluster
        cluster_centroid = np.mean(merged_data[:, :-1], axis=0)  # Exclude the last column (ID)
        all_clusters_info.append({
            'cluster_filename': name,
            'centroid': cluster_centroid.tolist(),
        })
        output_text_file = f'{centroid_path}/merged_clusters_info.txt'
        with open(output_text_file, 'w') as f:
            for info in all_clusters_info:
                f.write(f"Cluster_Filename: {info['cluster_filename']}\n")
                f.write(f"Centroid: {info['centroid']}\n")
        
        np.save(os.path.join(out_path,name),merged_data)
        print(f'Saving {name} : Done')
        c += 1
        

    # return merged_clusters, merged_sizes



        
