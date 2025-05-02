import numpy as np
import os
import ast
from scipy.spatial.distance import cdist


def group_query_multi(all_queries, centroid_path, num_centroids_per_query=3):
    centroid_query_dict = {}
    centroids_dict = get_centroids(centroid_path)
    
    # Initialize a set to keep track of assigned queries
    assigned_queries = set()
    total_search = 0

    for query_idx, query in enumerate(all_queries):
        # Calculate distance of current query to all centroids
        distances = {centroid_file: np.linalg.norm(query - centroid) for centroid_file, centroid in centroids_dict.items()}
        
        # Sort centroids by distance
        sorted_centroids = sorted(distances.items(), key=lambda x: x[1])  # [(centroid_file, distance), ...]

        # Select the closest num_centroids_per_query centroids
        closest_centroids = sorted_centroids[:num_centroids_per_query]

        # Get the distance to the closest centroid
        closest_distance = closest_centroids[0][1]

        selected_centroids = [
            centroid_file for centroid_file, dist in closest_centroids
            if dist <= 1.15 * closest_distance
        ]
        # total_search += len(selected_centroids)
        # print(f"{len(selected_centroids)} centroids.")
        # Add the query index to the closest centroids
        for centroid_file in selected_centroids:
            centroid_query_dict.setdefault(centroid_file, []).append(query_idx)
        
        assigned_queries.add(query_idx)
    # print(f'Number of total access: {total_search}')
    # Handle unassigned queries
    unassigned_queries = set(range(len(all_queries))) - assigned_queries
    for query_idx in unassigned_queries:
        # Assign unassigned queries to the nearest centroid
        closest_centroid_file = min(centroids_dict.keys(), key=lambda x: np.linalg.norm(all_queries[query_idx] - centroids_dict[x]))
        centroid_query_dict.setdefault(closest_centroid_file, []).append(query_idx)

    return centroid_query_dict

def group_query(all_queries, centroid_path):
    centroid_query_dict = {}
    centroids_dict = get_centroids(centroid_path)
    # print("Centroid Dict: ", centroids_dict)
    # Initialize a set to keep track of assigned queries
    assigned_queries = set()

    for query_idx, query in enumerate(all_queries):
        min_distance = float('inf')
        closest_centroid_file = None
        
        # Calculate distance of current query to all centroids
        for centroid_file, centroid in centroids_dict.items():
            distance = np.linalg.norm(query - centroid)
            if distance < min_distance and centroid_file not in centroid_query_dict.values():
                min_distance = distance
                closest_centroid_file = centroid_file
        
        # Add the query index to the closest centroid
        if closest_centroid_file is not None:
            centroid_query_dict.setdefault(closest_centroid_file, []).append(query_idx)
            assigned_queries.add(query_idx)
    
    # Handle unassigned queries
    unassigned_queries = set(range(len(all_queries))) - assigned_queries
    for query_idx in unassigned_queries:
        # Assign unassigned queries to the nearest centroid
        closest_centroid_file = min(centroids_dict.keys(), key=lambda x: np.linalg.norm(all_queries[query_idx] - centroids_dict[x]))
        centroid_query_dict.setdefault(closest_centroid_file, []).append(query_idx)
    
    return centroid_query_dict

def get_closest_centroids(query,centroid_path,num_centroid):
    centroid_dict = get_centroids(centroid_path)
    distances = {file_name: euclidean_distance(centroid , query) for file_name, centroid in centroid_dict.items()}

    # Sort the file names by distance
    closest_files = sorted(distances, key=distances.get)[:num_centroid]
    return closest_files

def assign_queries_to_centroids(query_array, centroid_path):
    centroid_dict = get_centroids(centroid_path)
    # Initialize dictionary to store query assignments
    query_assignment = {centroid: [] for centroid in centroid_dict.keys()}

    # Iterate through each query
    for i, query in enumerate(query_array):
        min_distance = float('inf')
        assigned_centroid = None

        # Iterate through each centroid and compute the distance
        for centroid_name, centroid in centroid_dict.items():
            distance = euclidean_distance(query, centroid)
            if distance < min_distance:
                min_distance = distance
                assigned_centroid = centroid_name

        # Assign the query index to the centroid with the minimum distance
        query_assignment[assigned_centroid].append(i)

    return query_assignment

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def get_centroids(centroid_path):
    # Initialize the dictionary to store cluster information
    all_cluster_info_dict = {}

    # Iterate through each text file in the folder
    for filename in os.listdir(centroid_path):
        if filename.endswith('centroid_info.txt'):
            filepath = os.path.join(centroid_path, filename)

            # Read the text file
            with open(filepath, 'r') as f:
                lines = f.readlines()

            # Process the lines to extract cluster information
            # current_cluster_info = {}

            current_cluster_filename = None
            current_centroid = None


            for line in lines:
                words = line.strip().split()

                if line == '\n':
                    continue

               
                current_cluster_filename = words[0]
                # print('Filename:',words[0])
                
                current_centroid_str = ' '.join(words[1:])
                current_centroid = ast.literal_eval(current_centroid_str)
                # print('Centroid: ',current_centroid)
                    

                # Add the cluster information to the overall dictionary
                key = current_cluster_filename
                all_cluster_info_dict[key] = [current_centroid]

    return all_cluster_info_dict

def check_common(dict):
    for k1,l1 in dict.items():
        for k2,l2 in dict.items():
            if k1 != k2:
                set1 = set(l1)
                set2 = set(l2)
                if set1.intersection(set2):
                    return True
    return False

def compare_dictionaries(dict1, dict2):
    """Check if two dictionaries have the same set of key-value pairs."""
    return set(dict1.items()) == set(dict2.items())
