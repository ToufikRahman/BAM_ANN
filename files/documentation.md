### make_chunks.py
`make_chunks.py` is designed to split large datasets into smaller, manageable chunks. This is particularly useful for processes that require handling large datasets but are constrained by memory limitations. The script supports different data formats and ensures that each chunk is stored with its corresponding indices, facilitating easy tracking and referencing within the larger dataset.

#### `create_chunk(data_name,data_path,out_path,chunk_size,total_data_size)`
- **Description**: This function is responsible for reading data from a specified file, dividing it into specified-sized chunks, and saving these chunks to a new location. It supports processing of data in both DEEP and SIFT formats.
- **Parameters**:
  - `data_name` (str): Identifier for the type of data being processed. It specifies the format or source of the data and currently supports 'deep' or 'sift'.
  - `data_path` (str): File path to the input data. This is where the data to be chunked is read from.
  - `out_path` (str): Directory path where the chunked files will be stored. Each chunk will be saved as a separate file in this directory.
  - `chunk_size` (int): The number of entries each chunk should contain.
  - `total_data_size` (int): The total number of entries to be processed from the dataset.

### minibatchKmeans.py
The `minibatchKmeans.py` script is designed for clustering large datasets using the MiniBatch KMeans algorithm from the scikit-learn library. It handles data in chunks to efficiently process potentially large volumes of data that may not fit into memory all at once.

#### `load_chunk(chunk_name,chunk_folder_path)`
- **Description**: Loads a data chunk from storage for processing.
- **Parameters**:
  - `chunk_name` (str): The filename of the chunk to load.
  - `chunk_folder_path` (str): The directory containing the chunk files.
- **Returns**: Numpy array of data loaded from the specified chunk file.

#### `cluster(cluster_folder_path,chunk_folder_path,centroid_path,model_path,num_clusters,batch_size)`
- **Description**: Manages the entire clustering process, from model initialization to processing data chunks and saving the trained model.
- **Parameters**:
  - cluster_folder_path (str): Path where clustered data should be saved. 
  - chunk_folder_path (str): Path containing the data chunks.
  - centroid_path (str): Path where centroids should be saved.
  - model_path (str): Path to save or load the MiniBatch KMeans model.
  - num_clusters (int): Number of clusters to form.
  - batch_size (int): Number of samples per batch to process.
#### `check_and_load_model(model_path)`
- **Description**: Checks for an existing MiniBatch KMeans model and loads it; if not found, returns None.
- **Parameters**:
  - model_path (str): Path where the MiniBatch KMeans model is stored.
- **Returns**: Loaded MiniBatch KMeans model or None.
  
#### `append_to_cluster_file(cluster_path,data)`
- **Description**: Appends data to a numpy file, creating a new file if it does not exist.
- **Parameters**:
  - cluster_path (str): Path to the cluster file.
  - data (numpy.ndarray): Data to append to the file.
  
### merge_clusters.py

#### `merge(centroid_path,clusters_path,out_path,max_size_per_cluster)`
- **Description**: The `merge` function is designed to consolidate clusters based on their proximity and a specified maximum cluster size constraint. This function retrieves cluster information, calculates the pairwise distances between cluster centroids, and then merges clusters that are closest to each other without exceeding the specified maximum size. The merged cluster data and centroids are then saved to specified directories.
- **Parameters**:
  - `centroid_path` (str): The directory path that contains the cluster information files.
  - `clusters_path` (str): The directory path where the individual cluster files are stored.
  - `out_path` (str): The directory path where the merged cluster files will be saved.
  - `max_size_per_cluster` (int): The maximum number of elements allowed in a merged cluster.
- **Returns**:
  - This function does not return any value but writes merged cluster data and information to files.
 
#### `get_cluster_info(centroid_path)`
- **Description**: The `get_cluster_info` function is designed to parse and extract clustering information from text files within a specified directory. This function reads files that end with 'all_clusters_info.txt', extracting details such as the cluster's filename, centroid, and size, and then compiles this information into a dictionary. This is useful for systems that need to manage or utilize cluster data, such as in machine learning applications for clustering analysis or data segmentation.
- **Parameters**:
  - `centroid_path` (str): The path to the directory containing the cluster information text files.
- **Returns**:
  - `all_cluster_info_dict` (dict): A dictionary where each key is a cluster's filename and the value is a list containing the centroid and size of the cluster.

### create_index.py

#### `build_index(data_path,index_num,out_path,index_name,param_list)`
- **Description**: The `build_index` function is designed to create efficient indices for large-scale data retrieval systems. It supports multiple indexing methods including HNSW (Hierarchical Navigable Small World) and is structured to handle large datasets by splitting the data during the index building process. The function integrates with external processing through the `process_data` module for data preparation.
- **Parameters**:
  - `data_path` (str): The path to the directory containing the data files.
  - `index_num` (int): An identifier for the type of index to create; `0` for HNSW, `1` for FAISS, and potentially others.
  - `out_path` (str): The output directory where the index files will be saved.
  - `index_name` (str): The name assigned to the index.
  - `param_list` (list): A list of parameters specific to the indexing method. For HNSW, this includes `m` (max number of edges per node) and `ef` (size of the dynamic candidate list). 
- **Returns**: The function does not return any value but outputs the index files directly to the specified output directory.

#### `loading_index(data_file_name,data_path,index_num,out_path,index_name,dim)`
- **Description**: The `loading_index` function is designed to load an existing index from the filesystem into memory, allowing for rapid querying and manipulation. It supports multiple index types including HNSW, FAISS etc. handling them based on the `index_num` parameter. This function is crucial for applications that need to quickly restore state and perform operations without rebuilding the index.
- **Parameters**:
  - `data_file_name` (str): Name of the data file containing the original data used to create the index.
  - `data_path` (str): The directory path where the data file is stored.
  - `index_num` (int): Numeric identifier for the type of index to load; `0` for HNSW, `1` for FAISS, and `2` for SG.
  - `out_path` (str): The output path where the index files are saved.
  - `index_name` (str): The name of the index file to be loaded.
  - `dim` (int): The dimensionality of the data points indexed.
- **Returns**:
  - `index` (object): The loaded index object, ready for use in querying or further processing.
  - `true_idx` (array): The array of true indices associated with the vectors, if applicable and available.

### query_processing.py

#### `group_query(all_queries,centroid_path)`

- **Description**:The `group_query` function is designed to group queries based on their proximity to predefined centroids. It calculates the Euclidean distance between each query and the centroids, assigning each query to the nearest centroid unless already assigned. This function helps in categorizing queries for systems that might need to process or respond to these queries based on their geographical or categorical proximity.
- **Parameters**:
  - `all_queries` (array): An array or list of query vectors that need to be grouped.
  - `centroid_path` (str): The path to the directory containing centroid information.
- **Returns**:
  - `centroid_query_dict` (dict): A dictionary where the keys are centroid filenames and the values are lists of indices of queries that are closest to these centroids.

#### `get_closest_centroids(query,centroid_path,num_centroid)`

- **Description**:The `get_closest_centroids` function calculates and returns the file names of the nearest centroids to a given query. This function is essential for systems requiring quick identification of the closest data points (centroids) in applications like nearest neighbor searches, clustering, or classification tasks.
- **Parameters**:
  - `query` (array): A single query vector for which the closest centroids are to be found.
  - `centroid_path` (str): The directory path that contains the centroid data.
  - `num_centroid` (int): The number of closest centroids to return.
- **Returns**:
  - `closest_files` (list): A list of filenames that correspond to the closest centroids to the query, sorted by proximity.

#### `assign_queries_to_centroids(query_array,centroid_path)`

- **Description**: The `assign_queries_to_centroids` function categorizes each query in a given array by assigning it to the closest centroid. This is commonly used in clustering applications where queries need to be associated with predefined groups (centroids) based on proximity. The function is crucial for partitioning data into relevant clusters for further analysis or processing.
- **Parameters**:
  - `query_array` (array): An array of query vectors that need to be assigned to centroids.
  - `centroid_path` (str): The path to the directory containing centroid data.
- **Returns**:
  - `query_assignment` (dict): A dictionary where keys are centroid names and values are lists of indices from the `query_array` that are closest to the corresponding centroid.

### search_index.py

#### `search(index, index_num, query, k_neighbors)`
- **Description**:The `search` function is designed to conduct k-nearest neighbor (k-NN) searches on a given index. It supports multiple index types, currently implementing functionality for an HNSW index and has a placeholder for potential FAISS index support. This function is used in scenarios where quick and efficient nearest neighbor searches are needed, such as in recommendation systems, anomaly detection, or clustering.
- **Parameters**:
  - `index` (object): The index object on which the k-NN search is performed. This object must support the `knn_query` method or similar, depending on the index type.
  - `index_num` (int): An identifier indicating the type of index used; `0` for HNSW, `1` for FAISS (currently not supported).
  - `query` (array): The query vector or array of query vectors for which neighbors are sought.
  - `k_neighbors` (int, optional): The number of nearest neighbors to retrieve. Defaults to 100.
- **Returns**:
  - `labels` (array): An array of labels indicating the nearest neighbors found for the given query.
  - `distances` (array): An array of distances corresponding to the nearest neighbors.

