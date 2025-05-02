# BAM: Efficient Billion-Scale Approximate Nearest Neighbor Search under Memory Constraints.

This repository contains a collection of Python scripts designed to implement a scalable data indexing and querying system. Break And Make (BAM) algorithm tackles the challenges posed by limited memory resources. BAM algorithm divides the data into smaller manageable chunks and then partitions each chunk using clustering. The smaller clusters are merged to form a few balanced clusters, and a graph index is created for each cluster. A query-aware search algorithm targets only relevant clusters for increased memory efficiency and faster retrieval. BAM is capable of billion-scale similarity search.

## BAM pipeline

![BAM_pileline](https://github.com/ToufikRahman/BAM-ANN/blob/main/files/BAM_Pipeline_final.png)

## Installation

- Python 3.x
- NumPy
- scikit-learn
- hnswlib
- joblib

## Key features
- **Fast Search Capabilities:** Achieve millisecond-level search times across datasets containing billion vectors.
- **Streamlined Management of Unstructured Data:** Simplify the handling of unstructured data, making it more accessible and manageable.
- **Dependable, Always-Available Vector Database:** Maintain a vector database that is consistently reliable and accessible at all times.
- **Exceptionally Scalable and Flexible:** Ensure high scalability and elasticity to adapt to varying data demands and computational requirements.
- **Versatile Hybrid Search Functions:** Support hybrid search capabilities that blend different types of search techniques for enhanced performance and accuracy.

An example of creating index, inserting elements and searching
```python
import os
import time
import bam_modular
import read_DEEP
import pickle


ground_truth = read_DEEP.read_ibin(gt_path)
all_query, dim = read_DEEP.read_fbin(query_path)
data_path = './Data/DEEP/base.1B.fbin'
query_path = './Data/DEEP/query.public.10K.fbin'
gt_path = './Data/DEEP/groundtruth.public.10K.ibin'


# Check if paths exist
if not (os.path.exists(data_path) and os.path.exists(query_path) and os.path.exists(gt_path)):
    print("Error: One or more required files are missing.")
    exit(1)

print('Data path exists: ', os.path.exists(data_path),
      '\nQuery path exists: ', os.path.exists(query_path),
      '\nGT path exists: ', os.path.exists(gt_path))

# Prompt user for input
try:
    index_num = int(input('Select indexing algorithm from above list: \n 0 : HNSW \n 1 : FAISS \n 2 : SG \nIndex Number: '))
    m = int(input('Enter the maximum neighbors number: '))
    ef = int(input('Enter the maximum depth of search: '))
except ValueError:
    print("Invalid input. Please enter valid numbers.")
    exit(1)

# Set additional parameters
k = 100
num_centroids_per_query = 5

# Create BAM instance
bam_idx = bam_modular.BAM(data_path, m, ef)
name = data_path.split('/')[-1][:-5]
name = f"{name}_BAM_B"
idx_path = f"{name}/indexes_M{m}_ef{ef}"

if 'BAM_IDX.bin' not in os.listdir(idx_path):
    # Build index
    bam_idx.build_index(data_path,idx_path,index_num)

with open(f"{idx_path}/BAM_IDX.bin", "rb") as file:
    index = pickle.load(file)
# print("Loaded Data:", index)

# Perform search
t = time.time()
pred_idx, pred_dist = bam_idx.search(index, all_query, dim, gt_path, index_num, k, num_centroids_per_query)
total_time = time.time() - t

# Print results
bam_idx.print_results(pred_idx, ground_truth, total_time, k)

```

## For detailed API documentation see the [BAM documentation](https://git.txstate.edu/DataLab/hdIndexing/blob/main/python/BAM/files/documentation.md) file.

### Conclusion
Existing state-of-the-art indexing algorithms for nearest neighbor search have high memory usage, scalability issues due to hardware limitations, and impact performance and accessibility due to their intensive memory requirements. BAM divides the data into smaller chunks, uses clustering to manage these chunks, and then merges them into balanced clusters. A graph index is created for each cluster, and a query-aware search algorithm targets only relevant clusters, enhancing memory efficiency and retrieval speed. BAM performs comparably to leading methods such as DiskANN which involve complex processes like data partitioning, graph merging, and compression. BAM's approach simplifies the process while ensuring high efficiency and scalability.

### Next Steps
- **Hardware optimization:** We plan to optimize the library further for faster retrieval. Also, we plan to extend the functionalities to support distributed systems.


