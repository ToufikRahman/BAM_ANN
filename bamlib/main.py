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

