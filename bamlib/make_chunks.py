import read_DEEP
import read_SIFT
import os
import numpy as np

# data_path = '/home/m_r1117/Desktop/Toufik/Practice/Indexing/Benchmark_Data/DEEP/base.1B.fbin'
# out_path = '/home/m_r1117/Desktop/Toufik/Practice/Indexing/indexing-billion/'
# write_path = os.path.join(out_path, 'chunks/')

# if not os.path.exists(write_path):
#     os.makedirs(write_path)

# chunk_size = 1000000

# def create_chunk(data_name, data_path, out_path, chunk_size = 1000000, total_data_size = 10000000):

#     if data_name == 'deep':

#         start_idx = 0
#         counter = 0
#         while(start_idx < total_data_size):
#             current_chunk, dim = read_DEEP.read_fbin(data_path,start_idx,chunk_size)
#             name = 'chunk' + str(counter) +'.fbin'
#             file_path = os.path.join(out_path, name)
#             print(file_path)
#             read_DEEP.write_fbin(file_path, current_chunk)
#             counter += 1
#             start_idx += chunk_size
    
#     return dim

 
def create_chunk(data_name, data_path, out_path, chunk_size=1000000, total_data_size=10000000):
    print('Creating chunks...')
    if data_name == 'deep':
        start_idx = 0
        counter = 0
        while start_idx < total_data_size:
            current_chunk, dim = read_DEEP.read_fbin(data_path, start_idx, chunk_size)
            indices = np.arange(start_idx, start_idx + len(current_chunk)).reshape(-1, 1)
            current_chunk_with_index = np.concatenate((current_chunk, indices), axis=1)
 
            name = f'chunk{counter}.npy'
            file_path = os.path.join(out_path, name)
            np.save(file_path, current_chunk_with_index)
            # print(f'Saved: {name}')
            counter += 1
            start_idx += chunk_size

    if data_name == 'sift':
        start_idx = 0
        counter = 0
        while start_idx < total_data_size:
            current_chunk, dim = read_SIFT.fvecs_read(data_path, start_idx, chunk_size)
            indices = np.arange(start_idx, start_idx + len(current_chunk)).reshape(-1, 1)
            current_chunk_with_index = np.concatenate((current_chunk, indices), axis=1)
 
            name = f'chunk{counter}.npy'
            file_path = os.path.join(out_path, name)
            np.save(file_path, current_chunk_with_index)
            # print(f'Saved: {name}')
            counter += 1
            start_idx += chunk_size