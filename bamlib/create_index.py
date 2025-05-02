import numpy as np
import hnswlib
import os
import process_data
import pickle

def build_index(data_path, index_num, out_path, index_name, param_list):
    
    # HNSW
    if index_num == 0 :
        print(f'Creating HNSW index {index_name}...')
        m = param_list[0]
        ef = param_list[1]
        thread_num = 4
        space='l2'
        all_data_path_list = os.listdir(data_path)
        
        for single_data in all_data_path_list:
            single_data_path = os.path.join(data_path,single_data)
            vectors, true_idx =  process_data.get_data_and_true_idx(single_data_path)
            # print(f'Data:   {single_data_path}  Vectors: {vectors.shape}  True index: {true_idx} \n')
            num_elements = len(vectors)
            dim = vectors.shape[1]
            name = single_data_path.split('/')[-1][:-4]
            full_name = f'{name}_{index_name}'
            idx_path = f'{out_path}/{full_name}'
            true_idx_file = full_name.split('.')[0]
            full_true_idx_file = f'{true_idx_file}.pkl'
            full_true_idx_file_path = f'{out_path}/{full_true_idx_file}'
            # print(full_true_idx_file_path)
            print(idx_path)
            # with open(full_true_idx_file_path, 'wb') as f:
            #     pickle.dump(true_idx,f)

            
            index = hnswlib.Index(space, dim)
            vectors1 = vectors[:num_elements // 2]
            vectors2 = vectors[num_elements // 2:]
            true_idx1 = true_idx[:num_elements // 2]
            true_idx2 = true_idx[num_elements // 2:]
            index = hnswlib.Index(space=space,dim=dim) 
            index.init_index(ef_construction=ef, M=m ,max_elements=num_elements//2)
            index.set_ef(ef)
            index.set_num_threads(thread_num)
            index.add_items(vectors1,true_idx1)
            index.save_index(idx_path)
            del index

            index = hnswlib.Index(space=space,dim=dim)
            index.load_index(idx_path, max_elements = num_elements)
            index.add_items(vectors2,true_idx2)
            index.save_index(idx_path)
        idx_dict = {}
        for filename in os.listdir(out_path):
            print(filename)
            index = hnswlib.Index(space='l2', dim=dim)
            index.init_index(max_elements=num_elements, ef_construction=ef, M=m)
            file_path = os.path.join(out_path, filename)
            index.load_index(file_path)
            idx_dict[filename] = index
            # print(index.max_elements)
            del index
        with open(f"{out_path}/BAM_IDX.bin", "wb") as file:
            pickle.dump(idx_dict, file)

    # FAISS
    if index_num == 1 :
        print(f'Creating FAISS index {index_name}...')


    # SG
    if index_num == 2 :
        print(f'Creating SG index {index_name}...')
    


def loading_index(key, cluster_path, index_num, index_path, index_name, dim):
    # HNSW
    if index_num == 0 :
        
        # full_data_path = f'{cluster_path}/{data_file_name}'
        name = key.split('.')[0]
        idx_path = f'{index_path}/{name}_{index_name}'
        # name_2 = index_name.split('.')[0]
        # true_idx_file_name = f'{name}_{name_2}.pkl'
        # true_idx_file_path = f'{out_path}/{true_idx_file_name}'
        # print(f'Loading HNSW index: {idx_path}')

        space='l2'     

        # vectors, true_idx =  process_data.get_data_and_true_idx(full_data_path)
        # # print(full_data_path)
              
        # num_elements = len(vectors)
        # dim = vectors.shape[1]
        index = hnswlib.Index(space=space,dim=dim)
        index.load_index(idx_path)
        # with open(true_idx_file_path, 'rb') as f:
        #     true_idx = pickle.load(f)

    # FAISS
    if index_num == 1 :
        print(f'Loading FAISS index {index_name}...')


    # SG
    if index_num == 2 :
        print(f'Loading SG index {index_name}...')

    return index


# def load_index(data_path, index_num, out_path, index_name, dim):
#     # HNSW
#     if index_num == 0 :
#         print(f'Loading HNSW index {index_name}...')

#         name = single_data_path.split('/')[-1][:-4]
#         idx_path = f'{out_path}/{name}_{index_name}'
#         space='l2'

#         all_data_path_list = os.listdir(data_path)
        
#         for single_data in all_data_path_list:
#             single_data_path = os.path.join(data_path,single_data)
#             vectors, true_idx =  process_data.get_data_and_true_idx(single_data_path)
#             # print(f'Data:   {single_data_path}  Vectors:    {vectors.shape}  True index:    {len(true_idx)} \n')
#             num_elements = len(vectors)
#             dim = vectors.shape[1]
#             name = single_data_path.split('/')[-1][:-4]
#             idx_path = f'{out_path}/{name}_{index_name}'
#             index = hnswlib.Index(space=space,dim=dim)
#             index.load_index(idx_path,num_elements)

#     # FAISS
#     if index_num == 1 :
#         print(f'Loading FAISS index {index_name}...')


#     # SG
#     if index_num == 2 :
#         print(f'Loading SG index {index_name}...')

#     return index