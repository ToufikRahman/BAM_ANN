import numpy as np

def get_data_and_true_idx(data_path):

    vectors = np.load(data_path)
    vec_new = vectors[:,:-1]
    indexes = vectors[:,-1]
    indexes = indexes.tolist()

    idx = list(map(int,indexes))

    return vec_new,idx