def search(index, index_num, query, k_neighbors=100):

    if index_num == 0:
        labels, distances = index.knn_query(query, k=k_neighbors)
        # print('Labels: ',labels)
        # labels = labels[0]
        # distances = distances[0]

    if index_num == 1:
        print('Faiss not supported yet!')

    return labels, distances