import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import time


def recall(predicted_array, true_array):
    recalls = []

    for pred_row, gt_row in zip(predicted_array, true_array):

        match = np.intersect1d(pred_row, gt_row)
        recall_row = len(match)/len(gt_row)
        recalls.append(recall_row)

    return np.mean(recalls)


def recall_lists(pred_lists,true_lists):
    recalls = []
    for true, pred in zip(true_lists, pred_lists):
        true_set = set(true)
        pred_set = set(pred)
        if len(true_set) == 0:
            continue  # Avoid division by zero if the true set is empty
        true_positives = len(true_set.intersection(pred_set))
        false_negatives = len(true_set - pred_set)
        recall = true_positives / (true_positives + false_negatives)
        recalls.append(recall)
    return sum(recalls) / len(recalls) if recalls else 0 

def recall_at_k(pred_lists,true_lists,at_k):
    hits = 0
    for p,t in zip(pred_lists,true_lists):
        print(f'XXXXXXXXXXXXX: {p}')
        if p[0] in t[:at_k]:
            hits +=1
    return hits/len(true_lists)

# pred = np.array([[1,2,3], [4,5,6], [7,8,9], [10,11,12]])
# true = np.array([[1,2,3], [4,5,7], [6,8,9], [10,14,12]])

# rec = recall(pred,true)
# print(rec)