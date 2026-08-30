from typing import List, Tuple, Optional, Dict


def normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization of 2D array X."""
    X = np.asarray(X, dtype=np.float32)
    norms = np.linalg.norm(X, axis=1, keepdims=True) #Frobenius norm, sum of absolute values
    norms = np.maximum(norms, eps) #mak sure norms is > 0 
    return X / norms


def getRelevance(id_lst, cosine_dict):
    '''set cosine similarity threshold of 0.4'''
    bm_sim = {key: cosine_dict[key] for key in id_lst if key in cosine_dict}
    #print(bm_sim)
    relevance = [1 if i > 0.4 else 0 for i in list(bm_sim.values())]
    return relevance

# -----------------------------
# Recall@k
# -----------------------------
def recall_at_k(relevance, k):
    """
    relevance: list or list of lists. Relevance labels (1 = relevant, 0 = not)
    k: cutoff rank
    """
    rel = relevance
    recalls = []

    for rel_list in rel:
        R = sum(rel_list)
        if R == 0:
            recalls.append(0.0)
            continue
        recalls.append(sum(rel_list[:k]) / R)

    return recalls if len(recalls) > 1 else recalls[0]

# -----------------------------
# Average Precision (AP)
# -----------------------------
def average_precision(relevance):
    """
    relevance: list of 0/1 integers for one query
    """
    rel_list = relevance
    R = sum(rel_list)
    if R == 0:
        return 0.0

    precisions = []
    relevant_found = 0

    for i, r in enumerate(rel_list, start=1):
        if r == 1:
            relevant_found += 1
            precisions.append(relevant_found / i)

    return sum(precisions) / R


# -----------------------------
# MAP (Mean Average Precision)
# -----------------------------
def mean_average_precision(relevance_lists):
    """
    relevance_lists: list of lists, where each list is relevance for one query
    """
    rel = relevance_lists
    aps = [average_precision(r) for r in rel]
    return sum(aps) / len(aps)
# -----------------------------
# NDCG@k
# -----------------------------
def ndcg_at_k(relevance, k):
    """
    relevance: list or list of lists. Relevance scores (0,1,2...)
    k: cutoff
    """
    rel = relevance
    ndcgs = []

    for rel_list in rel:
        # DCG
        dcg = 0.0
        for i, r in enumerate(rel_list[:k], start=1):
            dcg += (2 ** r - 1) / np.log2(i + 1)

        # IDCG: sort by best possible
        ideal = sorted(rel_list, reverse=True)
        idcg = 0.0
        for i, r in enumerate(ideal[:k], start=1):
            idcg += (2 ** r - 1) / np.log2(i + 1)

        ndcgs.append(0.0 if idcg == 0 else dcg / idcg)

    return ndcgs if len(ndcgs) > 1 else ndcgs[0]
