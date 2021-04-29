from get2datasets import get2datasets

from sklearn.cluster import AffinityPropagation

import numpy as np

def get_discrete_labels(X1, X2):
    y1 = AffinityPropagation(random_state=5).fit(X1).labels_
    y2 = AffinityPropagation(random_state=5).fit(X2).labels_
    return y1, y2


if __name__ == "__main__":
    X1, X2 = get2datasets()
    y1, y2 = get_discrete_labels(X1, X2)
    clusters1 = len(np.unique(y1))
    clusters2 = len(np.unique(y2))


