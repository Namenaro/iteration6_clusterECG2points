from get2datasets import get2datasets

from sklearn.cluster import AffinityPropagation
import numpy as np
import matplotlib.pyplot as plt

def get_discrete_labels(X1, X2):
    y1 = AffinityPropagation().fit(X1).labels_
    y2 = AffinityPropagation().fit(X2).labels_
    return y1, y2

def split_y2_by_y1(Y1, Y2):
    result = []
    clusters1 = len(np.unique(Y1))
    for _ in range(clusters1):
        result.append([])
    for i in range(len(Y1)):
        result[Y1[i]].append(Y2[i])
    return result

def show_conditional_hists(y2_by_y1, n_bins):
    num_hists = len(y2_by_y1)
    fig, axs = plt.subplots(num_hists, sharey=True, sharex=True)
    fig.suptitle('Vertically stacked subplots')
    for i in range(num_hists):
        axs[i].hist(y2_by_y1[i], n_bins, histtype='bar')

    plt.show()

def visualise_conditionality_of_clusters(Y1, Y2):
    n_bins = len(np.unique(Y2))
    y2_by_y1 = split_y2_by_y1(Y1, Y2)
    show_conditional_hists(y2_by_y1, n_bins)

if __name__ == "__main__":
    X1, X2 = get2datasets()
    Y1, Y2 = get_discrete_labels(X1, X2)
    visualise_conditionality_of_clusters(Y1, Y2)



