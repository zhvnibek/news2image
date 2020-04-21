import os
import numpy as np
from random import randint, sample
from scipy.linalg import subspace_angles
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.cluster import AgglomerativeClustering

WS_LOC = "/home/zhanibek/Desktop/Fall '19/Senior Project/news2image/data/subspaces/goi5k"


def get_subspace(folder: str = WS_LOC, index=None) -> np.ndarray:
    npy_subspaces = os.listdir(folder)
    if index is None:
        index = randint(0, len(npy_subspaces))
    S = np.load(os.path.join(folder, npy_subspaces[index]))
    return S


def sample_subspaces(folder: str = WS_LOC, count: int = 10):
    sample_subs = sample(population=os.listdir(folder), k=count)
    return np.asarray([np.load(os.path.join(folder, sub)) for sub in sample_subs], dtype=np.float32)
#     return [np.load(os.path.join(WS_LOC, sub)) for sub in sample_subs]


def subspaces_similarity(S1, S2):
    canon_angles = subspace_angles(S1, S2)
    return np.average(np.square(np.cos(canon_angles)))


def dist(x, y):
    return randint(0, 10)


if __name__ == '__main__':
    from pprint import pprint
    # sub_one = get_subspace()
    # sub_two = get_subspace()
    # sim = subspaces_similarity(S1=sub_one, S2=sub_two)
    # print(sim)
    subs = sample_subspaces()
    X = np.array(
        [[5, 3],
         [10, 15],
         [15, 12],
         [24, 10],
         [30, 30],
         [85, 70],
         [71, 80],
         [60, 78],
         [70, 55],
         [80, 91], ])
    print(X.ndim)
    cond_dm = np.array([1, 3, 1, 2, 2, 1])
    linked = linkage(y=cond_dm, method='single', metric=dist)
    pprint(linked)
