class LSHIndex():
    def __init__(self, vectors, labels):
        self.dimension = vectors.shape[1]
        self.vectors = vectors.astype('float32')
        self.labels = labels

    def build(self, num_bits=8):
        self.index = faiss.IndexLSH(self.dimension, num_bits)
        self.index.add(self.vectors)

    def query(self, vectors, k=10):
        distances, indices = self.index.search(vectors, k)
        print(indices[0])
        for i in indices[0]:
            print(self.labels[i])
        # I expect only query on one vector thus the slice
        return [self.labels[i] for i in indices[0]]

import annoy

import os
import faiss
#from sklearn.cluster import AgglomerativeClustering
import sys
import argparse

import lmdb

path = os.getcwd()
num_trees=30
import numpy as np


data1 = np.load(path + '/goi5k/5613638506_3c5db3e3ab_o.npy')
names = []

vectors = np.empty([300, 5])

for fn in os.listdir(path + '/goi5k/'):
    names.append(fn.strip(".npy"))
    #f.write(fn.strip(".npy") + "\n")

for fn in os.listdir(path + '/goi5k/'):
    data = np.load(path + '/goi5k/' + fn)
    np.append(vectors, data)

#print(vectors.shape)
#print(names[1])
#clustering = AgglomerativeClustering(n_clusters=None, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=0.001).fit(vectors)


index = LSHIndex(vectors, names)
index.build()
index.query(np.ascontiguousarray(data1, dtype=np.float32))