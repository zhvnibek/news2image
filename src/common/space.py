import numpy as np
from collections import Counter
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA
from scipy.linalg import subspace_angles

class Space:

    def __init__(self, keyed_vectors: KeyedVectors):
        self.keyed_vectors = keyed_vectors
        self.embed_size: int = 300

    def get_embedding(self, word: str) -> np.ndarray:
        """ What is the vector for missing words? """
        return self.keyed_vectors[word] if word in self.keyed_vectors else None

    def _vectorize_keywords(self, keywords: Counter, use_tf: bool = True) -> np.ndarray:
        if not use_tf:
            return np.array([
            self.get_embedding(word)
            for word in keywords.keys()
            if self.get_embedding(word) is not None
        ])
        return np.array([
            self.get_embedding(word) * count
            for word, count in keywords.items()
            if self.get_embedding(word) is not None
        ])

    def _autocorr(self, matrix) -> np.ndarray:
        return matrix.T.dot(matrix)

    def create_subspace(self, keywords: Counter, dims: int, use_tf: bool = True) -> np.ndarray:
        embeddings = self._vectorize_keywords(keywords, use_tf)
        R = self._autocorr(embeddings)
        return PCA(n_components=dims).fit_transform(R)

    def subspaces_similarity(self, S1, S2):
        canon_angles = subspace_angles(S1, S2)
        s = np.average(np.square(np.cos(canon_angles)))
        return s
