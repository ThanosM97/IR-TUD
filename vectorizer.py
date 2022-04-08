"""This module implements a vectorizer for the word embedding models."""
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


class Word2VecVectorizer:
    """Word vectorizer for Glove and FastText.

    Args:
        - model (KeyedVectors): Gensim's KeyedVectors of Glove or FastText.
    """

    def __init__(self, model: KeyedVectors):
        self.word_vectors = model

    def transform(self, data):
        # determine the dimensionality of vectors
        v = self.word_vectors.get_vector('king')
        self.D = v.shape[0]

        X = np.zeros((len(data), self.D))
        n = 0
        emptycount = 0
        for sentence in data:
            tokens = sentence.split()
            vecs = []
            m = 0
            for word in tokens:
                try:
                    # throws KeyError if word not found
                    vec = self.word_vectors.get_vector(word)
                    vecs.append(vec)
                    m += 1
                except KeyError:
                    pass
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
            n += 1
        return X
