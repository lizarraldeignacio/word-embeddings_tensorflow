import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class WordSimilarity(object):

    def most_similar(self, embeddings, word, k = 5):
        '''Returns the k most similar words'''
        similarity = cosine_similarity(embeddings, word)
        similarity = np.fromiter(map(lambda x: x[0], similarity), dtype=np.float)
        return similarity.argsort()[-k:][::-1]
