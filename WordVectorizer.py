
from sklearn.feature_extraction.text import CountVectorizer

class WordVectorizer(object):
    '''Word vectorizer'''

    def __init__(self):
        self._vectorizer = CountVectorizer()

    def _map_word(self, word):
        w_map = self._vectorizer.vocabulary_.get(word)
        if w_map:
            return w_map
        return self._max_features

    def get_vocab_size(self):
        return self._max_features + 1

    def vectorize(self, data, split_c=' '):
        self._vectorizer.fit([data])
        self._max_features = len(self._vectorizer.vocabulary_)
        feature_list = list(map(self._map_word, data.split(split_c)))
        return feature_list

    def get_word_id(self, word):
        return self._vectorizer.vocabulary_.get(word)

    def get_word(self, id):
        if id < self._max_features - 1:
            return self._vectorizer.get_feature_names()[id]
        return None




