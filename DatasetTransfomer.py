import random
import numpy as np

class DatasetTransformer(object):

    def transform(self, data):
        raise NotImplementedError


class SkipGramTransformer(DatasetTransformer):

    def __init__(self, skip_window, num_skips):
        self._skip_window = skip_window
        self._num_skips = num_skips

    def transform(self, data):
        span = 2 * self._skip_window + 1
        X = []
        y = []
        for i in range(len(data) - span + 1):
            window = data[i: i + span]
            train = window[self._skip_window]
            del window[self._skip_window]
            for _ in range(self._num_skips):
                target = random.choice(window)
                window.remove(target)
                X.append(train)
                y.append(target)
        # Transforms lists to numpy arrays and adjusts the shape
        # TODO: improve performance, X_train, y_train should already
        # be numpy arrays
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)
        return X, y