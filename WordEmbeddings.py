import math
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle

class WordEmbeddings(object):
    '''Word Embeddings implementation using Noise Contrastive Estimation'''

    def __init__(self, embedding_size=200, skip_window=1, num_skips=2,\
                num_sampled=64, vocabulary_size=None):
        self._embedding_size = embedding_size  # Dimension of the embedding vector.
        self._skip_window = skip_window  # How many words to consider left and right.
        self._num_skips = num_skips  # How many times to reuse an input to generate a label.
        self._num_sampled = num_sampled  # Number of negative examples to sample.
        self._vocabulary_size = vocabulary_size  # Top words to consider from the vocabulary
        self._embeddings = None
    
    def train(self, X_train, y_train, learning_rate=1.0, batch_size=100, epochs=3):
        '''Trains the model using NCE implementation'''

        # If the vocabulary size is not specified we calculate it from the data
        if not self._vocabulary_size:
          self._vocabulary_size = len(set(X_train))
        
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        embeddings = tf.Variable(tf.random_uniform([self._vocabulary_size, \
                                                    self._embedding_size], -1.0, 1.0))
        embbed = tf.nn.embedding_lookup(embeddings, train_inputs)
        weights = tf.Variable(tf.truncated_normal([self._vocabulary_size,\
                                                    self._embedding_size],\
                                                    stddev=1.0 / math.sqrt(self._embedding_size)))
        biases = tf.Variable(tf.zeros([self._vocabulary_size]))
        loss = tf.reduce_mean(tf.nn.nce_loss(weights, biases, train_labels, embbed,\
                                                self._num_sampled, self._vocabulary_size))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

        with tf.Session() as session:
            print('Training starts...')
            session.run(tf.global_variables_initializer())
            step = 0
            avg_loss = 0
            for i in range(epochs):
                print('Epoch: ' + str(i))
                for X_train_batch, y_train_batch in \
                    self._generate_batch(X_train, y_train, batch_size):
                    # Last batch is ignored if < than batch_size
                    # TODO: Last batch should be also considered for the training
                    if len(X_train_batch) < batch_size:
                        break
                    feed_dict = {train_inputs: X_train_batch,\
                                train_labels: y_train_batch}
                    step += 1
                    _, current_loss = session.run([optimizer, loss], feed_dict=feed_dict)
                    avg_loss += current_loss
                    if step % 2000 == 0 and step > 0:
                        avg_loss /= 2000
                        print('Avg loss (last 2000 batches): ' + str(avg_loss))
                        average_loss = 0
                # Shuffle training set to improve generalization
                X_train, y_train = shuffle(X_train, y_train)
            self._embeddings = embeddings.eval()

    def get_embeddings(self):
        '''Returns embedding representation of words'''
        return self._embeddings

    def _generate_batch(self, X_train, y_train, batch_size):
        for i in range(0, len(X_train), batch_size):
            yield X_train[i:i + batch_size], y_train[i:i + batch_size]