import pickle
import argparse
import numpy as np
from WordEmbeddings import WordEmbeddings
from WordVectorizer import WordVectorizer
from WordSimilarity import WordSimilarity
from DatasetTransfomer import SkipGramTransformer

def load_dataset(path):
    with open(path, 'r', encoding='utf8') as data:
        return data.readline()

BATCH_SIZE = 128
EMBEDDING_SIZE = 300  # Dimension of the embedding vector.
SKIP_WINDOW = 4       # How many words to consider left and right.
NUM_SKIPS = 3         # How many times to reuse an input to generate a label.
NUM_SAMPLED = 64      # Number of negative examples to sample.
EPOCHS = 1


parser = argparse.ArgumentParser(description='Generates word embeddings')
parser.add_argument('dataset', required=True, type=str , help='dataset path')
parser.add_argument('--out', required=True, type=str, default='embeddings', help='output file path')
parser.add_argument('--batch', type=int, default=BATCH_SIZE, help='batch size')
parser.add_argument('--esize', type=int, default=EMBEDDING_SIZE, help='dimension of the embedding vector')
parser.add_argument('--skips', type=int, default=SKIP_WINDOW, help='how many words to consider left and right')
parser.add_argument('--nskips', type=int, default=NUM_SKIPS, help='how many times to reuse an input to generate a label.')
parser.add_argument('--sampled', type=int, default=NUM_SAMPLED, help='number of negative examples to sample')
parser.add_argument('--epochs', type=int, default=EPOCHS, help='number of iterations')


args = parser.parse_args()
print(args)

def generate_embeddings(from_file = False):
    global args

    if from_file:
        vectorizer = pickle.load(open("word_vectors.pkl", "rb"))
        word_embeddings = np.load('embeddings.npy')
        return word_embeddings, vectorizer
    print('Loading data...')
    data = load_dataset(args.dataset)
    print('Vectorizing...')
    vectorizer = WordVectorizer()
    word_vectors = vectorizer.vectorize(data)
    pickle.dump(vectorizer, open("word_vectors.pkl", "wb"))
    #Hint to release memory
    del data
    print('Creating skip-gram representation...')
    dataset_trasformer = SkipGramTransformer(args.skips, args.nskips)
    X_train, y_train = dataset_trasformer.transform(word_vectors)
    '''embedding_size=200, skip_window=1, num_skips=2,\
                num_sampled=64, vocabulary_size=None'''
    embeddings = WordEmbeddings(vocabulary_size=vectorizer.get_vocab_size(),\
                                    embedding_size=args.esize)
    embeddings.train(X_train, y_train, epochs=args.epochs, learning_rate=1.0)
    return embeddings.get_embeddings(), vectorizer

word_embeddings, vectorizer = generate_embeddings(from_file=False)

#Print similar words
print('Training finished')
word = 'argentina'
print('5 most similar words to ' + word + ':')
similarity = WordSimilarity()
print('Word embeddings shape')
print(word_embeddings.shape)
word_embedding = word_embeddings[vectorizer.get_word_id(word), :]
np.save(args.out, word_embeddings)
most_similar = similarity.most_similar(word_embeddings, word_embedding.reshape(1, -1))
for id in most_similar:
    word = vectorizer.get_word(id)
    if word:
        print(word)
