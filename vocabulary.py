import nltk
import numpy as np
import pickle
from collections import Counter
from nltk.corpus import stopwords

nltk.download('stopwords') 
nltk.download('punkt') 
stoplist = stopwords.words('english')



class Vocabulary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

def build_vocab(threshold=3):

    corpus = np.load('train.npy',allow_pickle=True)
    counter = Counter()
    
    for sent in corpus:
        sent = sent[0]
        tokens = nltk.tokenize.word_tokenize(sent.lower())
        counter.update(tokens)


    # Ignore rare words
    words = [word for word, cnt in counter.items() if (cnt >= threshold and word not in stoplist)]

    # Create a vocabulary and initialize with special tokens
    vocab = Vocabulary()
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<pad>')
    vocab.add_word('<unk>')

    # Add the all the words
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


if __name__ == '__main__':
    
    vocab = build_vocab()
    with open('./vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
