

import numpy as np
import torchtext
from torchtext import data
from torchtext import datasets

thor_review = """
    The action scenes were top notch in this movie.
    Thor has never been this epic in the MCU.
    He does some pretty epic shit in this movie and he is definitely not under-powered anymore.
    Thor in unleashed in this, I love that.
"""


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.length = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = self.length
            self.length += 1

    def __len__(self):
        return len(self.idx2word)

    def onehot_encoded(self, word):
        vec = np.zeros(self.length)
        vec[self.word2idx[word]] = 1
        return vec

# dic = Dictionary()
# print(thor_review.split())
# for tok in thor_review.split():
#     dic.add_word(tok)
# print(dic.word2idx)
# print(dic.onehot_encoded('were'))

TEXT = data.Field(lower=True, batch_first=True, fix_length=20)
LABEL = data.Field(sequential=False)

datasets.IMDB.splits()
TEXT.build_vocab()
