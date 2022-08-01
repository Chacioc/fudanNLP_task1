import numpy as np


class Ngram:
    def __init__(self, ngram=3):
        self.ngram = ngram
        self.vocab = {}

    def fit_transform(self, sent_list):
        for sent in sent_list:
            sent = sent.lower()
            words = sent.split()
            for gram in range(self.ngram):
                for i in range(len(words) - gram + 1):
                    word = "_".join(words[i: i + gram])
                    if word not in self.vocab:
                        self.vocab[word] = len(self.vocab)
        return self.transform(sent_list)

    def transform(self, sent_list):
        features = np.zeros((len(sent_list), len(self.vocab)))
        for idx, sent in enumerate(sent_list):
            sent = sent.lower()
            words = sent.split()
            for gram in range(self.ngram):
                for i in range(len(words) - gram + 1):
                    word = "_".join(words[i: i + gram])
                    if word in self.vocab:
                        features[idx][self.vocab[word]] = 1
        return features
