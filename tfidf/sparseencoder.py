import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

class SparseEncoder(object):
    def __init__(self, ngram, pretrained_dictionary, load_pretrained = False):
        self.ngram = ngram
        self.dictionary = pretrained_dictionary
        if ngram == 'ngram12':
            self.encoder = TfidfVectorizer(analyzer='char', ngram_range=(1,2))
        else:
            self.encoder = TfidfVectorizer(analyzer='char', ngram_range=(1,3))
        
        if load_pretrained:
            self.load_pretrained()

    def fit(self, train_corpus):
        self.encoder.fit(train_corpus)
        return self
    
    def save_pretrained(self, path):
        with open(path,'wb') as f:
            joblib.dump(self.encoder, f)
    
    def load_pretrained(self):
        self.encoder = joblib.load(f'pretrained/{self.ngram}/{self.dictionary}.pk','r')
        return self

    def transform(self, mentions):
        vec = self.encoder.transform(mentions)  # return sparse array
        return vec

    def __call__(self, mentions):
        return self.transform(mentions)

    def vocab(self):
        return self.encoder.vocabulary_
