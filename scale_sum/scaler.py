import numpy as np
import scipy.stats as stats
from joblib import Parallel, delayed
from sklearn.preprocessing import normalize

class Scaler():
    def __init__(self,):
        pass

    def cos_product_sparse(self, input_a, input_b):
        scores = normalize(input_a,axis=1).dot(normalize(input_b,axis=1).transpose())
        return scores        

    def inner_product(self, input_a, input_b):
        scores = input_a.dot(input_b.transpose())
        return scores

    def cos_product(self, input_a, input_b):
        scores = normalize(input_a,axis=1).dot(normalize(input_b,axis=1).transpose())
        return scores

    def tanh(self, x):
        m = np.mean(x, axis=(0,1))
        std = np.std(x, axis=(0,1))
        x_scaled = 0.5 * (np.tanh(0.01 * ((x - m) / std)) + 1)
        return x_scaled

    def zscore(self,x):
        x_scaled = stats.zscore(x, axis = (0,1))
        return x_scaled

    def min_max(self,x):
        max = np.max(x, axis = (0,1))
        min = np.min(x, axis = (0,1))
        x_scaled = (x-min)/(max-min)
        return x_scaled

    def scale_sum(self, x, y, scaler = 'tanh', dense_ratio=1, sparse_ratio=1):
        '''
        x: dense score, y: sparse score, ratio search.
        '''
        if scaler == 'tanh':
            sumed = self.tanh(x) + self.tanh(y)
        elif scaler == 'zscore':
            sumed = self.zscore(x) + self.zscore(y)
        elif scaler == 'min_max':
            sumed = self.min_max(x) + self.min_max(y)
        elif scaler == 'weighted_sum':
            sumed = dense_ratio*x + sparse_ratio*y
        else:
            sumed = x+y
        return sumed

    def index_argsort(self, matrix, topk=10):   
        index = np.argsort(matrix)[:,-topk:]
        reverse_index = np.array([i[::-1] for i in index],dtype= np.int32)
        return reverse_index

    def batch_argsort(self, scores):
        batch_size=128
        iterations = range(0, scores.shape[0],batch_size)
        matrix_batches = [scores[start:min(start+batch_size,scores.shape[0])] for start in iterations]
        candidates_index = Parallel(n_jobs=-1)(delayed(self.index_argsort)(i) for i in matrix_batches)
        candidates_index = np.concatenate(candidates_index, axis=0).squeeze()
        return candidates_index