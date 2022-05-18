import os
import sys
SCRIPT = os.path.dirname(os.path.abspath('bert_recom'))
sys.path.append(os.path.dirname(SCRIPT))
from bert.bertmodel import BertModel
from bert.tokenizer import Tokenizer
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import torch.nn
from sklearn.preprocessing import normalize
from joblib import Parallel, delayed

# def index_argsort(matrix, topk=10):   
#     index = np.argsort(matrix)[:,-topk:]
#     reverse_index = np.array([i[::-1] for i in index],dtype= np.int32)
#     return reverse_index

class Bert_Re():
    def __init__(self, use_cuda = "False", load_pretrained=True, checkpoint = 'sapbert'):
        self.tokenizer = Tokenizer(checkpoint = checkpoint)
        self.encoder = BertModel(use_cuda = use_cuda, load_pretrained = load_pretrained, checkpoint = checkpoint)

    def tokenize(self, terms):
        tokens = self.tokenizer.tokenize(terms)
        return tokens

    def embed_dense(self, tokens ,show_progress=True):
        
        self.encoder.eval() # prevent dropout
        
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        batch_size=1024
        dense_embeds = []

        with torch.no_grad():
            if show_progress:
                iterations = tqdm(range(0, input_ids.shape[0], batch_size))
            else:
                iterations = range(0, input_ids.shape[0], batch_size)
                
            for start in iterations:
                end = min(start + batch_size, input_ids.shape[0])
                batch_ids = input_ids[start:end]
                batch_masks = attention_mask[start:end]
                ## use saved tokenize.pt
                batch_dense_embeds = self.encoder(batch_ids, batch_masks)
                # batch_dense_embeds = batch_dense_embeds.detach().cpu().numpy()
                dense_embeds.append(batch_dense_embeds.detach().cpu().numpy().astype(np.float32))
                torch.cuda.empty_cache()
            
        dense_embeds = np.concatenate(dense_embeds, axis = 0)
        
        return dense_embeds   

    def inner_product(self, input_a, input_b):
        scores = input_a.dot(input_b.transpose())
        return scores

    def cos_product(self, input_a, input_b):
        scores = normalize(input_a,axis=1).dot(normalize(input_b,axis=1).transpose())
        return scores

    def index_argsort(self, matrix, topk=500):   
        index = np.argsort(matrix)[:,-topk:]
        reverse_index = np.array([i[::-1] for i in index],dtype= np.int32)
        return reverse_index

    def batch_argsort(self, scores):
        batch_size=64
        iterations = range(0, scores.shape[0],batch_size)
        matrix_batches = [scores[start:min(start+batch_size,scores.shape[0])] for start in iterations]
        candidates_index = Parallel(n_jobs=-1)(delayed(self.index_argsort)(i) for i in matrix_batches)
        candidates_index = np.concatenate(candidates_index, axis=0).squeeze()
        return candidates_index
    
    def predict_labels(self, querycui, candidates, candidates_index, scores):
        candidates = np.array(candidates).reshape(1,-1)
        candidates_cui = np.take_along_axis(candidates, candidates_index, axis = -1)
        candidates_scores = np.take_along_axis(scores, candidates_index, axis = -1)
        labels = np.array([[1 if len(set(q.split('|')).intersection((x.split('|')))) > 0 else 0 for x in c] for q, c in zip(querycui, candidates_cui)])
        return labels, candidates_cui, candidates_scores

    def predict_labels_multi(self, querycui, candidates, scores, topk= 10):
        index = np.argsort(scores, axis=-1)[:, -topk:]
        candidates_index = np.array([i[::-1] for i in index],dtype= np.int32)
        candidates_cui = np.take_along_axis(candidates, candidates_index, axis = -1)
        labels = np.array([[1 if len(set(q.split('|')).intersection((x.split('|')))) > 0 else 0 for x in c] for q, c in zip(querycui, candidates_cui)])
        return labels, candidates_cui
    
    def accuracy(self, labels):
        accs = []
        for i in [1,5,10]:
            top = round(np.array([1 for i in labels[: , :i] if i.sum() > 0], dtype=np.int32).sum() / labels.shape[0] * 100,2)
            accs.append(top)
        return accs
