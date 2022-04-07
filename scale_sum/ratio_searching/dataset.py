import pandas as pd
import numpy as np
import argparse 
import joblib
from scaler import Scaler
import time
import gc
from tqdm import tqdm

def predict_labels(querycui, candidates, candidates_index, bert_scores, tfidf_scores):
    candidates = np.array(candidates).reshape(1,-1)
    candidates_cui = np.take_along_axis(candidates, candidates_index, axis = -1)
    bert_candidates_scores = np.take_along_axis(bert_scores, candidates_index, axis = -1)
    tfidf_candidates_scores = np.take_along_axis(tfidf_scores, candidates_index, axis = -1)
    labels = np.array([[1 if len(set(q.split('|')).intersection((x.split('|')))) > 0 else 0 for x in c] for q, c in zip(querycui, candidates_cui)])
    return labels, bert_candidates_scores, tfidf_candidates_scores

def predict_labels_multi(querycui, candidates, scores, topk= 10):
    index = np.argsort(scores, axis=-1)[:, -topk:]
    candidates_index = np.array([i[::-1] for i in index],dtype= np.int32)
    candidates_cui = np.take_along_axis(candidates, candidates_index, axis = -1)
    labels = np.array([[1 if len(set(q.split('|')).intersection((x.split('|')))) > 0 else 0 for x in c] for q, c in zip(querycui, candidates_cui)])
    return labels, candidates_cui

def accuracy(labels):
    accs = []
    for i in [1,5,10]:
        top = round(np.array([1 for i in labels[: , :i] if i.sum() > 0], dtype=np.int32).sum() / labels.shape[0] * 100,2)
        accs.append(top)
    return accs

parser = argparse.ArgumentParser()
parser.add_argument('-q','--query')
parser.add_argument('-d','--dictionary')
parser.add_argument('--checkpoint')
parser.add_argument('--method')

args = parser.parse_args()

## instantiate scaler
scaler = Scaler()

## load matrices
tfidf_dictionary_matrix_path = f'../../tfidf/dictmatrix/ngram13/{args.dictionary}.pk'
print(f'reading tfidf matrix: {time.asctime()}')
with open(tfidf_dictionary_matrix_path,'rb') as f:
    tfidf_dictionary_matrix = joblib.load(f).astype(np.float32)
print(f'reading bert matrix: {time.asctime()}')
bert_dictionary_matrix_path = f'../../bert/bert_semantic_matrix/{args.checkpoint}_{args.dictionary}.pk'
with open(bert_dictionary_matrix_path,'rb') as f:
    bert_dictionary_matrix = joblib.load(f).astype(np.float32)

## load_dictioanry
dictionary_path = f'../../data/dictionary/{args.dictionary}.csv'
query_path = f'../../data/query/{args.query}.csv'
query_df = pd.read_csv(query_path)
querycui = list(query_df['cui'])
dictionary_df = pd.read_csv(dictionary_path)
dictcui = list(dictionary_df['cui'])

## single translate
translates = ['baidu']

labels = []
bert_candidates_scores = []
tfidf_candidates_scores = []
for translate in translates:
    tfidf_query_matrix_path = f'../../tfidf/dictmatrix/ngram13/{args.query}_{args.dictionary}_{translate}.pk'
    print(f'reading tfidf query matrix: {time.asctime()}')
    with open(tfidf_query_matrix_path , 'rb') as f:
        tfidf_query_matrix = joblib.load(f)
    print(f'transforming tfidf matrix: {time.asctime()}')
    tfidf_query_matrix = tfidf_query_matrix.astype(np.float32)
    bert_query_matrix_path = f'../../bert/bert_semantic_matrix/{args.checkpoint}_{args.query}_{translate}.pk'
    with open(bert_query_matrix_path,'rb') as f:
        bert_query_matrix = joblib.load(f).astype(np.float32)
    batch_size = 4096
    iterations = [i for i in range(0, len(querycui),batch_size)]


    for start in tqdm(iterations,total= len(iterations)):
        end = min(start+batch_size, len(querycui))
        batch_bert_query_matrix = bert_query_matrix[start:end, :]
        print(f'start bert multiply: {time.asctime()}')
        batch_bert_score = scaler.cos_product(batch_bert_query_matrix, bert_dictionary_matrix)
        print(f'end bert multiply: {time.asctime()}')
        del batch_bert_query_matrix
        gc.collect()
        print(f'start tfidf multiply: {time.asctime()}')
        batch_tfidf_query_matrix = tfidf_query_matrix[start:end,:]
        batch_tfidf_score = []
        tfidf_iterations = [i for i in range(0, batch_tfidf_query_matrix.shape[0], 1024)]
        for t_start in tqdm(tfidf_iterations,total=len(tfidf_iterations)):
            t_end = min(t_start+1024, batch_tfidf_query_matrix.shape[0])
            t_batch_tfidf_score = scaler.cos_product_sparse(batch_tfidf_query_matrix[t_start:t_end, :], tfidf_dictionary_matrix).toarray()
            batch_tfidf_score.append(t_batch_tfidf_score)
        batch_tfidf_score = np.concatenate(batch_tfidf_score, axis=0)
        print(f'end tfidf multiply: {time.asctime()}')
        del batch_tfidf_query_matrix
        gc.collect()
        print(f'start sum: {time.asctime()}')
        score_sumed = scaler.scale_sum(batch_bert_score, batch_tfidf_score, scaler = args.method)
        print(f'end sum: {time.asctime()}')
        print(f'start argsort: {time.asctime()}')
        batch_candidates_index = scaler.batch_argsort(score_sumed)
        del score_sumed
        gc.collect()
        print(f'end argsort: {time.asctime()}')
        batch_labels , batch_bert_candidates_scores, batch_tfidf_candidates_scores = predict_labels(querycui[start:end], dictcui, batch_candidates_index, batch_bert_score, batch_tfidf_score)
        labels.append(batch_labels)
        bert_candidates_scores.append(batch_bert_candidates_scores)
        tfidf_candidates_scores.append(batch_tfidf_candidates_scores)
        del batch_bert_score,batch_tfidf_score
        gc.collect()
labels = np.concatenate(labels, axis = 0)
bert_candidates_scores = np.concatenate(bert_candidates_scores, axis = 0 )
tfidf_candidates_scores = np.concatenate(tfidf_candidates_scores, axis =0)
print(labels.shape, bert_candidates_scores.shape, tfidf_candidates_scores.shape)

with open(f'{args.checkpoint}_{args.query}_{args.dictionary}_train_sample.joblib','wb') as f:
    joblib.dump(
        {
            "bert_scores" : bert_candidates_scores,
            "tfidf_scores" : tfidf_candidates_scores,
            "labels":labels
    },f
    )
