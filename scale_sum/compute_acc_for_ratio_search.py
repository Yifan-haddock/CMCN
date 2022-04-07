import pandas as pd
import numpy as np
import argparse 
import joblib
from scaler import Scaler
import time
import gc
from tqdm import tqdm

def predict_labels(querycui, candidates, candidates_index, scores):
    candidates = np.array(candidates).reshape(1,-1)
    candidates_cui = np.take_along_axis(candidates, candidates_index, axis = -1)
    candidates_scores = np.take_along_axis(scores, candidates_index, axis = -1)
    labels = np.array([[1 if len(set(q.split('|')).intersection((x.split('|')))) > 0 else 0 for x in c] for q, c in zip(querycui, candidates_cui)])
    return labels, candidates_cui, candidates_scores

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
parser.add_argument('--ratio')

args = parser.parse_args()

## instantiate scaler
scaler = Scaler()

## load matrices
tfidf_dictionary_matrix_path = f'../tfidf/dictmatrix/ngram13/{args.dictionary}.pk'
print(f'reading tfidf matrix: {time.asctime()}')
with open(tfidf_dictionary_matrix_path,'rb') as f:
    tfidf_dictionary_matrix = joblib.load(f).astype(np.float32)
print(f'reading bert matrix: {time.asctime()}')
bert_dictionary_matrix_path = f'../bert/bert_semantic_matrix/{args.checkpoint}_{args.dictionary}.pk'
with open(bert_dictionary_matrix_path,'rb') as f:
    bert_dictionary_matrix = joblib.load(f).astype(np.float32)

## load_dictioanry
dictionary_path = f'../data/dictionary/{args.dictionary}.csv'
query_path = f'../data/query/{args.query}.csv'
query_df = pd.read_csv(query_path)
querycui = list(query_df['cui'])
dictionary_df = pd.read_csv(dictionary_path)
dictcui = list(dictionary_df['cui'])

## single translate
accs = []
scores_multi = []
candidates_cui_multi = []
translates = ['baidu','youdao','tencent']

for translate in translates:
    tfidf_query_matrix_path = f'../tfidf/dictmatrix/ngram13/{args.query}_{args.dictionary}_{translate}.pk'
    print(f'reading tfidf query matrix: {time.asctime()}')
    with open(tfidf_query_matrix_path , 'rb') as f:
        tfidf_query_matrix = joblib.load(f)
    print(f'transforming tfidf matrix: {time.asctime()}')
    tfidf_query_matrix = tfidf_query_matrix.astype(np.float32)
    bert_query_matrix_path = f'../bert/bert_semantic_matrix/{args.checkpoint}_{args.query}_{translate}.pk'
    with open(bert_query_matrix_path,'rb') as f:
        bert_query_matrix = joblib.load(f).astype(np.float32)
    labels = []
    candidates_scores = []
    candidates_cui = []
    print(f'start bert multiply: {time.asctime()}')
    bert_score = scaler.cos_product(bert_query_matrix, bert_dictionary_matrix)
    print(f'end bert multiply: {time.asctime()}')
    del bert_query_matrix
    gc.collect()
    print(f'start tfidf multiply: {time.asctime()}')
    tfidf_score = scaler.cos_product_sparse(tfidf_query_matrix, tfidf_dictionary_matrix).toarray()
    print(f'end tfidf multiply: {time.asctime()}')
    del tfidf_query_matrix
    gc.collect()
    print(f'start sum: {time.asctime()}')
    score_sumed = scaler.scale_sum(bert_score, tfidf_score, scaler = args.method, ratio= args.ratio)
    del bert_score, tfidf_score
    gc.collect()
    print(f'end sum: {time.asctime()}')
    print(f'start argsort: {time.asctime()}')
    batch_candidates_index = scaler.batch_argsort(score_sumed)
    print(f'end argsort: {time.asctime()}')
    labels , candidates_cui, candidates_scores = predict_labels(querycui, dictcui, batch_candidates_index, score_sumed)
    del score_sumed
    gc.collect()
    print(labels.shape, candidates_scores.shape, candidates_cui.shape)
    acc_single = accuracy(labels)
    accs.append(acc_single)
    scores_multi.append(candidates_scores)
    candidates_cui_multi.append(candidates_cui)

print('multi start')
scores_multi = np.concatenate(scores_multi, axis=-1)
candidates_cui_multi = np.concatenate(candidates_cui_multi, axis = -1)
labels_multi, candidates_cui_multi = predict_labels_multi(querycui, candidates_cui_multi, scores_multi, topk=10)
acc_multi = accuracy(labels_multi)
accs.append(acc_multi)

method = []
for translate in translates:
    method.append(f'{args.checkpoint}_{args.method}_{args.query}_{args.dictionary}_{translate}')
method.append(f'{args.checkpoint}_{args.method}_{args.query}_{args.dictionary}_multi')

acc_dict = {
    'method':method,
    'acc@1':[a[0] for a in accs],
    'acc@5':[a[1] for a in accs],
    'acc@10':[a[2] for a in accs]
}
acc_table = pd.DataFrame.from_dict(acc_dict)
acc_table.to_csv(f'ratio_searching/{args.checkpoint}_{args.method}_{args.query}_{args.dictionary}_acctable.csv',index=False)

