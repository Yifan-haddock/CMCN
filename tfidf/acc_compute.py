from sparseencoder import SparseEncoder
import pandas as pd
import joblib
import argparse
import numpy as np
import gc
import time
from joblib import Parallel, delayed

def matrix_multiply(qmatrix, dmatrix):
    qmatrix = qmatrix.astype(np.float32).toarray()
    dmatrix = dmatrix.astype(np.float32).toarray()
    score = qmatrix.dot(dmatrix.transpose())
    return score

def index_argsort(matrix, topk=10):
    index = np.argsort(matrix)[:,-topk:]
    reverse_index = np.array([i[::-1] for i in index],dtype= np.int32)
    return reverse_index
    
def batch_argsort(scores):
    batch_size=1024
    iterations = range(0, scores.shape[0],batch_size)
    matrix_batches = [scores[start:min(start+batch_size,scores.shape[0])] for start in iterations]
    candidates_index = Parallel(n_jobs=-1)(delayed(index_argsort)(i) for i in matrix_batches)
    candidates_index = np.concatenate(candidates_index, axis=0).squeeze()
    return candidates_index

def predict_labels(querycui, dictcui, scores, topk=10):
    topk = int(topk)
    print(f'argsort start time{time.asctime()}')
    # index = np.argsort(scores, axis=-1)[:, -topk:]
    candidates_index = batch_argsort(scores)
    print(f'argsort finish time{time.asctime()}')
    candidates_cui = [[dictcui[i] for i in candidates] for candidates in candidates_index]
    candidates_score = np.take_along_axis(scores, candidates_index ,axis = -1)
    labels = np.array([[1 if len(set(q.split('|')).intersection((x.split('|')))) > 0 else 0 for x in c] for q, c in zip(querycui, candidates_cui)])
    return labels, np.array(candidates_cui), candidates_score

def predict_labels_multi(querycui, dictcui, multi_scores, topk=10):
    topk = int(topk)
    index = multi_scores.argsort()[:, -topk:]
    candidates_index = np.array([i[::-1] for i in index],dtype= np.int32)
    candidates_cui = np.take_along_axis(dictcui, candidates_index, axis = -1)
    labels = np.array([[1 if len(set(q.split('|')).intersection((x.split('|')))) > 0 else 0 for x in c] for q, c in zip(querycui, candidates_cui)])
    return labels

def accuracy(labels,topk = 10):
    accs = []
    for i in [1,5,int(topk)]:
        top = round(np.array([1 for i in labels[: , :i] if i.sum() > 0], dtype=np.int32).sum() / labels.shape[0] * 100,2)
        accs.append(top)
    return accs


parser = argparse.ArgumentParser()
parser.add_argument('-f','--file')
parser.add_argument('--ngram')
parser.add_argument('-dm','--dictionary_matrix')
parser.add_argument('-d','--dictionary')
parser.add_argument('--topk')

args = parser.parse_args()

sparseencoder = SparseEncoder(args.ngram, args.dictionary_matrix,load_pretrained=True)

query_path = f'../data/query/{args.file}.csv'
dictionary_path = f'../data/dictionary/{args.dictionary}.csv'
query_df = pd.read_csv(query_path)
dictionary_df = pd.read_csv(dictionary_path)
querycui = list(query_df['cui'])
dictcui = list(dictionary_df['cui'])

with open(f'dictmatrix/{args.ngram}/{args.dictionary_matrix}.pk','rb') as f:
    dmatrix = joblib.load(f)

scores_multi = []
candidates_cui_multi = []
accs = []
## single
print('single start')
translates = ['baidu','youdao','tencent']
for translate in translates:
    qmatrix = sparseencoder.transform(list(query_df[translate]))
    print(f'start time: {time.asctime()}')
    scores_single = matrix_multiply(qmatrix, dmatrix)
    print(f'end multiply time: {time.asctime()}')
    labels_single, candidates_cui_single, candidates_score_single = predict_labels(querycui, dictcui, scores_single, topk=args.topk)
    acc_single = accuracy(labels_single, topk=args.topk)
    accs.append(acc_single)
    scores_multi.append(candidates_score_single)
    candidates_cui_multi.append(candidates_cui_single)
    del acc_single
    del scores_single
    gc.collect()

## multi
print('multi start')
scores_multi = np.concatenate(scores_multi, axis=-1)
candidates_cui_multi = np.concatenate(candidates_cui_multi, axis = -1)
labels_multi = predict_labels_multi(querycui, candidates_cui_multi, scores_multi, topk=args.topk)
acc_multi = accuracy(labels_multi,topk=args.topk)
accs.append(acc_multi)


## write acc into csv
method = []
for translate in translates:
    method.append(f'{args.ngram}_{args.file}_{args.dictionary}_{translate}')
method.append(f'{args.ngram}_{args.file}_{args.dictionary}_multi')

acc_dict = {
    'method':method,
    'acc@1':[a[0] for a in accs],
    'acc@5':[a[1] for a in accs],
    f'acc@{args.topk}':[a[2] for a in accs]
}
acc_table = pd.DataFrame.from_dict(acc_dict)
acc_table.to_csv(f'tfidf_{args.file}_{args.dictionary}_{args.ngram}_acctable.csv',index=False)