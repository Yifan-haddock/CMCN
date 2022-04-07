import pandas as pd
import numpy as np
import argparse 
import joblib
from bert_recom import Bert_Re
import time


parser = argparse.ArgumentParser()
parser.add_argument('-q','--query')
parser.add_argument('-d','--dictionary')
parser.add_argument('--checkpoint')
parser.add_argument('--load_pretrained')
parser.add_argument('--method')
parser.add_argument('--topk')

args = parser.parse_args()

query_path = f'../data/query/{args.query}.csv'
query_df = pd.read_csv(query_path)
querycui = list(query_df['cui'])

dictionary_path = f'../data/dictionary/{args.dictionary}.csv'
dictionary_df = pd.read_csv(dictionary_path)
dictcui = list(dictionary_df['cui'])

bert_recommend = Bert_Re(load_pretrained = args.load_pretrained, checkpoint= args.checkpoint)

dictionary_matrix_path = f'bert_semantic_matrix/{args.checkpoint}_{args.dictionary}.pk'
with open(dictionary_matrix_path,'rb') as f:
    dictionary_matrix = joblib.load(f)

accs = []
scores_multi = []
candidates_cui_multi = []

print('single start')
translates = ['baidu','youdao','tencent']
for translate in translates:
    query_matrix_path = f"bert_semantic_matrix/{args.checkpoint}_{args.query}_{translate}.pk"
    with open(query_matrix_path,'rb') as f:
        query_matrix = joblib.load(f)
    print(f'start multiply: {time.asctime()}')
    if args.method == 'cos':
        scores = bert_recommend.cos_product(query_matrix, dictionary_matrix)
    else:
        scores = bert_recommend.inner_product(query_matrix, dictionary_matrix)
    print(f'end multiply: {time.asctime()}')
    print(f'start argsort: {time.asctime()}')
    candidates_index = bert_recommend.batch_argsort(scores)
    print(f'end argsort: {time.asctime()}')
    labels , candidates_cui, candidates_scores = bert_recommend.predict_labels(querycui, dictcui, candidates_index, scores)
    acc_single = bert_recommend.accuracy(labels)
    print(acc_single)
    accs.append(acc_single)
    scores_multi.append(candidates_scores)
    candidates_cui_multi.append(candidates_cui)


print('multi start')
scores_multi = np.concatenate(scores_multi, axis=-1)
candidates_cui_multi = np.concatenate(candidates_cui_multi, axis = -1)
labels_multi, candidates_cui_multi = bert_recommend.predict_labels_multi(querycui, candidates_cui_multi, scores_multi, topk=10)
acc_multi = bert_recommend.accuracy(labels_multi)
accs.append(acc_multi)

## testifying
with open(f'candidates_cui_{args.checkpoint}_{args.method}_{args.query}_{args.dictionary}.joblib','wb') as f:
    joblib.dump(
        {
            "labels" : labels_multi,
            'candidates' : candidates_cui_multi
        },f
    )

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
acc_table.to_csv(f'{args.checkpoint}_{args.method}_{args.query}_{args.dictionary}_acctable.csv',index=False)
