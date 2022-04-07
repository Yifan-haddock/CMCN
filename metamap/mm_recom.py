import os
import pandas as pd
import numpy as np
import re
import argparse
from tqdm import tqdm

## choice of indexes: hpo_dictionary, icd10_dictionary, disorder_dictionary
def metamap_recommend(term, dictionary, topk=10):
    semantic_type = 'acab,anab,comd,cgab,dsyn,emod,fndg,inpo,mobd,neop,patf,sosy'
    if dictionary == 'disorder_dictionary':
        var = os.popen("echo " + term + " | " + f'./public_mm/bin/metamap -I -y -Z 2020AB -V Base -c -b -y -J {semantic_type}').read()
    elif dictionary == 'icd10_dictionary':
        var = os.popen("echo " + term + " | " + f'./public_mm/bin/metamap -I -y -Z 2020AB -V Base -c -b -y -R ICD10').read()
    else:
        var = os.popen("echo " + term + " | " + f'./public_mm/bin/metamap -I -y -Z 2020AB -V Base -c -b -y -R HPO').read()

    cuis = var.split('\n')
    candidates_cui = []
    scores = []
    for item in cuis[4:]:
        cui = re.search(r'(\d+).*(C\d+)', item)
        if cui != None:
            cui_id = cui.group(2)
            score = cui.group(1).strip(' ')
            candidates_cui.append(cui_id)
            scores.append(float(score))
    if len(scores) < 10:
        scores.extend([0 for i in range(0,10-len(scores))])
        candidates_cui.extend(['null' for i in range(0,10-len(candidates_cui))])
    return candidates_cui[:topk], scores[:topk]

def predict_labels(querycui, candidates, scores, topk=10):
    index = np.argsort(scores, axis=-1)[:, -topk:]
    candidates_index = np.array([i[::-1] for i in index],dtype= np.int32)
    candidates_cui = np.take_along_axis(candidates, candidates_index, axis = -1)
    labels = np.array([[1 if len(set(q.split('|')).intersection((x.split('|')))) > 0 else 0 for x in c] for q, c in zip(querycui, candidates_cui)])
    return labels

def accuracy(labels):
    accs = []
    for i in [1,5,10]:
        top = round(np.array([1 for i in labels[: , :i] if i.sum() > 0], dtype=np.int32).sum() / labels.shape[0] * 100,2)
        accs.append(top)
    return accs

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file')
parser.add_argument('--index')

args = parser.parse_args()

query_path = f'../data/query/{args.file}.csv'
query_df = pd.read_csv(query_path)
querycui = list(query_df['cui'])

translates = ['baidu', 'youdao', 'tencent']
scores_multi = []
candidates_cui_multi = []
accs = []

print('single start')
for translate in translates:
    query_terms = query_df[translate]
    batch_candidates_cui = []
    batch_scores = []
    for q_term in tqdm(query_terms, total = len(query_terms)):
        candidates_cui, scores = metamap_recommend(q_term, args.index)
        batch_candidates_cui.append(candidates_cui)
        batch_scores.append(scores)
    batch_candidates_cui = np.stack(batch_candidates_cui, axis = 0)
    batch_scores = np.stack(batch_scores, axis = 0)
    labels = predict_labels(querycui, batch_candidates_cui, batch_scores)
    acc_single = accuracy(labels)
    accs.append(acc_single)
    scores_multi.append(batch_scores)
    candidates_cui_multi.append(batch_candidates_cui)

print('multi start')
scores_multi = np.concatenate(scores_multi, axis = -1)
candidates_cui_multi = np.concatenate(candidates_cui_multi, axis = -1)
labels_multi = predict_labels(querycui, candidates_cui_multi, scores_multi)
acc_multi = accuracy(labels_multi)
accs.append(acc_multi)

method = []
for translate in translates:
    method.append(f'metamap_{args.file}_{args.index}_{translate}')
method.append(f'metamap_{args.file}_{args.index}_multi')

acc_dict = {
    'method':method,
    'acc@1':[a[0] for a in accs],
    'acc@5':[a[1] for a in accs],
    'acc@10':[a[2] for a in accs]
}
acc_table = pd.DataFrame.from_dict(acc_dict)
acc_table.to_csv(f'metamap_{args.file}_{args.index}_acctable.csv',index=False)