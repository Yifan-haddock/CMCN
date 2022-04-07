from scipy.sparse import csr
from sparseencoder import SparseEncoder
import pandas as pd
import argparse
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dictionary')
parser.add_argument('-q','--query')
parser.add_argument('--ngram')

args = parser.parse_args()
query_path = f'../data/query/{args.query}.csv'

tfidf = SparseEncoder(args.ngram, args.dictionary, load_pretrained=True)
translates = ['baidu','tencent','youdao','en']
for translate in translates:
    corpus = list(pd.read_csv(query_path)[translate].values.astype('U'))
    csr_matrix = tfidf.transform(corpus)
    with open(f'dictmatrix/{args.ngram}/{args.query}_{args.dictionary}_{translate}.pk','wb') as f:
        joblib.dump(csr_matrix, f)