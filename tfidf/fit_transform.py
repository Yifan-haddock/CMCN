from scipy.sparse import csr
from sparseencoder import SparseEncoder
import pandas as pd
import argparse
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file')
parser.add_argument('--ngram')
parser.add_argument('-p','--path')

args = parser.parse_args()
path = f'../data/dictionary/{args.file}.csv'

corpus = list(pd.read_csv(path)['terms'])

tfidf = SparseEncoder(args.ngram, args.path, load_pretrained=True)
tfidf.fit(corpus)
name = args.file.split('.')[0]
tfidf.save_pretrained(f'pretrained/{args.ngram}/{name}.pk')

csr_matrix = tfidf.transform(corpus)
with open(f'dictmatrix/{args.ngram}/{name}.pk','wb') as f:
    joblib.dump(csr_matrix, f)
