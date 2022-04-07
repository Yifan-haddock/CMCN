import pandas as pd
import numpy as np
import joblib
import argparse
from bert_recom import Bert_Re
from ast import literal_eval

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint')
parser.add_argument('--cuda')
parser.add_argument('--load_pretrained')
parser.add_argument('-q','--query')
parser.add_argument('-t', '--translate')

args = parser.parse_args()

bert_recommend = Bert_Re(use_cuda=args.cuda, load_pretrained=args.load_pretrained, checkpoint = args.checkpoint)

path = f'../data/query/{args.query}.csv'
dictionary = list(pd.read_csv(path)[args.translate])
dictionary_token = bert_recommend.tokenize(dictionary)
dictionary_semantic_embedding = bert_recommend.embed_dense(dictionary_token)

path = f'bert_semantic_matrix/{args.checkpoint}_{args.query}_{args.translate}.pk'
with open(path,'wb') as f:
    joblib.dump(dictionary_semantic_embedding,f)
