from transformers import AutoTokenizer
import numpy as np

class Tokenizer():
    
    def __init__(self,checkpoint = "sapbert"):
        if checkpoint == 'biobert':
            self.tokenizer = AutoTokenizer.from_pretrained(r'/Share/home/qiyifan/filebase/source/biobert-base-cased-v1.1',use_fast=True)
        elif checkpoint == 'sapbert':
            self.tokenizer = AutoTokenizer.from_pretrained(r'/Share/home/qiyifan/filebase/source/SapBERT-UMLS-2020AB-all-lang-from-XLMR',use_fast=True)
        elif checkpoint == 'multilingual':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased',use_fast=True)
        elif checkpoint == 'biosyn':
            self.tokenizer = AutoTokenizer.from_pretrained(r'/Share/home/qiyifan/filebase/source/biobert-base-cased-v1.1',use_fast=True)
        elif checkpoint == 'coder':
            self.tokenizer = AutoTokenizer.from_pretrained(r"/Share/home/qiyifan/filebase/source/coder_all",use_fast=True)
        elif checkpoint == 'bert':
            self.tokenizer = AutoTokenizer.from_pretrained(r'/Share/home/qiyifan/filebase/source/bert-base-uncase',use_fast=True)
        elif checkpoint == 'roberta':
            self.tokenizer = AutoTokenizer.from_pretrained(r'/Share/home/qiyifan/filebase/source/xlm-roberta-base',use_fast=True)
        elif checkpoint == 'umlsbert':
            self.tokenizer = AutoTokenizer.from_pretrained(r'/Share/home/qiyifan/filebase/source/umlsbert',use_fast=True)
        elif checkpoint == 'xlmsapbert':
            self.tokenizer = AutoTokenizer.from_pretrained(r'/Share/home/qiyifan/filebase/source/xlm-umlsboost-sapbert',use_fast=True)
        elif checkpoint == 'bertse':
            self.tokenizer = AutoTokenizer.from_pretrained(r'/Share/home/qiyifan/filebase/projects/normalization/marginalbert_umls/tmp/checkpoint_2',use_fast=True)

    def tokenize(self, batch):
        tokens = self.tokenizer.batch_encode_plus(batch,
                                            max_length=42,
                                            add_special_tokens=True,
                                            padding="max_length",
                                            return_attention_mask=True,
                                            return_tensors='pt',
                                            truncation=True)
        return tokens


