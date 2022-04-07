from transformers import AutoTokenizer
import numpy as np

class Tokenizer():
    
    def __init__(self,checkpoint = "sapbert"):
        if checkpoint == 'biobert':
            self.tokenizer = AutoTokenizer.from_pretrained(r'/Share/home/qiyifan/filebase/source/biobert-base-cased-v1.1')
        elif checkpoint == 'sapbert':
            self.tokenizer = AutoTokenizer.from_pretrained(r'/Share/home/qiyifan/filebase/source/SapBERT-UMLS-2020AB-all-lang-from-XLMR')
        elif checkpoint == 'multilingual':
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
        elif checkpoint == 'biosyn':
            self.tokenizer = AutoTokenizer.from_pretrained(r'/Share/home/qiyifan/filebase/source/biobert-base-cased-v1.1')
        elif checkpoint == 'coder':
            self.tokenizer = AutoTokenizer.from_pretrained(r"GanjinZero/UMLSBert_ALL")
    def tokenize(self, batch):
        tokens = self.tokenizer.batch_encode_plus(batch,
                                            max_length=42,
                                            add_special_tokens=True,
                                            padding="max_length",
                                            return_attention_mask=True,
                                            return_tensors='pt',
                                            truncation=True)
        return tokens


