import torch
from torch import nn
import numpy
from transformers import AutoTokenizer, AutoModel
import os
from ast import literal_eval

class BertModel(nn.Module):

    def __init__(self, use_cuda = "False", load_pretrained = "False", checkpoint = 'sapbert'):
        super(BertModel, self).__init__()
        if checkpoint == 'sapbert':
            self.model = AutoModel.from_pretrained(r"/Share/home/qiyifan/filebase/source/SapBERT-UMLS-2020AB-all-lang-from-XLMR")
        elif checkpoint == 'biobert':
            self.model = AutoModel.from_pretrained(r"/Share/home/qiyifan/filebase/source/biobert-base-cased-v1.1")
        elif checkpoint == 'multilingual':
            self.model = AutoModel.from_pretrained('bert-base-multilingual-cased')
        elif checkpoint == 'biosyn':
            self.model = AutoModel.from_pretrained(r"/Share/home/qiyifan/filebase/source/biobert-base-cased-v1.1")
        elif checkpoint == 'coder':
            self.model = AutoModel.from_pretrained(r"/Share/home/qiyifan/filebase/source/coder_all")
        elif checkpoint == 'bert':
            self.model = AutoModel.from_pretrained(r'/Share/home/qiyifan/filebase/source/bert-base-uncase')
        elif checkpoint == 'roberta':
            self.model = AutoModel.from_pretrained(r'/Share/home/qiyifan/filebase/source/xlm-roberta-base')
        elif checkpoint == 'umlsbert':
            self.model = AutoModel.from_pretrained(r'/Share/home/qiyifan/filebase/source/umlsbert')
        elif checkpoint == 'xlmsapbert':
            self.model = AutoModel.from_pretrained(r'/Share/home/qiyifan/filebase/source/xlm-umlsboost-sapbert')
        elif checkpoint == 'bertse':
            self.model = AutoModel.from_pretrained(r'/Share/home/qiyifan/filebase/projects/normalization/marginalbert_umls/tmp/checkpoint_2')

        if literal_eval(use_cuda):
            self.use_cuda = True
            self.model = self.model.cuda()
        else:
            self.use_cuda = False

        if load_pretrained != "False":
            self.load_pretrained(load_pretrained) # load pretrained
            
    def load_pretrained(self, path):
        if not self.use_cuda:
            state_dict = torch.load(path,map_location=torch.device('cpu'))
        else:
            state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
    def save_pretrained(self):
        path = self.path
        torch.save(self.model.state_dict(), os.path.join(path, "t_bio_bert_pretrained.bin"))

    def save_global_pretrained(self, epoch, sample_num):
        path = self.path
        torch.save(self.model.state_dict(), os.path.join(path, f"t{epoch}epoch{sample_num}_global_pretrained_bio_bert_pretrained.bin"))

    def forward(self, input_ids, attention_masks):
        """
        @ input: variable of tensor with shape [batch, len(subwords)]
        @ output: variable of tensor with shape [batch, word_embed_dim]
        """
        if self.use_cuda:
            input_ids = input_ids.cuda()
            attention_masks = attention_masks.cuda()
        last_hidden_state = self.model(input_ids ,attention_mask=attention_masks).last_hidden_state
        x = last_hidden_state[:,0]
         # CLS token representation
        return x