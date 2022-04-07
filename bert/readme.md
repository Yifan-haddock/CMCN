## bert_recom.py
Before use, please make bert checkpoint stay the same in bertmodel and tokenizer.

Functions:
1. Batch input, Bert semantic embedding.
2. Batch and multiprocessing argsort.
3. Matrix cos_similarity and computation.

### get_dictionary_embedding.py
```bash
python get_dictionary_embedding.py \
        --checkpoint biosyn \
        --cuda True \
        --load_pretrained bert_pretrained/biosyn/t3epoch105_global_pretrained_bio_bert_pretrained.bin \
        -d chpo_dictionary
```

### get_query_embedding.py
add tranlate argument, options contains baidu, tencent, youdao, cn
```bash
python get_query_embedding.py \
        --checkpoint biosyn \
        --cuda True \
        --load_pretrained bert_pretrained/biosyn/t3epoch105_global_pretrained_bio_bert_pretrained.bin \
        -q chpo_query \
        -t baidu
```

### compute_acc.py
Functions:
1. compute cosine scores, predict topk candidates.
2. compute acc

ATTENTION: translates arguments are not specified, it contains three translates and multi translates.

```bash
python compute_acc.py \
        -q realworld_query \
        -d disorder_dictionary \
        --checkpoint biosyn \
        --load_pretrained bert_pretrained/biosyn/t3epoch105_global_pretrained_bio_bert_pretrained.bin \
        --method cos
```

### compute_acc_cn.py
Functions:
1. compute cosine scores, predict topk candidates.
2. compute acc for cn

```bash
python compute_acc_cn.py \
        -q chip_query \
        -d icd10_dictionary \
        --checkpoint multilingual \
        --load_pretrained False \
        --method cos
```