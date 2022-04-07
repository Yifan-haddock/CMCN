### acc_compute
need to change pretrained path in sparse encoder before run.

ngram range in parse ngram, choose ngram12 or ngram13

```bash
python acc_compute.py \
        -f realworld_query \
        --ngram ngram13 \
        -dm disorder_dictionary \
        -d disorder_dictionary \
        --topk 10
```

### fit_transform
```bash
python fit_transform.py \
        -f disorder_dictionary \
        --ngram ngram13 \
        -p disorder_dictionary
```

### transform_query (only for special use)
```bash
python transform_query.py \
        -q realworld_query \
        --ngram ngram13 \
        -d disorder_dictionary
```