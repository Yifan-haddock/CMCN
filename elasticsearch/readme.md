### es_recom
stand for elastic_search recommendation.
need to check umls_search prepared before run.
index arguments has three options allowed, hpo_dictionary, icd10_dictionary and disorder_dictionary, please choose accordingly.

```bash
python es_recom.py \
        -f realworld_query \
        --index disorder_dictionary
```