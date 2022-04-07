## scaler
function:
1. compute argsort using multiprocessing method.
2. scale distribution. three scales method: min_max, zscore, tanh.
3. compute matrix multiply.

## compute_acc.py
method argument has three choices: min_max, zscore, tanh, please choose one from these.

```bash
python compute_acc.py \
        -q chpo_query \
        -d disorder_dictionary \
        --checkpoint sapbert \
        --method weighted_sum
``` 

```bash
python compute_acc_small.py \
        -q chpo_query \
        -d chpo_dictionary \
        --checkpoint sapbert \
        --method sum \
        --dense_ratio 32.0 \
        --sparse_ratio 4
```  

## compute_acc_for_ratio_searching.py
method argument has three choices: min_max, zscore, tanh, please choose one from these.

```bash
python compute_acc_small.py \
        -q chpo_query \
        -d chpo_dictionary \
        --checkpoint sapbert \
        --method weighted_sum \
        --ratio 0.1
``` 

## ratio searching results
realworld icd10 chpo sapbert
28.56, 4.78
realworld icd10 chpo biosyn
53.48 4.68
realworld icd10 chpo biobert
53.90 18.52