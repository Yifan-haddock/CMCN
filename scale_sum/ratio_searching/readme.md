# dateset.py
use for nerual network learning alpha ratio.

```bash
python dataset.py \
    -q disorder_query \
    -d disorder_dictionary \
    --checkpoint sapbert \
    --method min_max
```

```bash
python train.py \
    -n 10000 \
    --pretrained biosyn \
    --epoch 501 \
    --times 9
```