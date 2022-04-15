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
    -n 100 \
    --pretrained biosyn \
    --epoch 10001 \
    --times 9
```