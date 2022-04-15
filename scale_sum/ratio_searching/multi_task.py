import subprocess
import re
import json

## create dataset
n_samples_epoch = [
    # [100,10001],
    # [500,3001],
    # [1000,2001],
    [2500,2001],
    # [5000,501],
    # [10000,501],
    # [15000,501],
    # [20000,501],
]

times = ["1",'2','3','4','5','6','7','8','9','10']

for i in n_samples_epoch:
    n_sample, epoch = i
    print(i)
    for time in times:
        print(time)
        command = [
            'python', 'train.py', 
            '-n', str(n_sample), 
            '--pretrained', 'biosyn',
            '--epoch', str(epoch), 
            '--times' , time
        ]
        subprocess.call(command)