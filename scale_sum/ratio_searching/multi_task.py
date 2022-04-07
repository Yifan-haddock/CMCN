import subprocess
import re
import json

## create dataset
ratios =[]
for i in range(1,11):
    command = [
        'python',
        'train.py'
    ]
    ret = subprocess.check_output(command).decode('utf-8')
    pat = re.compile(r'\[(.*?)\]')
    ratios.append(pat.findall(ret))

with open('ratios.json','w') as f:
    f.write(json.dumps(ratios, ensure_ascii=False))