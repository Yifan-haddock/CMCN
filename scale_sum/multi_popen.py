import subprocess

methods = ['weighted_sum']
berts = ["sapbert", "biosyn"]
queries = ['chpo_query','icd10_query','realworld_query']
ratios = [(33.11,7.28),(75.19, 6.32)]
for query in queries:
    print(query)
    for bert,ratio in zip(berts,ratios):
        print(bert)
        dense_ratio = str(ratio[0])
        sparse_ratio = str(ratio[1])
        for method in methods:
            print(method)
            command = [
                'python',
                'compute_acc.py',
                '-q', 
                query,
                '-d',
                'disorder_dictionary',
                '--checkpoint',
                bert,
                '--method',
                method,
                '--dense_ratio',
                dense_ratio,
                '--sparse_ratio',
                sparse_ratio
            ]
            subprocess.call(command)