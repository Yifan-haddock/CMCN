import subprocess

methods = ['weighted_sum']
berts = ["sapbert", "biosyn"]
queries = ['realworld_query']
ratios = [(32.36,4.13),(60.43, 4.12)]
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