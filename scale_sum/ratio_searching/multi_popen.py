import subprocess


berts = ['biobert']
queries = ['chpo_query','icd10_query']
methods = ['min_max']

for bert in berts:
    print(bert)
    for query in queries:
        print(query)
        for method in methods:
            print(method)
            command = [
                'python',
                'dataset.py',
                '-q',
                query,
                '-d',
                'disorder_dictionary',
                '--checkpoint',
                bert,
                '--method',
                method
            ]
            subprocess.call(command)