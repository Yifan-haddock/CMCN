import subprocess


queries = ['realworld_query']
ngrams = ["ngram12", "ngram13"]
for query in queries:
    print(query)
    for ngram in ngrams:
        print(ngram)
        command = [
            'python', 'acc_compute.py',
            '-f', 
            'realworld_query',
            '--ngram', ngram,
            '-dm', "disorder_dictionary",
            '-d', "disorder_dictionary",
            '--topk', '10'
        ]
        subprocess.call(command)

for ngram in ngrams:
    print(ngram)
    command = [
        'python', 'transform_query.py',
        '-q', 'realworld_query',
        '--ngram', ngram,
        '-d', "disorder_dictionary"
    ]
    subprocess.call(command)