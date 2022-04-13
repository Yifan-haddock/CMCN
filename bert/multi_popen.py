import subprocess


berts = ['sapbert']
queries = ['disorder_query']
translates = ['baidu']
# methods = ['cos','inner']

# for bert in berts:
#     print(bert)
#     for query in queries:
#         print(query)
#         for method in methods:
#             print(method)
#             command = [
#                 'python',
#                 'compute_acc.py',
#                 '-q',
#                 query,
#                 '-d',
#                 'disorder_dictionary',
#                 '--checkpoint',
#                 bert,
#                 '--load_pretrained',
#                 'False',
#                 '--method',
#                 method
#             ]
#             subprocess.call(command)

# for query in queries:
#     print(query)
#     for method in methods:
#         print(method,'multilingual')
#         command = [
#             'python',
#             'compute_acc_cn.py',
#             '-q',
#             query,
#             '-d',
#             'disorder_dictionary',
#             '--checkpoint',
#             'multilingual',
#             '--load_pretrained',
#             'False',
#             '--method',
#             method
#         ]
#         subprocess.call(command)

for query in queries:
    print(query)
    for bert in berts:
        print(bert)
        for translate in translates:
            print(translate)
            command = [
                'python',
                'get_query_embedding.py',
                '-q',
                query,
                '--checkpoint', 
                bert,
                '--load_pretrained',
                'False',
                '-t', 
                translate,
                '--cuda', 
                'True'
            ]
            subprocess.call(command)

for query in queries:
    print(query)
    for translate in translates:
        print(translate)
        command = [
            'python',
            'get_query_embedding.py',
            '-q',
            query,
            '--checkpoint', 
            'biosyn',
            '--load_pretrained',
            'bert_pretrained/biosyn/t3epoch105_global_pretrained_bio_bert_pretrained.bin',
            '-t', 
            translate,
            '--cuda', 
            'True'
        ]
        subprocess.call(command)

# for translate in translates:
#     print(translate,'multilingual')
#     command = [
#             'python',
#             'get_query_embedding.py',
#             '-q',
#             "realworld_query",
#             '--checkpoint', 
#             'multilingual',
#             '--load_pretrained',
#             'False',
#             '-t', 
#             translate,
#             '--cuda', 
#             'True'
#         ]
#     subprocess.call(command)