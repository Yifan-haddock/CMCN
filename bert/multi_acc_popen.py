import subprocess


berts = ['biobert','bert','sapbert']
queries = ['chpo_query','icd10_query','realworld_query']
dictionaries = ['disorder_dictionary']
translates = ['baidu','youdao','tencent']
methods = ['cos']

for bert in berts:
    print(bert)
    for query in queries:
        print(query)
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
                '--load_pretrained',
                'False',
                '--method',
                method
            ]
            subprocess.call(command)

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
#             'roberta',
#             '--load_pretrained',
#             'False',
#             '--method',
#             method
#         ]
#         subprocess.call(command)

# print(
#     'sapbert real_world'
# )
# command = [
#     'python',
#     'compute_acc_cn.py',
#     '-q',
#     'realworld_query',
#     '-d',
#     'disorder_dictionary',
#     '--checkpoint',
#     'sapbert',
#     '--load_pretrained',
#     'False',
#     '--method',
#     'cos'
# ]
# subprocess.call(command)