import subprocess


berts = ['sapbert','biobert','bert']
queries = ['chpo_query','icd10_query','realworld_query']
# dictionaries = ['disorder_dictionary']
translates = ['baidu','youdao','tencent']
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

# for dictionary in dictionaries:
#     print(dictionary)
#     for bert in berts:
#         print(bert)
#         command = [
#             'python',
#             'get_dictionary_embedding.py',
#             '-d',
#             dictionary,
#             '--checkpoint', 
#             bert,
#             '--load_pretrained',
#             'False',
#             '--cuda', 
#             'True'
#         ]
#         subprocess.call(command)

