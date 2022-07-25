import subprocess


berts = ['xlmsapbert']
# queries = ['chpo_query','icd10_query','realworld_query']
dictionaries = ['disorder_dictionary']
translates = ['baidu','youdao','tencent']

# for query in queries:
#     print(query)
#     for bert in berts:
#         print(bert)
#         for translate in translates:
#             print(translate)
#             command = [
#                 'python',
#                 'get_query_embedding.py',
#                 '-q',
#                 query,
#                 '--checkpoint', 
#                 bert,
#                 '--load_pretrained',
#                 'False',
#                 '-t', 
#                 translate,
#                 '--cuda', 
#                 'True'
#             ]
#             subprocess.call(command)

for dictionary in dictionaries:
    print(dictionary)
    for bert in berts:
        print(bert)
        command = [
            'python',
            'get_dictionary_embedding.py',
            '-d',
            dictionary,
            '--checkpoint', 
            bert,
            '--load_pretrained',
            'False',
            '--cuda', 
            'True'
        ]
        subprocess.call(command)

