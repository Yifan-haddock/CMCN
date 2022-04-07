# -*- coding: utf-8 -*-
# @Date    : 2021-02-26 10:30:09
# @Author  : lizong
# @Version : $Id$

import requests
import json
from nltk.stem import WordNetLemmatizer


def search_umls(candidate: str, index = 'hpo', lang: str = 'en', field: str = 'cui', num = 50, elastic_size = 50) -> dict:
    """CN - EN UMLS search API based on Elasticsearch

    Args:
        candidate (str): Candidate term waiting for search
        lang (str, optional): search language, currentlly support 'en' and 'cn'. Defaults to 'en'.
        field (str, optional): search field, currentlly support 'str', 'cui'. Defaults to 'str'.
        num (int, optional): Top N recommendation. Defaults to 5.
        elastic_size (int, optional): Number of return with elasticsearch. Defaults to 100.

    Returns:
        dict: Search Results
    """
    lemmatizer = WordNetLemmatizer()
    # print(pos_tag(word_tokenize(candidate)))
    if lang == "en":
        lemm_candidate = ' '.join([lemmatizer.lemmatize(i)
                                   for i in candidate.split(" ")])
        if index == 'disorder_dictionary':
            search_url = "http://192.168.1.35:9200/umls/planet2020ab/_search"
        elif index == 'hpo_dictionary':
            search_url = 'http://192.168.14.1:9200/umlshpo/hpoindex/_search'
        else:
            search_url = 'http://192.168.14.1:9200/umlsicd10/icd10index/_search'
    else:
        lemm_candidate = candidate
        search_url = "http://192.168.1.35:9200/cnumls/planet2017ab/_search"
    data = {
        "query": {
            "match": {
                field: lemm_candidate
            }
        },
        "size": elastic_size
    }
    headers = {'Content-Type': 'application/json'}

    # 调用接口
    r = requests.post(
        search_url,
        json=data,
        headers=headers
    )

    # de redundent
    cui_dict: dict[list[dict[str:str]]] = {}
    res_list = json.loads(r.text)['hits']['hits']
    for res in res_list:
        if res['_source']['cui'] in cui_dict:
            cui_dict[res['_source']['cui']].append(res)
        else:
            if len(cui_dict.keys()) < num:
                cui_dict[res['_source']['cui']] = [res]
            else:
                break

    return res_list


if __name__ == "__main__":
    print(json.dumps(search_umls("renal cyst", num=50), indent=2))
