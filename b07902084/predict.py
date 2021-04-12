from tqdm.auto import tqdm
from statistics import mean
import numpy as np
from operator import itemgetter

def predict_query(sparse_query, corpus, configs):
    target = configs["target"]
    sparse_matrix = corpus["sparse"]
    id_to_fname = corpus["id_to_fname"]
    dot_product = sparse_matrix.dot(sparse_query.transpose()).toarray()
    rank = sorted(range(len(dot_product)), key=lambda k: dot_product[k], reverse=True)
    return rank

def process_predictions(query_responses, configs, corpus):
    id_to_fname = corpus["id_to_fname"]
    for j in range(len(query_responses)):
        res = query_responses[j]
        for i in range(configs["target"]):
            res[i] = id_to_fname[res[i]].lower().split("/")[-1]
        
        query_responses[j] = res[:configs["target"]]
    
def write_predictions(query_responses, queries, configs):
    with open(configs["output_path"], 'w') as f:
        f.write('query_id,retrieved_docs\n')
        for i in range(len(query_responses)):
            f.write('{},{}\n'.format(queries[i]["id"][-3:], ' '.join(query_responses[i])))


def calc_MAP(query_responses, configs):
    answer = []
    with open("../queries/ans_train.csv") as f:
        for line in f:
            line = line.split(",")
            if line[0] == "query_id":
                continue
            docs = line[1].split(" ")
            answer.append(docs)
    MAP = []
    for i in range(len(answer)):
        guess = query_responses[i]
        ans = answer[i]
        precision = []
        for j in range(1, configs["target"]):
            precision.append( len(list(set(guess[:j]).intersection(ans[:j]))) / len(ans[:j]) )
        MAP.append(mean(precision))
    # print(MAP)
    # print(mean(MAP))