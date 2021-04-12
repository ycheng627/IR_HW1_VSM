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
    print(MAP)
    print(mean(MAP))
    # print(query_responses)


    # indices, dot_product_sorted = zip(*sorted(enumerate(dot_product), key=itemgetter(1), reverse=True))
    # print(indices[:100])
    # print(dot_product_sorted[:100])
    
    # print(list(sorted(with_index, key=lambda k: k[1], reverse=True)))
    # print()
    # print( np.argsort(dot_product) )
    # print(dot_product)



    # for word in (query["words"]):
    #     query_val = query["words"][word]
    #     # print(query_val)
    #     for doc_id in inverted_files[word]["docs"]:
    #         doc_freq = inverted_files[word]["docs"][doc_id]
    #         if doc_id in doc_cosine:
    #             doc_cosine[doc_id] += query_val * doc_freq
    #         else:
    #             doc_cosine[doc_id] = query_val * doc_freq
    # # print(doc_cosine)
    # for doc_id in doc_cosine:
    #     # one of cdn, ctc, cte, cts
    #     category = id_to_fname[doc_id].split("/")[1]
    #     doc_weight = configs[category]
    #     # print(doc_weight)
    #     if configs["use_cosine"]:
    #         doc_cosine[doc_id] /= corpus["id_to_magnitude"][doc_id]
    #     doc_cosine[doc_id] *= doc_weight
    # id_to_fname = corpus["id_to_fname"]
    # response = []
    # response = [id_to_fname[doc_id] for doc_id in sorted(doc_cosine, key=doc_cosine.get, reverse=True)]
    # # for doc_id in sorted(doc_cosine, key=doc_cosine.get, reverse=True):
    # #     response.append(corpus["id_to_fname"][doc_id])
    #     # print(doc_id, doc_cosine[doc_id])
    # return response