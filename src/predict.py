from tqdm.auto import tqdm
from statistics import mean

def predict_query(query, corpus, configs):
    target = configs["target"]
    inverted_files = corpus["inverted_files"]
    id_to_fname = corpus["id_to_fname"]
    doc_cosine = {}
    for word in (query["words"]):
        query_val = query["words"][word]
        # print(query_val)
        for doc_id in inverted_files[word]["docs"]:
            doc_freq = inverted_files[word]["docs"][doc_id]
            if doc_id in doc_cosine:
                doc_cosine[doc_id] += query_val * doc_freq
            else:
                doc_cosine[doc_id] = query_val * doc_freq
    # print(doc_cosine)
    for doc_id in doc_cosine:
        # one of cdn, ctc, cte, cts
        category = id_to_fname[doc_id].split("/")[1]
        doc_weight = configs[category]
        # print(doc_weight)
        if configs["use_cosine"]:
            doc_cosine[doc_id] /= corpus["id_to_magnitude"][doc_id]
        doc_cosine[doc_id] *= doc_weight
    id_to_fname = corpus["id_to_fname"]
    response = []
    response = [id_to_fname[doc_id] for doc_id in sorted(doc_cosine, key=doc_cosine.get, reverse=True)]
    # for doc_id in sorted(doc_cosine, key=doc_cosine.get, reverse=True):
    #     response.append(corpus["id_to_fname"][doc_id])
        # print(doc_id, doc_cosine[doc_id])
    return response

def process_predictions(query_responses, configs):
    for j in range(len(query_responses)):
        res = query_responses[j]
        for i in range(configs["target"]):
            res[i] = res[i].lower().split("/")[-1]
        
        query_responses[j] = res[:configs["target"]]
    

def write_predictions(query_responses, queries):
    with open('../prediction.csv', 'w') as f:
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
            # print(guess[:j])
            # print(ans[:j])
            precision.append( len(list(set(guess[:j]).intersection(ans[:j]))) / len(ans[:j]) )
        MAP.append(mean(precision))
    print(MAP)
    print(mean(MAP))
    # print(query_responses)
