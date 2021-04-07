def get_norm_tf(freq, max_freq, dl, avdl, k, b):
    '''
    Need both a doc length and a k normalization. Use formula
    '''
    x = freq
    BM25 = (( k + 1 ) * x ) / (x + k)
    doc_len_norm = (1 - b + (b * dl / avdl) ) 
    Okapi = (( k + 1 ) * x ) / (x + k * doc_len_norm)
    # return BM25 / doc_len_norm
    return Okapi

def gen_matrix(corpus, configs):
    '''
    Given an input of inverted files, generate a sparse matrix representation
    A 2d array, with one dimension of (bigram1, bigram2), and the second dimension 
    of normalized TF * IDF
    '''
    inverted_files = corpus["inverted_files"]
    avdl = corpus["avdl"]
    id_to_doclen = corpus["id_to_doclen"]
    k = configs["k"]
    b = configs["b"]
    for key in inverted_files:
        IDF = inverted_files[key]["IDF"]
        max_freq = inverted_files[key]["max_freq"]
        for doc_id in inverted_files[key]["docs"]:
            dl = id_to_doclen[doc_id]
            doc_freq = inverted_files[key]["docs"][doc_id]
            inverted_files[key]["docs"][doc_id] = get_norm_tf(doc_freq, max_freq, dl, avdl, k, b) * IDF
        # for i in range(len(inverted_files[key]["docs"])):
        #     doc_id = inverted_files[key]["docs"][i][0]
        #     dl = id_to_doclen[doc_id]
        #     freq = inverted_files[key]["docs"][i][1]
        #     inverted_files[key]["docs"][i][1] = get_norm_tf(freq, max_freq, dl, avdl, k, b) * IDF
    return inverted_files

def gen_id_to_magnitude(corpus, configs):
    '''
    Generate the magnitude of each document by sum of squares.
    Needed for Cosine similarity.
    '''
    inverted_files = corpus["inverted_files"]
    id_to_magnitude = dict.fromkeys(corpus["id_to_doclen"],0)
    for key in inverted_files:
        for doc_id in inverted_files[key]["docs"]:
            doc_freq = inverted_files[key]["docs"][doc_id]
            id_to_magnitude[doc_id] += pow(doc_freq, 2)
    id_to_magnitude = {k: pow(v,1/2) for k, v in id_to_magnitude.items()}
    return id_to_magnitude


def gen_query_vector(queries, corpus, configs):
    '''
    Given an input of query vector, normalize through TF IDF
    '''
    inverted_files = corpus["inverted_files"]
    avdl = corpus["avdl"]
    id_to_doclen = corpus["id_to_doclen"]
    k = configs["k"]
    b = configs["b"]
    for query in queries:
        for word in query["words"]:
            IDF = inverted_files[word]["IDF"]
            max_freq = inverted_files[word]["max_freq"]
            dl = query["dl"]
            freq = query["words"][word]
            query["words"][word] = get_norm_tf(freq, max_freq, dl, avdl, k, b) * IDF
    return queries

def gen_query_to_magnitude(queries):
    query_to_magnitude = dict.fromkeys(range(len(queries), 0))
    for i in range(len(queries)):
        s = 0
        for word in queries[i]["words"]:
            # print(queries[i]["words"][word])
            s += pow(queries[i]["words"][word], 2)

    for i in range(len(queries)):
        m = sum(pow(queries[i]["words"][word], 2) for word in queries[i]["words"])
        m = pow(m, 1/2)
        queries[i]["magnitude"] = m
