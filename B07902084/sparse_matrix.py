import numpy as np
from scipy.sparse import csr_matrix
from tqdm.auto import tqdm

def get_doc_norm_tf(freq, max_freq, dl, avdl, k, b):
    '''
    Need both a doc length and a k normalization. Use formula
    '''
    x = freq
    BM25 = (( k + 1 ) * x ) / (x + k)
    doc_len_norm = (1 - b + (b * dl / avdl) ) 
    Okapi = (( k + 1 ) * x ) / (x + k * doc_len_norm)
    # return BM25 / doc_len_norm
    return Okapi

def get_query_norm_tf(freq, max_freq, dl, avdl, ka):
    '''
    Need both a doc length and a k normalization. Use formula
    '''
    x = freq
    BM25 = (( ka + 1 ) * x ) / (x + ka)
    # return BM25 / doc_len_norm
    return BM25


def gen_matrix(corpus, configs):
    '''
    Given an input of inverted files, generate a sparse matrix representation
    A 2d array, with one dimension of (bigram1, bigram2), and the second dimension 
    of normalized TF * IDF
    '''
    inverted_files = corpus["inverted_files"]
    gram_to_id = corpus["gram_to_id"]
    avdl = corpus["avdl"]
    id_to_doclen = corpus["id_to_doclen"]
    k = configs["k"]
    b = configs["b"]
    gram_id_list = []
    doc_id_list = []
    freq_list = []
    for gram_id in (inverted_files):
        IDF = inverted_files[gram_id]["IDF"]
        max_freq = inverted_files[gram_id]["max_freq"]
        for doc_id in inverted_files[gram_id]["docs"]:
            dl = id_to_doclen[doc_id]
            doc_freq = inverted_files[gram_id]["docs"][doc_id]
            gram_id_list.append(gram_id)
            doc_id_list.append(doc_id)
            freq_list.append(get_doc_norm_tf(doc_freq, max_freq, dl, avdl, k, b) * IDF)
    gram_id_np = np.array(gram_id_list)
    doc_id_np = np.array(doc_id_list)
    freq_np = np.array(freq_list)
    sparse = csr_matrix((freq_np, (doc_id_np, gram_id_np)), shape=(configs["doc_count"], configs["gram_count"]))
    return sparse

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


def gen_query_vector(query, corpus, configs):
    '''
    Given an input of query vector, normalize through TF IDF
    '''
    inverted_files = corpus["inverted_files"]
    avdl = corpus["avdl"]
    id_to_doclen = corpus["id_to_doclen"]
    ka = configs["ka"]
    b = configs["b"]
    gram_id_list = []
    doc_id_list = []
    freq_list = []
    for gram_id in query["words"]:
        IDF = inverted_files[gram_id]["IDF"]
        max_freq = inverted_files[gram_id]["max_freq"]
        dl = query["dl"]
        freq = query["words"][gram_id]
        gram_id_list.append(gram_id)
        freq_list.append(get_query_norm_tf(freq, max_freq, dl, avdl, ka) * IDF)
    doc_id_list = [0 for i in range(len(gram_id_list))]

    gram_id_np = np.array(gram_id_list)
    doc_id_np = np.array(doc_id_list)
    freq_np = np.array(freq_list)
    sparse_query = csr_matrix((freq_np, (doc_id_np, gram_id_np)), shape=(1, configs["gram_count"]))
    return sparse_query

def gen_query_to_magnitude(queries):
    query_to_magnitude = dict.fromkeys(range(len(queries), 0))
    for i in range(len(queries)):
        s = 0
        for word in queries[i]["words"]:
            s += pow(queries[i]["words"][word], 2)

    for i in range(len(queries)):
        m = sum(pow(queries[i]["words"][word], 2) for word in queries[i]["words"])
        m = pow(m, 1/2)
        queries[i]["magnitude"] = m
