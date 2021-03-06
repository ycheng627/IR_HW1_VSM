from scipy.sparse import csr_matrix
import numpy as np

def rocchio_feedback(query_response, query, corpus, configs):
    # Consider the top 50 query responses as relevant
    # Scan through corpus, and add the value to relevant r or nr
    # Use rocchio to generate a new query, where all vocab will be used

    r_num = 50
    nr_num = 10
    relevant_doc_ids = query_response[:r_num]
    nonrelevant_doc_ids = query_response[-nr_num:]
    C_r = r_num
    C_nr = nr_num
    sparse_matrix = corpus["sparse"]
    relevant_vector = csr_matrix((1, configs["gram_count"]), dtype=np.float32)
    nonrelevant_vector = csr_matrix((1, configs["gram_count"]), dtype=np.float32)
    
    for doc_id in relevant_doc_ids:
        relevant_vector += sparse_matrix[doc_id]
    for doc_id in relevant_doc_ids:
        nonrelevant_vector += sparse_matrix[doc_id]
    relevant_vector /= C_r
    nonrelevant_vector /= C_nr
    return relevant_vector.multiply(configs["beta"])


    # inverted_files = corpus["inverted_files"]
    # relevant_vector = dict.fromkeys(inverted_files,0)
    # nonrelevant_vector = dict.fromkeys(inverted_files,0)
    # for word in inverted_files:
    #     for doc_id in relevant_doc_ids:
    #         if doc_id in inverted_files[word]["docs"]:
    #             relevant_vector[word] += inverted_files[word]["docs"][doc_id] / C_r
    #     for doc_id in nonrelevant_doc_ids:
    #         if doc_id in inverted_files[word]["docs"]:
    #             nonrelevant_vector[word] += inverted_files[word]["docs"][doc_id] / C_nr
    
    # for word in inverted_files:
    #     query_weight = query["words"][word] if word in query["words"] else 0
    #     relevant_weight = relevant_vector[word]
    #     nonrelevant_weight = nonrelevant_vector[word]
    #     if query_weight == 0 and relevant_weight == 0 and nonrelevant_weight == 0:
    #         continue
    #     query["words"][word] = configs["alpha"] * query_weight \
    #                 + configs["beta"] * relevant_weight \
    #                 + configs["gamma"] * nonrelevant_weight