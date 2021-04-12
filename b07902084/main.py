import parser
import sparse_matrix
import predict
import rocchio
from tqdm.auto import tqdm

configs = {
    "k": 1.5,
    "b": 0.75,
    "ka": 1.5,
    "alpha": 1,
    "beta": 0.2,
    "gamma": 0.15,
    "target": 100,
    "use_rocchio": False,
    "query_path": None,
    "output_path": None,
    "model_path": None,
    "corpus_path": None,
    "title_weight": 1,
    "question_weight": 1,
    "concepts_weight": 1,
    "narrative_weight": 1,
    "query_path": "../queries/query-train.xml",
    "cdn": 1,
    "ctc": 1,
    "cte": 1,
    "cts": 1,
    "unigram_weight": 1,
    "bigram_weight": 1,
    "rocchio_iters": 1,
    "use_cosine": False
}

if __name__ == '__main__':
    parser.parse_arg(configs)
    print("Rocchio Mode: ", configs["use_rocchio"])
    fname_to_id, id_to_fname = parser.parse_file_list(configs)
    vocab_to_id, id_to_vocab = parser.parse_vocab_list(configs)
    doc_count = len(fname_to_id)
    inverted_files, gram_to_id, gram_count, id_to_doclen = parser.parse_inverted_file(configs, doc_count)
    configs["gram_count"] = gram_count
    configs["doc_count"] = doc_count
    # Save checkpoint for notebook
    avdl = sum(id_to_doclen.values()) / len(id_to_doclen)
    corpus = {
        "fname_to_id": fname_to_id,
        "id_to_doclen": id_to_doclen,
        "id_to_fname": id_to_fname,
        "vocab_to_id": vocab_to_id,
        "id_to_vocab": id_to_vocab,
        "inverted_files": inverted_files,
        "gram_to_id": gram_to_id,
        "avdl": avdl,
    }
    corpus["sparse"] = sparse_matrix.gen_matrix(corpus, configs)
    print("Processing Query")
    queries = parser.parse_queries(corpus, configs, configs["query_path"])
    sparse_queries = []
    for query in queries:
        sparse_queries.append( sparse_matrix.gen_query_vector(query, corpus, configs) )
    query_responses = []
    for sparse_query in sparse_queries:
        query_responses.append( predict.predict_query(sparse_query, corpus, configs) )
    if configs["use_rocchio"]:
        print("Rocchio Feedback")
        for _ in range(configs["rocchio_iters"]):
            for i in range(len(query_responses)):
                feedback_vec = rocchio.rocchio_feedback(query_responses[i], sparse_queries[i],  corpus, configs)
                response = predict.predict_query(sparse_queries[i] + feedback_vec, corpus, configs) 
                query_responses[i] = response
    predict.process_predictions(query_responses, configs, corpus)
    print("writing predictions")
    predict.write_predictions(query_responses, queries, configs)
    # predict.calc_MAP(query_responses, configs)