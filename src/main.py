import parser
import sparse_matrix
import predict
import rocchio
from tqdm.auto import tqdm

configs = {
    "k": 1.5,
    "b": 0.5,
    "alpha": 1,
    "beta": 0.75,
    "gamma": 0.15,
    "target": 100,
    "rochio": False,
    "query_path": "../queries/query-test.xml",
    "output_path": "../prediction.csv",
    "model_path": "../model",
    "corpus_path": "../CIRB010",
    "title_weight": 1,
    "question_weight": 1,
    "concepts_weight": 1,
    "narrative_weight": 1,
    "query_path": "../queries/query-test.xml",
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
#     parser.parse_arg(configs)
    fname_to_id, id_to_fname, id_to_doclen = parser.parse_file_list(configs)
    vocab_to_id, id_to_vocab = parser.parse_vocab_list(configs)
    N = len(fname_to_id)
    inverted_files = parser.parse_inverted_file(configs, N)
    # old_inverted_files = inverted_files.copy()
    # # Save checkpoint for notebook
    # inverted_files = old_inverted_files.copy()
    avdl = sum(id_to_doclen.values()) / len(id_to_doclen)
    corpus = {
        "fname_to_id": fname_to_id,
        "id_to_doclen": id_to_doclen,
        "id_to_fname": id_to_fname,
        "vocab_to_id": vocab_to_id,
        "id_to_vocab": id_to_vocab,
        "inverted_files": inverted_files,
        "avdl": avdl,
    }
    sparse_matrix.gen_matrix(corpus, configs)
    id_to_magnitude = sparse_matrix.gen_id_to_magnitude(corpus, configs)
    corpus["id_to_magnitude"] = id_to_magnitude
    print("Processing Query")
    queries = parser.parse_queries(corpus, configs, configs["query_path"])
    queries = sparse_matrix.gen_query_vector(queries, corpus, configs)
    query_to_magnitude = sparse_matrix.gen_query_to_magnitude(queries)
    query_responses = []
    for query in tqdm(queries):
        response = predict.predict_query(query, corpus, configs)
        query_responses.append(response)
    for _ in tqdm(range(configs["rocchio_iters"])):
        for i in tqdm(range(len(query_responses))):
            rocchio.rocchio_feedback(query_responses[i], queries[i],  corpus, configs)
            response = predict.predict_query(queries[i], corpus, configs)
            query_responses[i] = response
    predict.process_predictions(query_responses, configs)
    predict.write_predictions(query_responses, queries)
    predict.calc_MAP(query_responses, configs)