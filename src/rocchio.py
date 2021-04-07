def rocchio_feedback(query_response, query, corpus, configs):
    # Consider the top 50 query responses as relevant
    # Scan through corpus, and add the value to relevant r or nr
    # Use rocchio to generate a new query, where all vocab will be used

    # top 50 relevant
    relevant_doc_ids = [corpus["fname_to_id"][query_response[i]] for i in range(50)]
    nonrelevant_doc_ids = [corpus["fname_to_id"][query_response[i]] for i in range(len(query_response)-1, len(query_response)-101, -1)]
    C_r = 50
    C_nr = 100
    inverted_files = corpus["inverted_files"]
    relevant_vector = dict.fromkeys(inverted_files,0)
    nonrelevant_vector = dict.fromkeys(inverted_files,0)
    for word in inverted_files:
        for doc_id in relevant_doc_ids:
            if doc_id in inverted_files[word]["docs"]:
                relevant_vector[word] += inverted_files[word]["docs"][doc_id] / C_r
        for doc_id in nonrelevant_doc_ids:
            if doc_id in inverted_files[word]["docs"]:
                nonrelevant_vector[word] += inverted_files[word]["docs"][doc_id] / C_nr
    
    for word in inverted_files:
        query_weight = query["words"][word] if word in query["words"] else 0
        relevant_weight = relevant_vector[word]
        nonrelevant_weight = nonrelevant_vector[word]
        if query_weight == 0 and relevant_weight == 0 and nonrelevant_weight == 0:
            continue
        query["words"][word] = configs["alpha"] * query_weight \
                    + configs["beta"] * relevant_weight \
                    + configs["gamma"] * nonrelevant_weight