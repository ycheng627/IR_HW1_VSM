file_names = [
    "ensemble-0.79813.csv",
    "ensemble-0.79728.csv",
]


all_docs = set()

file_dict = {}
for i in file_names:
    file_dict[i] = {}
    for j in range(11, 31):
        file_dict[i][j] = None
for file_path in file_names:
    with open(file_path) as f:
        for line_no, line in enumerate(f):
            if line_no == 0 or line_no == 21:
                continue
            query_no = line_no + 10
            ranked_docs = line.split(",")[1].split()
            all_docs.update(ranked_docs)
            file_dict[file_path][query_no] = ranked_docs
# print(all_docs)

output = open("ensemble.csv", "w")
output.write('query_id,retrieved_docs\n')

for query_no in range(11, 31):
    query_rank = dict.fromkeys(all_docs, 0)
    for csv in file_dict:
        for rank, doc in enumerate(file_dict[csv][query_no]):
            # print(rank, doc)
            query_rank[doc] += 100 - rank

    rank = sorted(query_rank, key=query_rank.get, reverse=True)[:100]
    query_no = '0' + str(query_no)
    output.write('{},{}\n'.format(query_no, ' '.join(rank)))

        

