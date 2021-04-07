import math
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET
import argparse

def parse_arg(configs):
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", action="store_true", default=False, help="Whether to use Relevance Feedback")
    parser.add_argument("-i", action="store", default="../queries/query-test.xml")
    parser.add_argument("-o", action="store", default="../prediction.csv")
    parser.add_argument("-m", action="store", default="../model")
    parser.add_argument("-d", action="store", default="../CIRB010")
    args = parser.parse_args()
    configs["use_rochio"] = args.r
    configs["query_path"] = args.i
    configs["output_path"] = args.o
    configs["model_path"] = args.m
    configs["corpus_path"] = args.d

def get_unigram(s, dic, vocab_to_id, weight, unigram_weight):
    for i in range(len(s)):
        if s[i] in vocab_to_id:
            key = (vocab_to_id[s[i]], -1)
            if key in dic:
                dic[key] += 1 * weight * unigram_weight
            else:
                dic[key] = 1 * weight * unigram_weight

def get_bigram(s, dic, vocab_to_id, inverted_files, weight, bigram_weight):
    for i in range(len(s)-1):
        # first make sure both exists
        if s[i] in vocab_to_id and s[i+1] in vocab_to_id and \
        (vocab_to_id[s[i]], vocab_to_id[s[i+1]]) in inverted_files:
            key = (vocab_to_id[s[i]], vocab_to_id[s[i+1]])
            if key in dic:
                dic[key] += 1 * weight * bigram_weight
            else:
                dic[key] = 1 * weight * bigram_weight

def get_doclen(corpus_path, file_path):
    tree = ET.parse(corpus_path + "/" +file_path)
    root = tree.getroot()
    dl = 0
    for doc in root:
        for child in doc:
            if child.tag == "title":
                if child.text:
                    dl += len(child.text)
            elif child.tag == "text":
                for paragraph in child:
                    if paragraph.text:
                        dl += len(paragraph.text)
    return dl

def get_IDF(N, k):
    return math.log( ( N-k+0.5 ) / ( k+0.5 ) + 1 )
    # return math.log((N+1)/k)

def parse_file_list(configs):
    fname_to_id = {}
    id_to_fname = {}
    id_to_doclen = {}
    file_list_path = configs["model_path"] + "/file-list"
    with open(file_list_path) as f:
        for index, name in enumerate((f)):
            name = name.strip()
            fname_to_id[name] = index
            id_to_fname[index] = name
            id_to_doclen[index] = get_doclen(configs["corpus_path"], name)
    return fname_to_id, id_to_fname, id_to_doclen
        
def parse_vocab_list(configs):
    vocab_to_id = {}
    id_to_vocab = {}
    file_list_path = configs["model_path"] + "/vocab.all"
    with open(file_list_path) as f:
        for index, name in enumerate((f)):
            name = name.strip()
            vocab_to_id[name] = index
            id_to_vocab[index] = name
    return vocab_to_id, id_to_vocab

def parse_inverted_file(configs, N):
    '''
    N: The total number of files in the corpus
    '''
    inverted_files = {}
    inverted_list_path = configs["model_path"] + "/inverted-file"
    with open(inverted_list_path, 'r') as f:
        print("Reading Inverted Files: ")
        pbar = tqdm(total = 1193467)
        line = f.readline()
        while line:
            line = line.split()
            vocab_1, vocab_2 = int(line[0]), int(line[1])
            key = (vocab_1, vocab_2) 
            k = int(line[2])
            inverted_files[key] = {"IDF": get_IDF(N, k), "docs": {}}
            max_freq = 0
            for _ in range(k):
                line = f.readline().strip().split()
                doc_id = int(line[0])
                doc_freq = int(line[1])
                inverted_files[key]["docs"][doc_id] = doc_freq
                max_freq = max(max_freq, doc_freq)
            inverted_files[key]["max_freq"] = max_freq
            line = f.readline()
            pbar.update(1)
    return inverted_files

def parse_queries(corpus, configs, path):
    tree = ET.parse(path)
    root = tree.getroot()
    queries = []
    vocab_to_id = corpus["vocab_to_id"]
    inverted_files = corpus["inverted_files"]
    for xml_query in root:
        query = {}
        query["id"] = xml_query.find("number").text
        words = {}
        dl = 0
        for properties in xml_query:
            if properties.tag == "number":
                continue
            weight = configs[properties.tag + "_weight"]
            text = properties.text.strip()
            dl += len(text)
            get_unigram(text, words, vocab_to_id, weight, configs["unigram_weight"])
            get_bigram(text, words, vocab_to_id, inverted_files, weight, configs["bigram_weight"])
        query["words"] = words
        query["dl"] = dl
        
        queries.append(query)
    return queries
    