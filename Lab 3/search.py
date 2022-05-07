#!/usr/bin/python3
import re
import nltk
import sys
import getopt

import math
import heapq
import pickle
from collections import Counter
from nltk.stem import PorterStemmer


def run_search(dict_file: str, postings_file: str, queries_file: str, results_file: str) -> None:
    """
    Use cosine similarity to find the top 10 most relevant documents for
    each free text query.

    Parameters:
        dict_file (str): file path for the dictionary file
        postings_file (str): file path for the postings file
        queries_file (str): file path for the queries file
        results_file (str): file path for the results file

    Returns:
        None
    """
    print("start searching")

    # initialises dictionary and lengths of documents
    with open(dict_file, "rb") as d:
        index = pickle.load(d)
        lengths = pickle.load(d)

    porter = PorterStemmer()
    results = open(results_file, "w")
    queries = open(queries_file, "r")
    posts = open(postings_file, "rb")

    for query in queries:

        heap = []
        scores = {}
        terms = nltk.word_tokenize(query.strip())
        term_freq = Counter(map(lambda word: porter.stem(word.lower()), terms))

        for term, tf in term_freq.items():

            # finds the idf and the postings list of the term
            if term in index:
                posts.seek(index[term]["ptr"])
                postings = pickle.load(posts)
                idf = index[term]["idf"]

            else:
                idf = 0
                postings = []

            # gets the logarithmic tf and idf in the "lt" of "ltc"
            q_weight = (1 + math.log(tf, 10)) * idf

            # gets the dot product for the documents
            for docID, d_weight in postings:
                scores[docID] = scores.get(docID, 0) + q_weight * d_weight

        # gets cosine similarity and push result to heap
        # negative docID to sort in ascending order
        for docID, dot_product in scores.items():
            cos_sim = dot_product / lengths[docID]
            heapq.heappush(heap, (cos_sim, -docID, docID))

        # gets top 10 results by descending
        # cosine similarity and ascending docID
        top_10 = heapq.nlargest(10, heap)

        # converts answer array to string
        answer = ""
        for _, _, docID in top_10:
            answer += str(docID) + " "

        print(answer.strip(), file=results)

    posts.close()
    results.close()
    queries.close()
    print("search completed")


def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")


dictionary_file = postings_file = file_of_queries = output_file_of_results = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file is None or postings_file is None or file_of_queries is None or file_of_output is None:
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
