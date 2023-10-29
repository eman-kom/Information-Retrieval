#!/usr/bin/python3
from boolean_retrieval import Boolean
from ranked_retrieval import Ranked
from re import match
import getopt
import heapq
import logging
import pickle
import sys


def run_search(dict_file: str, posts_file: str, query_file: str, results_file: str) -> None:
    """
    Performs the search.

    Parameters:
        dict_file (str): the file path of the dictionary file
        posts_file (str): the file path of the postings file
        query_file (str): the file path of the query file
        results_file (str): the file path of the results file

    Returns:
        None
    """

    result = ""

    with open(dict_file, "rb") as d:
        index = pickle.load(d)
        docs = pickle.load(d)

    with open(query_file, "r") as q:
        query = q.readline().strip()
        relevant_docs = q.read().splitlines()
        relevant_docs = list(map(int, relevant_docs))

    with open(posts_file, "rb") as p:

        if '"' in query or "AND" in query:
            # heap = Boolean(query, relevant_docs, index, all_docs, p).execute()
            query = query.replace("AND", " ").replace('"', ' ').strip()
            heap = Ranked(query, relevant_docs, index, docs, p).execute()
        else:
            heap = Ranked(query, relevant_docs, index, docs, p).execute()

    with open(results_file, "w") as r:
        while heap:
            entry = heapq.heappop(heap)
            result += str(entry[2]) + " "

        r.write(result.strip())


def usage():
    print("usage: "
          + sys.argv[0]
          + " -d dictionary-file -p postings-file -q query-file -o output-file")


def parseArguments():
    dictionary_file = postings_file = query_file = results_file = None

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
            query_file = a
        elif o == '-o':
            results_file = a
        else:
            assert False, "unhandled option"

    if dictionary_file is None or postings_file is None or query_file is None or results_file is None:
        usage()
        sys.exit(2)

    return dictionary_file, postings_file, query_file, results_file


if __name__ == '__main__':
    d, p, q, r = parseArguments()
    run_search(d, p, q, r)
