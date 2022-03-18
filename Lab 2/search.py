#!/usr/bin/python3
import re
import nltk
import sys
import getopt
import pickle
from parser import Parser
from runner import Runner


def run_search(dict_file: str, postings_file: str, queries_file: str, results_file: str) -> None:
    """
    Initialise dictionary with the dict_file, reads queries from queries_file
    finds all documents that contain the given parameters from the query and
    stores them in the results file.

    Parameters:
        dict_file (str): file path for the dictionary file
        postings_file (str): file path for the postings file
        queries_file (str): file path for the queries file
        results_file (str): file path for the results file

    Returns:
        None
    """
    print("start searching")

    with open(dict_file, "rb") as d:
        terms = pickle.load(d)

    results = open(results_file, "w")
    queries = open(queries_file, "r")
    postings = open(postings_file, "rb")

    parser = Parser()
    runner = Runner(postings, terms)

    for query in queries:
        reversePolish = parser.parse(query)
        print(runner.execute(reversePolish), file=results)

    results.close()
    queries.close()
    postings.close()
    print("search completed")


def usage():
    print("usage: " +
          sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")


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

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None:
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
