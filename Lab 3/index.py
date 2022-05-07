#!/usr/bin/python3
import re
import nltk
import sys
import getopt

import os
import math
import pickle
from string import punctuation
from collections import Counter
from nltk.stem import PorterStemmer


def build_index(in_dir: str, out_dict: str, out_postings: str) -> None:
    """
    Builds index from all the training documents in in_dir and splits
    the index into its dictionary, its posting lists and the document lengths.

    Parameters:
        in_dir (str): folder path of reuters training files
        out_dir (str): file path for dictionary and document lengths output
        out_postings (str): file path for storing the postings lists

    Returns:
        None
    """
    print("start indexing")

    index = {}
    terms = {}
    lengths = {}
    porter = PorterStemmer()
    docList = sorted(map(int, os.listdir(in_dir)))

    for docID in docList:
        print("reading", docID)

        tokens = []
        doc_length = 0
        file = os.path.join(in_dir, str(docID))

        # tokenizes the document
        with open(file, "r") as f:
            for sentence in nltk.sent_tokenize(f.read()):
                tokens.extend(nltk.word_tokenize(sentence))

        # Stems all the tokens then counts their frequencies
        term_freq = Counter(map(lambda word: porter.stem(word.lower()), tokens))

        # gets the term weights and appends them to the posting list
        for term, tf in term_freq.items():
            log_tf = 1 + math.log(tf, 10)
            terms.setdefault(term, []).append((docID, log_tf))
            doc_length += log_tf ** 2

        # gets the length of the document
        lengths[docID] = math.sqrt(doc_length)

    # gets the idf of each term and saves the posting list into the postings file
    with open(out_postings, "wb") as p:
        for term, postings in terms.items():
            idf = math.log(len(docList) / len(postings), 10)
            index[term] = {"idf": idf, "ptr": p.tell()}
            pickle.dump(postings, p)

    # Save Index and Length[N] into the dictionary file
    with open(out_dict, "wb") as d:
        pickle.dump(index, d)
        pickle.dump(lengths, d)

    print("indexing completed")


def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")


input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i':  # input directory
        input_directory = a
    elif o == '-d':  # dictionary file
        output_file_dictionary = a
    elif o == '-p':  # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_directory is None or output_file_postings is None or output_file_dictionary is None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)
