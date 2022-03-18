#!/usr/bin/python3
import re
import nltk
import sys
import getopt

import os
from postings import Postings


def build_index(in_dir: str, out_dict: str, out_postings: str) -> None:
    """
    Builds index from all files in in_dir and splits the index into its
    dictionary and its posting lists

    Parameters:
        in_dir (str): folder path of reuters training files
        out_dir (str): file path for dictionary output
        out_postings (str): file path for storing all posting lists

    Returns:
        None
    """
    print("start indexing")

    post = Postings()
    docList = sorted(map(int, os.listdir(in_dir)))

    for docID in docList:
        file = os.path.join(in_dir, str(docID))

        with open(file, "r") as f:
            for sentence in nltk.sent_tokenize(f.read()):
                tokenList = nltk.word_tokenize(sentence)
                tokenList.append("!!all!!")  # To ensure that docID is added to the 'all' list
                post.addToPostings(tokenList, docID)

    # write last remaining block not saved to disk
    post.writeBlockToDisk()

    # merge blocks and save to dictionary and postings to relevant output files
    with open(out_dict, "wb") as out_dict:
        with open(out_postings, "wb") as out_postings:
            post.mergeBlocks(out_dict, out_postings)

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
