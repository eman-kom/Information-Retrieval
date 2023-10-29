#!/usr/bin/python3
from collections import Counter, namedtuple
from itertools import chain
from math import log10, sqrt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pickle import dump
from re import sub
from string import ascii_lowercase
import csv
import getopt
import sys

indexDict = {}
postingList = {}
porter = PorterStemmer()

docs = {}
postings = {}


def process_tokens(docID: int, termList: list) -> None:
    """
    Finds the positions and frequencies of each term and builds the index.

    Parameters:
        docID (int): the docID of the current row
        termList (list): a list of tokenized and stemmed terms

    Returns:
        None
    """

    pos = 0
    length = 0
    termSet = set()
    freqs = {}

    # get positional indices
    for term in termList:

        if term not in postings:
            postings[term] = [[docID, 0, []]]

        if postings[term][-1][0] != docID:
            postings[term].append([docID, 0, []])

        termSet.add(term)
        postings[term][-1][1] += 1
        postings[term][-1][2].append(pos)

        pos += 1

    # get document vector and length
    for term in termSet:
        freqs[term] = postings[term][-1][1]
        log_tf = log10(postings[term][-1][1]) + 1
        postings[term][-1][1] = log_tf
        length += log_tf ** 2

    docs[docID] = [sqrt(length), Counter(freqs)]


def get_terms(text: str, stop_words: set) -> list:
    """
    Santitize the text so that it can be better processed. (e.g.
    expanding contractions, removing all non-alphanumeric characters and
    renoving stopwords). Also performs tokenization and stemming.

    Parameters:
        text (str): the text in question
        stop_words: a set containing the stop words to be used

    Returns:
        text (list): a list of tokenized and stemmed terms
    """

    text = text.lower()
    text = sub("`", "'", text)

    text = sub("shan't", "shall not", text)
    text = sub("won't", "will not", text)
    text = sub("let's", "let us", text)
    text = sub("'ll ", " will ", text)
    text = sub("'ve ", " have ", text)
    text = sub("n't ", " not ", text)
    text = sub("'re ", " are ", text)
    text = sub("'d ", " had ", text)
    text = sub("'s ", " is ", text)
    text = sub("i'm", "i am", text)

    text = sub("pre-", "pre", text)
    text = sub("[^0-9a-zA-Z]+", " ", text)

    text = text.split()
    text = filter(lambda term: term not in stop_words, text)
    text = map(lambda term: porter.stem(term), text)

    return text


def buildIndex(input_file: str, dict_file: str, postings_file: str) -> None:
    """
    Reads the input file and processes each row in the input file.

    Parameters:
        input_file (str): The file path of the input csv file
        dict_file (str): The file path of the output dictionary file
        postings_file (str): The file path of the output postings file

    Returns:
        None
    """

    docCount = 0
    processed_documents = set()
    csv.field_size_limit(sys.maxsize)

    stop_words = set(stopwords.words('english'))
    stop_words.update(list(ascii_lowercase))

    with open(input_file, "r") as i:
        csv_data = csv.DictReader(i)

        for row in csv_data:
            content = get_terms(row["content"], stop_words)
            court = get_terms(row["court"], stop_words)
            title = get_terms(row["title"], stop_words)
            docID = int(row["document_id"])

            if docID not in processed_documents:
                print("Processing docID:", docID)
                process_tokens(docID, chain(title, content, court))
                processed_documents.add(docID)
                docCount += 1

    with open(postings_file, "wb") as p:

        for term, postingList in postings.items():
            idf = log10(docCount / len(postingList))
            postings[term] = [idf, p.tell()]
            dump(postingList, p)

        for docID, info in docs.items():
            docs[docID] = [info[0], p.tell()]
            dump(info[1], p)

    with open(dict_file, "wb") as d:
        dump(postings, d)
        dump(docs, d)


def usage():
    print("usage: "
          + sys.argv[0]
          + " -i input-file -d dictionary-file -p postings-file")


def parseArguments():
    input_file = dictionary_file = postings_file = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == '-i':  # input file
            input_file = a
        elif o == '-d':  # dictionary file
            dictionary_file = a
        elif o == '-p':  # postings file
            postings_file = a
        else:
            assert False, "unhandled option"

    if input_file is None or dictionary_file is None or postings_file is None:
        usage()
        sys.exit(2)

    return input_file, dictionary_file, postings_file


if __name__ == '__main__':
    i, d, p = parseArguments()
    buildIndex(i, d, p)
