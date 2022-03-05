#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import nltk
import sys
import getopt
import math


def parse(text: str) -> str:
    """Removes noise from the text supplied"""

    text = text.lower()
    text = re.sub("[^a-zA-Z ]", "", text)
    return text


def extract(text: str) -> list:
    """Returns the list of 4-grams extracted from the text given"""

    size = 4
    ngrams = list()

    for i in range(0, len(text) - (size - 1)):
        ngrams.append(text[i:i + size])

    return ngrams


def build_LM(in_file: str) -> dict:
    """Parses the input file and builds the language model"""

    combined = set()
    counts = {"tamil": 0, "malaysian": 0, "indonesian": 0}
    models = {"tamil": {}, "malaysian": {}, "indonesian": {}}

    with open(in_file, "r", encoding="utf-8") as file:

        for line in file:
            lang, text = line.split(" ", 1)
            ngramList = extract(parse(text))

            for ngram in ngramList:
                models[lang][ngram] = models[lang].get(ngram, 0) + 1

            combined.update(ngramList)
            counts[lang] += len(ngramList)

        # converts the counts into probability after add-1 smoothing
        for ngram in combined:
            for lang in models:
                smooth = models[lang].get(ngram, 0) + 1
                models[lang][ngram] = smooth / (counts[lang] + len(combined))

    return models


def test_LM(in_file: str, out_file: str, LM: dict) -> None:
    """Runs the test file against the language model and outputs results into a file"""

    threshold = 0.6

    with open(in_file, "r", encoding="utf-8") as file:
        with open(out_file, "w", encoding="utf-8") as out:

            for line in file:
                skipped = 0
                isUnknown = False
                ngramList = extract(parse(line))
                scores = {"malaysian": 0, "indonesian": 0, "tamil": 0}

                for ngram in ngramList:
                    isIgnored = True

                    for lang in LM:
                        prob = LM[lang].get(ngram, 0)

                        if (prob != 0):
                            scores[lang] += math.log(prob)
                            isIgnored = False

                    if (isIgnored):
                        skipped += 1

                    if (skipped / len(ngramList) > threshold):
                        isUnknown = True
                        break

                if (isUnknown):
                    out.write("other " + line)
                else:
                    bestGuess = max(scores, key=scores.get)
                    out.write(bestGuess + " " + line)


def usage():
    print(
        "usage: "
        + sys.argv[0]
        + " -b input-file-for-building-LM -t input-file-for-testing-LM -o output-file"
    )


input_file_b = input_file_t = output_file = None
try:
    opts, args = getopt.getopt(sys.argv[1:], "b:t:o:")
except getopt.GetoptError:
    usage()
    sys.exit(2)
for o, a in opts:
    if o == "-b":
        input_file_b = a
    elif o == "-t":
        input_file_t = a
    elif o == "-o":
        output_file = a
    else:
        assert False, "unhandled option"
if input_file_b == None or input_file_t == None or output_file == None:
    usage()
    sys.exit(2)

LM = build_LM(input_file_b)
test_LM(input_file_t, output_file, LM)
