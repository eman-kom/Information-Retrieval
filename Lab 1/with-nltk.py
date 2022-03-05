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
from nltk.probability import FreqDist, MLEProbDist


def build_LM(in_file: str) -> dict:
    """build language models for each label from input file"""

    combinedSet = set()
    LM = {"tamil": FreqDist(), "indonesian": FreqDist(), "malaysian": FreqDist()}

    with open(in_file, "r", encoding="utf-8") as file:
        for line in file:
            language, text = line.split(" ", 1)
            ngrams = list(nltk.ngrams(parse(text), 4))
            LM[language].update(FreqDist(ngrams))
            combinedSet.update(ngrams)

    for language in LM:
        LM[language] = FreqDist(combinedSet) + LM[language]  # add-1 smoothing
        LM[language] = MLEProbDist(LM[language])  # MLEProbDist NEEDS FreqDist

    return LM


def parse(string: str) -> str:
    """removes of any non-alphabetic characters from input string"""
    return list(re.sub("[^a-zA-Z ]", "", string.lower()))


def test_LM(in_file: str, out_file: str, LM: dict) -> None:
    """
    test the language models on new strings from input file
    attaches label before string in output file
    """

    with open(in_file, "r", encoding="utf-8") as file:
        with open(out_file, "w", encoding="utf-8") as out:

            for line in file:
                skipped = 0
                isUnknown = False
                ngrams = list(nltk.ngrams(parse(text), 4))
                scores = {"malaysian": 0, "indonesian": 0, "tamil": 0}

                for ngram in ngrams:
                    isIgnored = True

                    for language in LM:
                        prob = LM[language].prob(ngram)

                        if (prob != 0):
                            scores[language] += math.log(prob)
                            isIgnored = False

                    if (isIgnored):
                        skipped += 1

                    if (skipped / len(ngrams) > 0.6):
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
if input_file_b is None or input_file_t is None or output_file is None:
    usage()
    sys.exit(2)

LM = build_LM(input_file_b)
test_LM(input_file_t, output_file, LM)
