import os
import math
import heapq
import pickle
import string
from typing import BinaryIO
from collections import OrderedDict
from nltk.stem import PorterStemmer


class Postings:
    """
    Creates posting list and stores them in index and postings files

    Attributes:
        pickleno (int): filename for the intermediate storage pickle file
        postings (dict): In-memory storage of postings list
        blocksize (int): Max size that the postings reach
        porter (PorterStemmer): Used to perform stemming on tokens
    """

    def __init__(self):
        self.pickleno = 0
        self.postings = {}
        self.blocksize = 500000  # in bytes (0.5 mb); should produce like 7 files
        self.porter = PorterStemmer()

    def addToPostings(self, tokenList: list, docID: int) -> None:
        """
        Adds each token from the token list to the dictionary


        Parameters:
            tokenList (list): List of tokens from the document parsed
            docID (int): docID of the file associated with the token list

        Returns:
            None
        """
        bannedTokens = ["``", "--", "''"]

        for token in tokenList:
            # filters tokens which do not have any value
            if not (token in string.punctuation or token in bannedTokens or ".." in token):

                term = self.porter.stem(token.lower())

                if term not in self.postings:
                    self.postings[term] = list()

                if docID not in self.postings[term]:
                    self.postings[term].append(docID)

                # store dictionary into intermediate storage if size exceeds allocated memory size
                if self.postings.__sizeof__() > self.blocksize:
                    self.writeBlockToDisk()

    def writeBlockToDisk(self) -> None:
        """
        Writes intermediate dictionary to disk

        Parameters:
            None

        Returns:
            None
        """
        print("creating block no:", self.pickleno)
        sortedTerms = OrderedDict(sorted(self.postings.items()))

        with open(str(self.pickleno) + ".pickle", "wb") as pickleFile:
            # saves each term, posting as a separate pickle
            # save pickle file number for comparison on heap later
            for term, posting in sortedTerms.items():
                toSave = [term, self.pickleno, posting]
                pickle.dump(toSave, pickleFile)

        self.pickleno += 1
        self.postings = {}  # Resets dictionary

    def mergeBlocks(self, out_dict: BinaryIO, out_postings: BinaryIO) -> None:
        """
        Merge blocks in n-way merge and stores terms with docCount and pointer
        in dictionary file and postings in postings file

        Parameters:
            out_dict (BinaryIO): File handle for the dictionary file
            out_postings (BinaryIO): File handle for the postings file

        Returns:
            None
        """
        print("merging blocks")

        heap = []
        dictionary = {}

        # initialise intermediate dictionary pickle file handles
        handles = [open(str(i) + ".pickle", "rb") for i in range(self.pickleno)]

        # initialise heap
        for entry in handles:
            heapq.heappush(heap, pickle.load(entry))

        # used to check if all postings for a term is combined
        currentToken = heapq.heappop(heap)

        # always ensure that there are i elements in the heap
        # so that the smallest token will be compared with all intermediate files
        heapq.heappush(heap, pickle.load(handles[currentToken[1]]))

        while heap:

            # gets smallest token also taking into account file order
            entry = heapq.heappop(heap)

            if entry[0] != currentToken[0]:
                # stores term and posting if all postings for that term is found
                self.processEntry(currentToken, out_postings, dictionary)
                currentToken = entry
            else:
                for docID in entry[2]:
                    if docID not in currentToken[2]:
                        currentToken[2].append(docID)

            try:
                heapq.heappush(heap, pickle.load(handles[entry[1]]))
            except EOFError:
                # delete intermediate file if no more tokens can be read
                handles[entry[1]].close()
                os.remove(str(entry[1]) + ".pickle")

        # process the last token
        self.processEntry(currentToken, out_postings, dictionary)
        pickle.dump(dictionary, out_dict)

    def processEntry(self, entry: list, out_postings: BinaryIO, dictionary: dict) -> None:
        """
        Process a completed entry from the merged blocks and stores its
        postings into the postings file

        Parameters:
            entry (list): [term, pickle_file_number, posting_list]
            out_postings (BinaryIO): File handle for the postings file
            dictionary (dict): Dictionary to store information about the entry itself

        Returns:
            None
        """
        docCount = len(entry[2])
        startPointer = out_postings.tell()  # get position of postings file current pointer
        dictionary[entry[0]] = {"count": docCount, "ptr": startPointer}
        postingList = self.addSkipPointers(entry[2])
        pickle.dump(postingList, out_postings)

    def addSkipPointers(self, postings: list) -> list:
        """
        Adds skip pointers to the list

        Parameters:
            postings (list): the posting list

        Returns:
            postings (list): the posting list with skip pointers added
        """
        ptr = 0
        skip = int(math.sqrt(len(postings)))

        while ptr < len(postings) - skip:
            postings[ptr] = str(postings[ptr]) + "|" + str(skip + ptr)
            ptr += skip

        # marks the last skip pointer as -1 to prevent further skips
        postings[ptr] = str(postings[ptr]) + "|" + str(-1)
        return postings
