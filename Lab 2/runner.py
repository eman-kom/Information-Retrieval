import pickle
from typing import BinaryIO, Tuple


class Runner:
    """
    Performs boolean retrieval on the query

    Parameters:
        post_handle (BinaryIO): The handle for the postings file
        dictionary (dict): Contains terms and their pointers to the posting list

    Attributes:
        terms (dict): Stores input dictionary
        postings (BinaryIO): Stores the handle of the postings file
        notSkip (int): To indicate entry does not have a valid skip pointer
    """

    def __init__(self, post_handle: BinaryIO, dictionary: dict):
        self.notSkip = -1
        self.terms = dictionary
        self.postings = post_handle

    def execute(self, query: list) -> str:
        """
        Performs boolean retrieval on the query

        Parameters:
            query (list): the query in reverse polish notation

        Returns
            out (str): The list of docIDs that satisfies the query
        """

        stack = list()

        while query:
            token = query.pop(0)

            if token == "AND":
                t1, t2 = self.get(stack.pop(), stack.pop())
                stack.append(self.AND(t1, t2))

            elif token == "OR":
                t1, t2 = self.get(stack.pop(), stack.pop())
                stack.append(self.OR(t1, t2))

            elif token == "NOT":
                # find all docIDs NOT in the given docs so need to compare with the !!all!! list
                t1, t2 = self.get(stack.pop(), '!!all!!')
                stack.append(self.NOT(t1, t2))

            else:
                stack.append(token)

        result = stack.pop()

        if isinstance(result, list):
            out = " ".join(result)
        else:
            # Used for cases where the query is just 1 string
            out = ""
            for term in self.getPostings(result):
                docID, skip = self.getID(term)
                out += str(docID) + " "

        return out.strip()

    def AND(self, t1: list, t2: list) -> list:
        """
        Performs the AND operation on 2 lists

        Parameters:
            t1 (list): The first list
            t2 (list): The second list

        Returns:
           intermediate (list): The list containing docIDs found in both input lists
        """

        p1 = 0
        p2 = 0
        intermediate = list()

        while p1 < len(t1) and p2 < len(t2):

            doc1, skip1 = self.getID(t1[p1])
            doc2, skip2 = self.getID(t2[p2])

            if doc1 == doc2:
                intermediate.append(str(doc1))
                p1 += 1
                p2 += 1

            elif doc1 < doc2:
                if skip1 != self.notSkip and self.getID(t1[skip1])[0] <= doc2:
                    while skip1 != self.notSkip and self.getID(t1[skip1])[0] <= doc2:
                        p1 = skip1
                        skip1 = self.getID(t1[p1])[1]
                else:
                    p1 += 1

            else:  # doc1 > doc2
                if skip2 != self.notSkip and self.getID(t2[skip2])[0] <= doc1:
                    while skip2 != self.notSkip and self.getID(t2[skip2])[0] <= doc1:
                        p2 = skip2
                        skip2 = self.getID(t2[p2])[1]
                else:
                    p2 += 1

        return intermediate

    def OR(self, t1: list, t2: list) -> list:
        """
        Performs the OR operation on 2 lists

        Parameters:
            t1 (list): The first list
            t2 (list): The second list

        Returns:
           intermediate (list): The list containing docIDs found in all input lists
        """

        p1 = 0
        p2 = 0
        intermediate = list()

        while p1 < len(t1) and p2 < len(t2):

            doc1, skip1 = self.getID(t1[p1])
            doc2, skip2 = self.getID(t2[p2])

            if doc1 == doc2:
                intermediate.append(str(doc1))
                p1 += 1
                p2 += 1

            elif doc1 < doc2:
                intermediate.append(str(doc1))
                p1 += 1

            else:
                intermediate.append(str(doc2))
                p2 += 1

        # when there are still docIDs left in list1
        while p1 < len(t1):
            doc1, skip1 = self.getID(t1[p1])
            intermediate.append(str(doc1))
            p1 += 1

        # when there are still docIDs left in list2
        while p2 < len(t2):
            doc2, skip2 = self.getID(t2[p2])
            intermediate.append(str(doc2))
            p2 += 1

        return intermediate

    def NOT(self, t1: list, t2: list) -> list:
        """
        Performs the NOT operation on list t1

        Parameters:
            t1 (list): The first list
            t2 (list): The second list

        Returns:
           intermediate (list): The list containing docIDs not found in list t1
        """

        p1 = 0
        p2 = 0
        intermediate = list()

        while p1 < len(t1) and p2 < len(t2):

            doc1, skip1 = self.getID(t1[p1])
            doc2, skip2 = self.getID(t2[p2])

            if doc1 == doc2:
                p1 += 1
                p2 += 1

            elif doc1 < doc2:
                intermediate.append(str(doc1))
                p1 += 1

            else:
                intermediate.append(str(doc2))
                p2 += 1

        # when there are still docIDs left in the "!!all!!" list
        while p2 < len(t2):
            doc2, skip2 = self.getID(t2[p2])
            intermediate.append(str(doc2))
            p2 += 1

        return intermediate

    def get(self, t1: any, t2: any) -> Tuple[list, list]:
        """
        Checks if top of stack is a term or a list and responds appropriately

        Parameters:
            t1 (any): Can either be a list or term
            t2 (any): Can either be a list or term

        Returns:
            t1 (list): The posting list associated with the first input (t1)
            t2 (list): The posting list associated with the first input (t2)
        """
        if not isinstance(t1, list):
            t1 = self.getPostings(t1)

        if not isinstance(t2, list):
            t2 = self.getPostings(t2)

        return t1, t2

    def getID(self, entry: str) -> Tuple[int, int]:
        """
        Gets the docID and skip pointer (if any) from the entry in the posting list

        Parameters:
            entry (str): The docID/docID with skip pointer from the posting list

        Returns:
            entry (int): The docID
            skip (int): The skip pointer (-1 if skip pointer is invalid)
        """

        if "|" in str(entry):
            docID, skip = entry.split("|")
            return int(docID), int(skip)

        return int(entry), self.notSkip

    def getPostings(self, key: str) -> list:
        """
        Gets the relevants posting list from the key

        Parameters:
            key (str): The search term

        Returns:
            postingList (list): An empty list if term is not found
                                The posting list if term is found
        """

        if key not in self.terms:
            return list()

        # uses absolute value for seek so no need to rewind
        self.postings.seek(self.terms[key]["ptr"])
        postingList = pickle.load(self.postings)

        return postingList
