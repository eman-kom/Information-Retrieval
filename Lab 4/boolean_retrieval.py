from collections import Counter
from math import log10
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pickle import load
from string import ascii_lowercase
import heapq
from itertools import chain


class Boolean:

    def __init__(self, q, r, i, d, p):
        self.docs = d
        self.index = i
        self.postings_file = p
        self.porter = PorterStemmer()
        self.query = q
        self.relevant_docs = r

    def execute(self) -> list:
        """
        Executes the boolean retrieval

        Parameters:
            None

        Returns:
            heap (list): docIDs sorted from most similar to least similar
        """
        heap = []
        containsAll = []
        temp = {}
        scores = {}
        processed_query = self.process_query()
        counted = self.count_freqs(processed_query)

        for terms, freq in counted.items():
            term = terms.split()

            if len(term) == 1:
                l1 = self.get_postings(term[0])
                containsAll.append(l1)
                temp[terms] = [1 + log10(freq), l1]

            if len(term) == 2:
                l1 = self.get_postings(term[0])
                l2 = self.get_postings(term[1])
                merged = self.phrase_2(l1, l2)
                containsAll.append(l2)
                temp[terms] = [1 + log10(freq), l2]

            elif len(term) == 3:
                l1 = self.get_postings(term[0])
                l2 = self.get_postings(term[1])
                l3 = self.get_postings(term[2])
                merged = self.phrase_2(l1, l2)
                merged = self.phrase_3(merged, l3)
                containsAll.append(l3)
                temp[terms] = [1 + log10(freq), l3]

        merged = self.merge_AND(containsAll)
        flat = [item for sublist in merged for item in sublist]

        for t, info in temp.items():
            for k in info[1]:
                if k[0] in flat:
                    scores[k[0]] = scores.get(k[0], 0) + info[0] * (1 + log10(len(k[2])))

        for docID, dot_product in scores.items():
            cos_sim = dot_product / self.docs[docID][0]
            heapq.heappush(heap, (cos_sim, -docID, docID))

        return heap

    def process_query(self) -> list:
        """
        Processes the query

        Parameter:
            None

        Returns:
            processed (list): the search terms in a list form
        """

        stop_words = set(stopwords.words('english'))
        stop_words.update(list(ascii_lowercase))

        query = self.query.strip().split(" AND ")
        processed = []

        for q in query:
            if '"' in q:
                new_term = q.replace('"', '').split()
            else:
                new_term = [q]

            new_term = filter(lambda term: term not in stop_words, new_term)
            new_term = list(map(lambda term: self.porter.stem(term), new_term))

            if new_term:
                processed.append(new_term)

        return processed

    def get_postings(self, term: list) -> list:
        """
        Gets the posting list from the postings file

        Parameter:
            term (str): the term in the query

        Returns:
            postings (list): the posting list from the posting file
        """

        if term in self.index:
            self.postings_file.seek(self.index[term][1])
            postings = load(self.postings_file)

        else:
            postings = []

        return postings

    def phrase_2(self, l1: list, l2: list) -> list:
        """
        Finds docs where 2 word phrase appears.

        Parameters:
            l1 (list): first posting list
            l2 (list): second posting list

        Returns:
            intermediate (list): merged list
        """

        p1 = p2 = 0
        intermediate = []

        while p1 < len(l1) and p2 < len(l2):

            d1 = l1[p1][0]
            d2 = l2[p2][0]

            if d1 == d2:
                pos_list = self.pos_phrase_2(l1[p1][2], l2[p2][2])

                if pos_list:
                    intermediate.append([d1, len(pos_list), pos_list])

                p1 += 1
                p2 += 1

            elif d1 < d2:
                p1 += 1

            else:
                p2 += 1

        return intermediate

    def pos_phrase_2(self, l1: list, l2: list) -> list:
        """
        Finds positions in doc where 2 word phrase appears.

        Parameters:
            l1 (list): first posting list
            l2 (list): second posting list

        Returns:
            intermediate (list): merged list
        """

        p1 = p2 = 0
        intermediate = []

        while p1 < len(l1) and p2 < len(l2):

            pos1 = l1[p1]
            pos2 = l2[p2]

            if pos2 - pos1 == 1:
                intermediate.append((pos1, pos2))
                p1 += 1
                p2 += 1

            elif pos1 < pos2:
                p1 += 1

            else:
                p2 += 1

        if intermediate:
            return intermediate
        else:
            return None

    def phrase_3(self, l2: list, l3: list) -> list:
        """
        Finds docs where 3 word phrase appears.

        Parameters:
            l2 (list): first merged posting list from 2 word phrase
            l3 (list): second posting list

        Returns:
            intermediate (list): merged list
        """

        p2 = p3 = 0
        intermediate = []

        while p2 < len(l2) and p3 < len(l3):

            d2 = l2[p2][0]
            d3 = l3[p3][0]

            if d2 == d3:
                pos_list = self.pos_phrase_3(l2[p2][2], l3[p3][2])

                if pos_list:
                    intermediate.append([d2, len(pos_list), pos_list])

                p2 += 1
                p3 += 1

            elif d2 < d3:
                p2 += 1

            else:
                p3 += 1

        return intermediate

    def pos_phrase_3(self, l2: list, l3: list) -> list:
        """
        Finds positions in doc where 3 word phrase appears.

        Parameters:
            l2 (list): first posting list
            l3 (list): second posting list

        Returns:
            intermediate (list): merged list
        """

        p2 = p3 = 0
        intermediate = []

        while p2 < len(l2) and p3 < len(l3):

            pos2 = l2[p2][1]
            pos3 = l3[p3]

            if pos3 - pos2 == 1:
                intermediate.append((l2[p2][0], l2[p2][1], pos3))
                p2 += 1
                p3 += 1

            elif pos2 < pos3:
                p2 += 1

            else:
                p3 += 1

        if intermediate:
            return intermediate
        else:
            return None

    def count_freqs(self, array: list) -> Counter:
        """
        Gets frequency of terms in array

        Parameters:
            array (list): list of terms in query

        Returns:
            temp (Counter): frequncy of terms in query
        """

        temp = [" ".join(i) for i in array]
        return Counter(temp)

    def merge_AND(self, containsAll: list) -> list:
        """
        Runner for the AND operation

        Parameters:
            containsAll (list): list of posting lists to be merged

        Returns:
            l1 (list): the merged posting list
        """

        if len(containsAll) == 0:
            return []

        elif len(containsAll) == 1:
            return containsAll

        else:
            l1 = containsAll.pop(0)

            while True:
                l1 = self.AND(l1, containsAll.pop(0))

                if not containsAll:
                    break

            return l1

    def AND(self, l1: list, l2: list) -> list:
        """
        Performs the AND operation for the 2 lists

        Parameters:
            l1 (list): the first posting list
            l2 (list): the second posting list

        Returns:
            intermediate (list): the merged list
        """
        p1 = p2 = 0
        intermediate = []

        while p1 < len(l1) and p2 < len(l2):

            d1 = l1[p1][0]
            d2 = l2[p2][0]

            if d1 == d2:
                intermediate.append([d1])
                p1 += 1
                p2 += 1

            elif d1 < d2:
                p1 += 1

            else:
                p2 += 1

        return intermediate
