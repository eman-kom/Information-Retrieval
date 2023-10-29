from collections import Counter
from math import log10
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pickle import load
from string import ascii_lowercase
import heapq


class Ranked:

    def __init__(self, q, r, i, d, p):
        self.docs = d
        self.index = i
        self.postings_file = p
        self.query = q
        self.relevant_docs = r

    def execute(self) -> list:
        """
        Executes the ranked retrieval.

        Parameters:
            None

        Returns:
            new_heap (list): list of docIDs sorted from most similar to least similar.
        """
        freqs = self.process_query()
        first_heap = self.cosine_similarity(freqs)

        for _, _, docID in heapq.nlargest(20, first_heap):
            self.relevant_docs.append(docID)

        new_freqs = self.rocchio(freqs)
        new_heap = self.cosine_similarity(new_freqs)

        return new_heap

    def process_query(self) -> Counter:
        """
        Parses the query

        Parameters:
            None

        Returns:
            query (Counter): the frequency of each term in the query
        """

        stop_words = set(stopwords.words('english'))
        stop_words.update(list(ascii_lowercase))
        porter = PorterStemmer()

        query = self.query.strip().split()
        query = filter(lambda term: term not in stop_words, query)
        query = map(lambda term: porter.stem(term), query)

        return Counter(query)

    def cosine_similarity(self, freqs: Counter) -> list:
        """
        Finds and sorts the cosine similarity between the query and the docs.

        Parameters:
            freqs (Counter): frequency of each term in the query

        Returns:
            heap (list): the docIDs sorted by most similar to least similar
        """

        heap = []
        scores = {}

        for term, tf in freqs.items():

            if term in self.index:
                self.postings_file.seek(self.index[term][1])
                postings = load(self.postings_file)
                idf = self.index[term][0]

            else:
                postings = []
                idf = 0

            q_weight = (1 + log10(tf)) * idf

            for docID, d_weight, _ in postings:
                scores[docID] = scores.get(docID, 0) + q_weight * d_weight

        for docID, dot_product in scores.items():
            cos_sim = dot_product / self.docs[docID][0]
            heapq.heappush(heap, (cos_sim, -docID, docID))

        return heap

    def rocchio(self, og_freqs: Counter) -> Counter:
        """
        Performs the rocchio algorithm

        Parameters:
            og_freqs (Counter): frequency of terms in original query

        Returns:
            (Counter): new freuqency of terms with new terms added
        """
        relevant_vector = Counter()
        beta = 0.2

        for docID in self.relevant_docs:
            self.postings_file.seek(self.docs[docID][1])
            doc_freqs = load(self.postings_file)
            relevant_vector += doc_freqs

        for term, freq in relevant_vector.items():
            relevant_vector[term] = freq / len(self.relevant_docs) * beta

        for term, freq in og_freqs.items():
            og_freqs[term] = freq * (1 - beta)
            og_freqs[term] += relevant_vector.pop(term, 0)

        temp = relevant_vector.most_common(2000)
        return og_freqs + Counter(dict(temp))
