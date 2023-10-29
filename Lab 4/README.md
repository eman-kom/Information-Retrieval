== General Notes about this assignment ==

Indexing:
Index.py reads the csv file. For each row in the csv file the program will expand
contractions, remove stop words, tokenize and stem the title, content and the 
court and concatenate them into a single list. From there, it will process the tokens 
to find the position and frequency of each token within the document. Next, it will find 
the length of the document. Then, it will add the information obtained into the postings 
list and the docslist respectively. Finally, it will output the postings list and the
frequencies of each term within the doc into the postings file and output the index and 
docs dictionary into the dictionary file.

Searching:
Search.py first initialise the index and docs dictionary from the dictionary file. Next,
it reads the query file to process the query and get its marked relevant documents. Next,
it opens the postings file and runs the search. If the query contains '"' or "AND", it will
be stripped away to force the query to be a free text query.

Ranked retrieval is executed in ranked_retrieval.py. It takes the query, remove stop words, 
tokenize and stem and finds its frequencies. Then it does a first pass of cosine similarity 
to get the initial relevant documents. Next, it gets the top 20 documents from the list and 
appends them to the list of relevant documents that was obtained from the query file. From 
there, the vectors associated with those documents are obtained and its vector addition is 
performed on the elements to combine those vectors. Each element in the combined vector is 
divided by the number of relevant documents and multiplied by a beta of 0.2. Next, for 
each query frequency in the original query, it is multiplied by (1 - beta) and its corresponding 
weight in the combined vector is added to it. Next, the top 2000 common words are then 
added into the query vector. Ranked retrieval is performed again with the new query.

Finally, the docIDs that was obtained from the new query is sorted by its relevancy and
written into the results file.

Boolean Retrieval:
Originally there were plans to also do boolean retrieval, but however when testing it with
the scoreboard, its results were worse than the baseline. But how it works is that it first
splits the query by '"' and 'AND'. Then, each term is stemmed, tokenized and counted into
a Counter class. For each term in the Counter, its array length is determined. If there is
only 1 element (1 word only), the associated posting list is obtained and appended into a 
master list. If there is 2 elements in the array (2 words only), the associated postings 
list are merged only if the words are within 1 space apart and added into the master list.
If there are 3 elements in the array (3 words only), the first 2 associated postings list 
are merged only if the words are within 1 space apart. From the first merged list, it will 
also be merged with the 3rd term where the same rules apply and added into the master list.
For each of the cases (1,2,3 words), the log tf of the each of the merged lists are 
obtained. Next, all the postings lists in the master list will be merged together in an 
AND operation and flattened into a single list. Next, the cosine similarity between the 
query and the processed postings list is obtained and the docIDs, will be ranked from its
most similar to its least similar.

== Files included with this submission ==

List the files in your submission here and provide a short 1 line
description of each file.  Make sure your submission's files are named
and formatted correctly.


index.py        : Parses the dataset.csv into its postings list while finding the idf
                  and lengths of the corresponding documents and the position of the each term 
		  within the file
search.py       : Performs the searching of the terms from the query file.
README.txt      : This file.
postings.txt    : A pickle file containing the posting lists of the terms, the position of
		  each term in the csv dataset and the document vector.
dictionary.txt  : A file containing a pickled dictionary of the terms and its corresponding
                  idfs and a pickled dictionary of the docIDs and its corresponding lengths.
ranked_retrieval.py  : Performs ranked retrieval, cosine similarity and rocchio formula
boolean_retrieval.py : Performs boolean retrieval and cosine similarity to rank the documents
