#!/usr/bin/python

import gensim
import glob
import os

from nltk.tokenize import word_tokenize
from rake_nltk import Rake
from nltk.corpus import stopwords 

r = Rake() 


file_list = glob.glob(os.path.join(os.getcwd(), "data/", "*.txt"))

raw_documents = []

for file_path in file_list:
    with open(file_path) as f_input:
		#te = f_input.read()
		#a=r.extract_keywords_from_text(te)
		#b=r.get_ranked_phrases()
		#converted = " ".join(str(x) for x in b)
		#raw_documents.append(converted)

		raw_documents.append(f_input.read())



gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in raw_documents]


#create a dictionary from a list of documents. A dictionary maps every word to a number.
dictionary = gensim.corpora.Dictionary(gen_docs)

#reate a corpus. A corpus is a list of bags of words. A bag-of-words representation for a document just lists the number of times each word occurs in the document.
corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

#create a tf-idf model from the corpus
tf_idf = gensim.models.TfidfModel(corpus)
#print(tf_idf)
s = 0
for i in corpus:
    s += len(i)
#print(s)


#create a similarity measure object in tf-idf space.
sims = gensim.similarities.Similarity('/var/www/html/Auto-tagging-of-Theses-and-Dissertations-of-Bicol-University-Searching-and-Matching-/result/result',tf_idf[corpus],
                                      num_features=len(dictionary))


#create a query document and convert it to tf-idf.
query_doc = [w.lower() for w in word_tokenize("I'm taking the show on the road.")]
#print(query_doc)
query_doc_bow = dictionary.doc2bow(query_doc)
#print(query_doc_bow)
query_doc_tf_idf = tf_idf[query_doc_bow]
#print(query_doc_tf_idf)



#show an array of document similarities to query. We see that the second document is the most similar with the overlapping of socks and force.
print(sims[query_doc_tf_idf])
