#!/usr/bin/python

import gensim
from nltk.tokenize import word_tokenize

#documents
raw_documents = ["I'm taking the show on the road.",
                 "My socks are a force multiplier.",
             "I am the barber who cuts everyone's hair who doesn't cut their own.",
             "Legend has it that the mind is a mad monkey.",
            "I make my own fun."]

#A document will now be a list of tokens.
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
sims = gensim.similarities.Similarity('/var/www/html/',tf_idf[corpus],
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