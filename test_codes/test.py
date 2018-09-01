#!/usr/bin/python

from rake_nltk import Rake
from nltk.corpus import stopwords 
r = Rake() 

f=open("data/data2.txt", "r")
if f.mode == 'r':
	contents =f.read()


a=r.extract_keywords_from_text(contents)
b=r.get_ranked_phrases()
c=r.get_ranked_phrases_with_scores()
print(b)
print("\n\n\n\n")
print(c)
