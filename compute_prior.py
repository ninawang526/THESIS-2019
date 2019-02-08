from collections import defaultdict
import pickle
import datetime
import to_lexicon as lex
from base_words import *
from sklearn.feature_extraction.text import CountVectorizer as CV

import numpy as np
import time as time 

def background_frequencies(articles):
	d =  defaultdict(lambda: 0)
	for article in articles:
		text = article.text
		cleaned_text = lex.text_cleanup(text,proper_nouns=LEFT_WORDS+RIGHT_WORDS)
		for w in cleaned_text:
			d[w] += 1
	return d

def cv_frequencies(articles):
	texts = []
	for article in articles:
		text = article.text
		cleaned_text = lex.text_cleanup(text,proper_nouns=LEFT_WORDS+RIGHT_WORDS)
		texts.append((" ").join(cleaned_text))

	cv = CV(decode_error = 'ignore',
			binary = False, ngram_range=(1,3))

	# fit to cv, grab vocabulary, and then sum counts
	text_cv = cv.fit_transform(texts)
	
	text_vocab = cv.vocabulary_
	text_d = np.asarray(text_cv.sum(axis=0))[0]

	# put into usage dictionaries
	d = defaultdict(lambda: 0)
	for phrase,index in text_vocab.items(): 
		count = text_d[index]
		d[phrase] = count
	return d    


if __name__ == '__main__':
	start_time = time.time()
	path = "../ARTICLES-2016/processed/NYT-POL2016-2017-processed/"
	
	start = datetime.datetime(2016, 6, 1)
	end = datetime.datetime(2017, 7, 1)
	
	articles = lex.get_articles_from_filepath(path,start,end)

	print ("finished grabbing articles")

	freqs = cv_frequencies(articles)
	print (type(freqs))

	# ------- ! DUMP ! ------- #
	with open('background_corpus_frequencies_CV.pkl', 'wb') as fp:
		pickle.dump(dict(freqs), fp)

	end_time = time.time()

	print("--- {} seconds ---".format(end_time - start_time))


	# # ------- ! LOAD ! ------- #
	# with open ('background_corpus_frequencies.pkl', 'rb') as fp:
	# 	freqs = pickle.load(fp)

	# for k,v in freqs.items():
	# 	print (k,v)