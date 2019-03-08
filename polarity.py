import datetime

from newsplease import NewsPlease
import to_lexicon as lex
from base_words import *

import pickle
from collections import defaultdict

import numpy as np

from collections import Counter
from nltk import ngrams
from nltk.stem.porter import *

import FightinWords as fw
from sklearn.feature_extraction.text import CountVectorizer as CV



# collocations by article -> ngrams 
def collocations_to_ngrams(collocations, n):
	ngrams_list = []
	ngrams_counter = Counter([])
	ngrams_by_article = []
	for col in collocations:
		d = Counter(ngrams(col, n))
		ngrams_counter += d
		ngrams_by_article.append(d)
		ngrams_list += (list(ngrams(col, n)))

	return ngrams_list, ngrams_counter, ngrams_by_article


def pearson(p, r, d, tr, td):
	# freq of trigram p used by a member of party k
	def f(p, k):
		return k[p]
	    
	# freq of all phrases used by a member of party k excluding phrase p
	def fnot(p, k, total):
		return total - k[p]

	def sq(num):
		return num*num

	num = sq((f(p,r)*fnot(p,d,td)) - (f(p,d)*fnot(p,r,tr)))
	denom = (f(p,r)+f(p,d))*(f(p,r)+fnot(p,r,tr))*(f(p,d)+fnot(p,d,td))*(fnot(p,r,tr)+fnot(p,d,td))
	return float(num)/denom


# takes in the document frequency matrix for left & right
def get_most_polar(left_collocations, right_collocations):
	def filt(p):
		filtlist = ["columnist", "thank","editor","editori","newslett","stori","click","product","contribut"]
		for w in filtlist:
			if w in p:
				return False
		return True
	    
	cv = CV(decode_error = 'ignore',
			binary = False, ngram_range=(1,3))

	# fit to cv, grab vocabulary, and then sum counts
	left_cv = cv.fit_transform(left_collocations)
	
	left_vocab = cv.vocabulary_
	left_d = np.asarray(left_cv.sum(axis=0))[0]


	right_cv = cv.fit_transform(right_collocations)	
	
	right_vocab = cv.vocabulary_
	right_d = np.asarray(right_cv.sum(axis=0))[0]


	# counts of total usages
	TL = sum(left_d)
	TR = sum(right_d)

	# put into usage dictionaries
	l = defaultdict(lambda: 0)
	for phrase,index in left_vocab.items(): 
		count = left_d[index]
		if filt(phrase) and count < .00039*TL:
			l[phrase] = count
	    
	r = defaultdict(lambda: 0)
	for phrase,index in right_vocab.items(): 
		count = right_d[index]
		if filt(phrase) and count < .00039*TR:
			r[phrase] = count

	# gathered pearson's stats for all trigrams
	pearson_scores = []
	all_ngrams = set(list(l.keys())+list(r.keys()))

	for trigram in all_ngrams:
		chi2 = pearson(trigram, l, r, TL, TR)
		pearson_scores.append((trigram, chi2))

	# sorted all trigrams in order of most likely to be polar
	# print ("PRINTING MOST POLAR")
	sorted_tot = sorted(pearson_scores, key=lambda x:x[1],reverse=True)
	# for i in sorted_tot[:100]:
	# 	print("{}\t\t".format(i[0]), i[1])

	total_ngrams = sorted_tot[:10000]

	return total_ngrams, l,r


def get_polarity_score_bayesian():
	# get collocations
	with open ('left_collocations_article.pkl', 'rb') as fp:
		left_collocations_article = pickle.load(fp)
	with open ('right_collocations_article.pkl', 'rb') as fp:
		right_collocations_article = pickle.load(fp)

	# all grams!
	left = []
	for article in left_collocations_article:
		left.append((" ").join(article))

	right = []
	for article in right_collocations_article:
		right.append((" ").join(article))

	# sort ngrams by most likely to be polar, according to pearson's statistic
	total_ngrams, l_dict, r_dict = get_most_polar(left, right) 

	# ------- ! LOAD ! ------- #
	with open ('background_corpus_frequencies_CV.pkl', 'rb') as fp:
		background_freqs = pickle.load(fp)

	v = [phr for phr, c in total_ngrams]

	total = sum([v for k,v in background_freqs.items()])

	prior = []
	for wd in v:
		try:
			y_i = background_freqs[wd]
			p = 2*y_i #10*(float(y_i)/total)
		except:
			p = 2*1 #10*(1./total)
		prior.append(p)
	prior = np.array(prior)


	cv = CV(decode_error = 'ignore', ngram_range=(1,3),vocabulary=v)

	r = fw.bayes_compare_language(left,right,prior=prior,cv=cv)
	print (r[:5])


	sorted_results = sorted(r,key=lambda x:x[1])
	print_results(sorted_results)

	total_counts = Counter(l_dict) + Counter(r_dict)
	return sorted_results, total_counts


def print_results(results):
	print ("PRINTING - most right")
	for i in range(len(results)):
		obj = results[i]
		phr, score = obj[0],obj[1]
		print ("{}\t\t{}".format(phr,score))
		# print ("{}.{}\t\t{}".format(phr[0],phr[1],score))
		# print ("{}.{}.{}\t\t{}".format(phr[0],phr[1], phr[2],score))
		if i == 100:
			break

	print ("PRINTING - most left")
	for i in range(len(results)-1,-1,-1):
		obj = results[i]
		phr, score = obj[0],obj[1]
		print ("{}\t\t{}".format(phr,score))
		# print ("{}.{}\t\t{}".format(phr[0],phr[1],score))
		# print ("{}.{}.{}\t\t{}".format(phr[0],phr[1], phr[2],score))
		if i == len(results)- 100:
			break


if __name__ == '__main__':
	res = get_polarity_score_bayesian()




