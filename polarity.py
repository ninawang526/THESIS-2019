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


def get_most_polar(left_ngrams_article, right_ngrams_article):
	def filt(p):
		filtlist = ["columnist", "thank","editor","editori","newslett","stori","click","product","contribut"]
		for w in filtlist:
			if w in p:
				return False
		return True

	l = defaultdict(lambda: 0)
	for p,c in left_ngrams_article.most_common():
		l[p] = c
	
	r = defaultdict(lambda: 0)
	for p,c in right_ngrams_article.most_common():
		r[p] = c
	    
	TL = sum([count for ngram,count in l.items()])
	TR = sum([count for ngram,count in r.items()])

	# put into usage dictionaries
	l = defaultdict(lambda: 0)
	for p,c in left_ngrams_article.most_common():
		if filt(p) and c < .00039*TL:
			l[p] = c
	    
	r = defaultdict(lambda: 0)
	for p,c in right_ngrams_article.most_common():
		if filt(p) and c < .00039*TR:
			r[p] = c

	# gathered pearson's stats for all trigrams
	pearson_scores = []
	all_ngrams = set(list(l.keys())+list(r.keys()))

	for trigram in all_ngrams:
		chi2 = pearson(trigram, l, r, TL, TR)
		pearson_scores.append((trigram, chi2))

	# sorted all trigrams in order of most likely to be polar
	sorted_tot = sorted(pearson_scores, key=lambda x:x[1],reverse=True)
	for i in sorted_tot[:150]:
		print("{}.{}.{}\t\t".format(i[0][0],i[0][1],i[0][2]), i[1])

	total_ngrams = sorted_tot[:10000]

	return total_ngrams


def get_polarity_score_bayesian():
	# get collocations
	with open ('left_collocations_article.pkl', 'rb') as fp:
		left_collocations_article = pickle.load(fp)
	with open ('right_collocations_article.pkl', 'rb') as fp:
		right_collocations_article = pickle.load(fp)

	# turn collocations into ngrams
	ll, left_ngrams_article, left_ngrams_by_article = collocations_to_ngrams(left_collocations_article, 3)
	rl, right_ngrams_article, right_ngrams_by_article = collocations_to_ngrams(right_collocations_article, 3)

	# sort ngrams by most likely to be polar, according to pearson's statistic
	total_ngrams = get_most_polar(left_ngrams_article, right_ngrams_article) 
	
	# set up bayesian comparison
	v = ["{} {} {}".format(phr[0],phr[1], phr[2]) for phr, c in total_ngrams]
	left = ["{} {} {}".format(phr[0],phr[1], phr[2]) for phr in ll]
	right = ["{} {} {}".format(phr[0],phr[1], phr[2]) for phr in rl]

	cv = CV(decode_error = 'ignore', vocabulary=v, ngram_range=(1,3))

	r = fw.bayes_compare_language(left,right,cv=cv)
	print (r[:5])

	for i in sorted(r,key=lambda x:x[1]):
		phr, score = i[0].split(" "),i[1]
		#print ("{}\t\t{}".format(phr[0],score))
		print ("{}.{}.{}\t\t{}".format(phr[0],phr[1], phr[2],score))


	print (len(total_ngrams))



if __name__ == '__main__':
	get_polarity_score_bayesian()




