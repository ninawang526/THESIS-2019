# take collocations & create a document-matrix
# def create_documents(col):


import sys
p = "/Users/ninawang/Thesis/remote"
if p not in sys.path:
	sys.path.append(p)
	sys.path.append("/usr/local/lib/python3.7/site-packages")

print (sys.path)



import THESIS2019.to_lexicon as lex
from THESIS2019.base_words import *
import datetime
import pickle

import gensim
from gensim.models.wrappers import DtmModel
from gensim.models import LdaSeqModel
from gensim.corpora import Dictionary, bleicorpus
import os, numpy

def ldaseq():
	path = "/Users/ninawang/Thesis/remote/THESIS2019/NYT-OPINION2012-2013-processed/"

	start = datetime.datetime(2012, 6, 1)
	end = datetime.datetime(2013, 7, 1)

	articles2012 = lex.get_articles_from_filepath(path,start,end)

	path = "/Users/ninawang/Thesis/remote/THESIS2019/NYT-OPINION2016-2017-processed/"

	start = datetime.datetime(2016, 6, 1)
	end = datetime.datetime(2017, 7, 1)

	articles2016 = lex.get_articles_from_filepath(path,start,end)

	print("Gathered %d articles from 2012" %len(articles2012))
	print("Gathered %d articles from 2016" %len(articles2016))

	dem2012, rep2012 = lex.get_collocations(articles2012, LEFT_WORDS, RIGHT_WORDS)
	dem2016, rep2016 = lex.get_collocations(articles2016, LEFT_WORDS, RIGHT_WORDS)

	def get_doc_and_timeslices(col):
	    total = [d for d in col if d[0] is not None]
	    docs = [d[1] for d in total]
	    timeslices = [d[0].year for d in total]
	    return docs, timeslices

	dems_docs_2012, dems_timeslices_2012 = get_doc_and_timeslices(dem2012)
	dems_docs_2016, dems_timeslices_2016 = get_doc_and_timeslices(dem2016)
	combined = dems_docs_2012 + dems_docs_2016

	dct = Dictionary(combined)
	docs = []
	for text in combined:
	    docs.append(dct.doc2bow(text))
	    
	my_corpus = docs
	my_timeslices = [len(dems_docs_2012),len(dems_docs_2016)]
	exe = "../dtm-darwin64"

	import time
	start_time = time.time()
	print(start_time)

	ldaseq = LdaSeqModel(corpus=my_corpus, id2word=dct, time_slice=my_timeslices, num_topics=15)

	print("--- %s seconds ---" % (time.time() - start_time))

	with open("ldaseqmodel.pkl", "wb") as f:
		pickle.dump(ldaseq, f)


if __name__ == '__main__':
	ldaseq()