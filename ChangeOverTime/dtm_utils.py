# take collocations & create a document-matrix
# def create_documents(col):


# import sys
# p = "/Users/ninawang/Thesis/remote/THESIS2019/"
# if p not in sys.path:
# 	sys.path.append(p)
# 	sys.path.append("/usr/local/lib/python3.7/site-packages")

# print (sys.path)

import to_lexicon as lex
from base_words import *
import datetime
import pickle

import gensim
from gensim.models.wrappers import DtmModel
from gensim.models import LdaSeqModel
from gensim.corpora import Dictionary, bleicorpus
import os, numpy

# slow python-wrapped model
def ldaseq():
	print("starting")

	path = "/n/fs/thesis-ninaw/ARTICLES-2012/processed/NYT-OPINION2012-2013-processed/"

	start = datetime.datetime(2012, 6, 1)
	end = datetime.datetime(2013, 7, 1)

	articles2012 = lex.get_articles_from_filepath(path,start,end)

	path = "/n/fs/thesis-ninaw/ARTICLES-2016/processed/NYT-OPINION2016-2017-processed/"

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


def get_doc_and_timeslices(col):
    total = [d for d in col if d[0] is not None]
    docs = [d[1] for d in total]
    timeslices = [d[0].year for d in total]
    return docs, timeslices

# alter topic evolutionwith the parameter "top_chain_var", default=0.005
def to_document_dat():
	# load 2008
	with open("../Jupyter Notebooks/2008-collocations-dem.pkl", "rb") as f:
		dem2008 = pickle.load(f)
	with open("../Jupyter Notebooks/2008-collocations-rep.pkl", "rb") as f:
		rep2008 = pickle.load(f)

	# load 2012
	with open("../Jupyter Notebooks/2012-collocations-dem.pkl", "rb") as f:
		dem2012 = pickle.load(f)
	with open("../Jupyter Notebooks/2012-collocations-rep.pkl", "rb") as f:
		rep2012 = pickle.load(f)

	# load 2016
	with open("../Jupyter Notebooks/2016-collocations-dem.pkl", "rb") as f:
		dem2016 = pickle.load(f)
	with open("../Jupyter Notebooks/2016-collocations-rep.pkl", "rb") as f:
		rep2016 = pickle.load(f)

	dems_docs_2008 = [d[1] for d in dem2008]
	dems_docs_2012 = [d[1] for d in dem2012]
	dems_docs_2016 = [d[1] for d in dem2016]
	combined = dems_docs_2008+dems_docs_2012+dems_docs_2016

	print("2008:%d\n2012:%d\n2016:%d\n"%(len(dems_docs_2008),len(dems_docs_2012),len(dems_docs_2016)))
	print("total:%d\n"%(len(combined)))

	dct = Dictionary(combined)
	
	# write dictionary
	with open("data/dictionary.pkl","wb") as f:
		pickle.dump(dct,f)

	# write document file
	with open("data/doc-mult.dat", "w") as f:
		for text in combined:
			bow = dct.doc2bow(text)
			row=str(len(bow))+" "+(" ").join([str(idx)+":"+str(c) for idx,c in bow])+"\n"
			f.write(row)

	# write timestamp file
	with open("data/doc-seq.dat", "w") as f:
		f.write("3\n"+str(len(dems_docs_2008))+"\n"+str(len(dems_docs_2012))+"\n"+str(len(dems_docs_2016)))










if __name__ == '__main__':
	# ldaseq()
	to_document_dat()





