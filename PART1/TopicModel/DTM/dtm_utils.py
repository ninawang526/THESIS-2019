# take collocations & create a document-matrix
import THESIS2019.utils.to_lexicon as lex
from THESIS2019.utils.base_words import *
from THESIS2019.utils.get_articles import *

import datetime
import pickle
import os, math
import numpy as np
from collections import defaultdict
import subprocess

import gensim
from gensim.models.wrappers import DtmModel
from gensim.models import LdaSeqModel
from gensim.corpora import Dictionary, bleicorpus

# graph change of a topic over time - using Jensen-Shannon Divergence
# https://dit.readthedocs.io/en/latest/measures/divergences/jensen_shannon_divergence.html
    
import dit
from dit.divergences import jensen_shannon_divergence



PATH = "/Users/ninawang/Thesis/remote/THESIS2019/PART1/TopicModel/DTM"

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

# PRE-PROCESS
def get_doc_and_timeslices(col):
    total = [d for d in col if d[0] is not None]
    docs = [d[1] for d in total]
    timeslices = [d[0].year for d in total]
    return docs, timeslices

# alter topic evolution with the parameter "top_chain_var", default=0.005
# doc_set = {"label",[]}
def to_document_dat(doc_set):
	combined = []
	for label,dset in doc_set.items():
		# doc = [d[1] for d in dset] # grabbing the dset
		print ("dset length: %d" %(len(dset)))
		combined += dset

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

	# write timesequence file
	with open("data/doc-seq.dat", "w") as f:
		s = str(len(doc_set.items()))
		for label,dset in doc_set.items():
			s+="\n"
			s+=str(len(dset))
		f.write(s)

		# f.write("2\n"+str(len(dems_docs_2016))+"\n"+str(len(reps_docs_2016)))


# RUN DTM
def run_dtm():
	subprocess.call(["./dtm.sh"])


# POST-PROCESS
def get_path(topic):
    folder = "/data/model_run//lda-seq"
    if topic / 100 >= 1:
        topicnum = str(topic)
    elif topic / 10 >= 1:
        topicnum = "0"+str(topic)
    else:
        topicnum = "00"+str(topic)
    topicpath = "/topic-"+topicnum+"-var-e-log-prob.dat"
    fullpath = PATH+folder+topicpath

    return fullpath


def get_matrix(topic_num):
    matrix = []
    path = get_path(topic_num)
    with open(path, "r") as f:
        for i,line in enumerate(f):
            if i%NUM_TIMESTAMPS==0:
                matrix.append([])
            matrix[-1].append(float(line))

    matrix=np.array(matrix)

    return matrix


# print top words for each topic
def word(idx):
	with open(PATH+"/data/dictionary.pkl","rb") as f:
		dct = pickle.load(f)
	return dct[idx]

# get the words of topic_num from document slice
def get_top_words(topic_num, slice_id):
	def prob(n):
		return math.exp(n)
	matrix = get_matrix(topic_num)
	timeslice = matrix[:,slice_id]
	indexed = [(i,prob(v)) for i,v in enumerate(timeslice)]
	sorted_by_prob = sorted(indexed, key=lambda x:x[1], reverse=True)

	return sorted_by_prob


def topic_over_time(doc_set, topic_num, pr=False):
	distros = {}
	keys = list(doc_set.keys())
	for slice_id in range(len(keys)):
		distro = get_top_words(topic_num, slice_id)
		distros[keys[slice_id]]=distro

	return distros
        

def print_topic(topic, num_words=10):
	for tup in topic[:num_words]:
		print(word(tup[0]),tup[1])
	print("\n")


def write_topic(f, topic, num_words=10):
	for tup in topic[:num_words]:
		f.write("%s %f\n" %(word(tup[0]),tup[1]))
	f.write("\n")


def to_js_distro(dist, topn=50):
	ids = [str(d[0]) for d in dist][:topn]
	probs = [d[1] for d in dist][:topn]
	sumprobs = sum(probs)
	probs = [p/sumprobs for p in probs]

	distro = dit.ScalarDistribution(ids, probs)
	return distro


# for each topic, find the JSD among years for that topic
# for each topic, find the JSD among media outlets for that topic
def highest_js(doc_set, topn=None):
	topics = []
	for i in range(20):	
		distros = topic_over_time(doc_set, i)
		js_distros = [to_js_distro(distro) for key,distro in distros.items()]
		jsd = jensen_shannon_divergence(js_distros)
		topics.append((i, jsd))

	sorted_topics = sorted(topics, key=lambda x:x[1], reverse=True)
	for i,jsd in sorted_topics:
		print("JS Divergence of topic %d: %f"%(i, jsd))
	jsds = np.array([jsd for i,jsd in sorted_topics])
	print("mean divergence: %f" %(np.mean(jsds)))
	print("std divergence: %f" %(np.mean(jsds)))
	if topn is not None:
		return sorted_topics[:topn]
	return sorted_topics


### VISUALIZE
def to_dict(dist):
	dct = defaultdict(lambda: 0)
	for idx,prob in dist:
		dct[word(idx)]=prob
	return dct


def graph_topic_over_time(topic):
	t0, t1, t2 = topic_over_time(topic)
	print ("here!")

	# data to plot
	words = list(set([word(t[0]) for t in t0[:50]+t1[:50]+t2[:50]]))
	dct_means0 = to_dict(t0)
	dct_means1 = to_dict(t1)
	dct_means2 = to_dict(t2)

	means0 = [dct_means0[w] for w in words]
	means1 = [dct_means1[w] for w in words]
	means2 = [dct_means2[w] for w in words]

	n_groups = len(words)

	# create plot
	fig, ax = plt.subplots(figsize=(20, 10))
	index = np.arange(n_groups)
	bar_width = 0.35
	opacity = 0.8

	rects0 = plt.bar(index, means0, bar_width,
	alpha=opacity,
	color='b',
	label='2008')

	rects1 = plt.bar(index + bar_width, means1, bar_width,
	alpha=opacity,
	color='g',
	label='2012')

	rects2 = plt.bar(index + 2*bar_width, means2, bar_width,
	alpha=opacity,
	color='r',
	label='2016')

	plt.xlabel('Year')
	plt.ylabel('Scores')
	plt.title('Scores by Year')
	plt.xticks(index + bar_width, words, rotation=45)
	plt.legend()
	plt.show()


# track word of interest over time
def graph_words_of_interest(topic, keywords):
    t0, t1, t2 = topic_over_time(topic)
    
    # data to plot
    words = list(set([word(t[0]) for t in t0[:50]+t1[:50]+t2[:50]]))
    dct_means0 = to_dict(t0)
    dct_means1 = to_dict(t1)
    dct_means2 = to_dict(t2)
    
    probs = defaultdict(lambda: [])
    for kw in keywords:
        for dct in [dct_means0,dct_means1,dct_means2]:
            probs[kw].append(dct[kw])
            
    print(probs)

    x = ["2008","2012","2016"]
    plt.figure()
    plt.subplot()
    plt.plot(x, probs["american"], 'b')

    plt.subplot()
    plt.plot(x, probs["first"], 'r')
    plt.show()



if __name__ == '__main__':
	
	datapath = "/Users/ninawang/Thesis/remote/THESIS2019/example_data_1000/"
	outlets = ['BREITBART','NATIONALREVIEW','FOX',
				'WASHINGTONEXAMINER','REUTERS','NPR',
				'NYT', 'MSN','CNN','SLATE']
	articles = get_articles_outlets(datapath,outlets,2012)

	# doc_set = {}
	# for label, arts in articles.items():
	# 	print(label, len(arts))
	# 	filt, docs =lex.get_topicmodel_docs(arts, LEFT_WORDS)
	# 	doc_set[label]=docs

	# load
	doc_set = {}
	for label, arts in articles.items():
		with open("../LDA/docs/"+label+"_docs2012.pkl","rb") as f:
			docs = pickle.load(f)
		print(label, len(docs))
		doc_set[label]=docs

	global NUM_TIMESTAMPS
	NUM_TIMESTAMPS = len(doc_set.keys())

	# to_document_dat(doc_set)
	# print("running dtm...")
	# run_dtm()
	# print("finished dtm...")
	maxtopics = highest_js(doc_set,topn=20)

	with open("dtm_output.txt","w") as f:
		for i,jsd in maxtopics:
			f.write("topic %d\n" %i)
			tps = topic_over_time(doc_set, i, pr=True)
			for key,distro in tps.items():
				f.write("distro from %s\n" %(key))
				write_topic(f,distro)





