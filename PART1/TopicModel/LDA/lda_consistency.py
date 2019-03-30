import pickle
import datetime

import THESIS2019.utils.to_lexicon as lex
from THESIS2019.utils.base_words import *

# import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
# import matplotlib.pyplot as plt
from collections import defaultdict

import dit
from dit.divergences import jensen_shannon_divergence



# get the topic distribution for a document set (corpus)
def total_topic_distribution(model, corpus):
    dist = defaultdict(lambda:0)
    for doc in corpus:
        topics = model[doc]
        for topic,prob in topics:
            dist[topic] += prob    
    # normalize
    s = sum(dist.values())
    for d in dist:
        dist[d] /= s
    return dist


def to_js_distro(distro):
    ids, probs = list(distro.keys()), list(distro.values())
    js_distro = dit.ScalarDistribution(ids, probs)
    return js_distro


def compare_topic_distributions(distros):
	js_distros = []
	for distro in distros:
		js_distros.append(to_js_distro(distro))
	return jensen_shannon_divergence(js_distros)
    

def graph_distribution(dist):
    x, y = dist.keys(), dist.values()
    plt.bar(x, y, alpha=0.5)
    plt.xlabel('Topic number')
    plt.ylabel('Proportion in document set')
    plt.title('Topic distribution')

    plt.show()




if __name__ == '__main__':
	path = "/Users/ninawang/Thesis/remote/THESIS2019/example_data/NYT-OPINION2012-2013-processed/"
	start = datetime.datetime(2012, 6, 1)
	end = datetime.datetime(2013, 7, 1)
	articles = lex.get_articles_from_filepath(path,start,end)

	# articles, dictionary, corpus, model = lex.topic_model(articles, LEFT_WORDS+RIGHT_WORDS,num_topics=20)

	# LOAD
	path = "/Users/ninawang/Thesis/remote/THESIS2019/Jupyter Notebooks/"
	with open(path+"prelim_model2012.pkl","rb") as f:
		model = pickle.load(f)
	with open(path+"prelim_dictionary2012.pkl","rb") as f:
		dictionary = pickle.load(f)
	with open(path+"prelim_articles2012.pkl","rb") as f:
		articles = pickle.load(f)
	with open(path+"prelim_corpus2012.pkl","rb") as f:
		corpus = pickle.load(f)

	arts = {"2012":[],"2013":[]}
	for i,article in enumerate(articles):
		if article.date_publish is not None:
			arts[str(article.date_publish.year)].append(corpus[i])

	dist2012 = total_topic_distribution(model, arts["2012"])
	dist2013 = total_topic_distribution(model, arts["2013"])

	jsd = compare_topic_distributions(dist2012, dist2013)
	print("JS Divergence of topic: %f"%(jsd))





