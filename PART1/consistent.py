from THESIS2019.PART1.TopicModel.LDA import lda_consistency as ldac

import THESIS2019.utils.to_lexicon as lex
from THESIS2019.utils.base_words import *

import pickle
import datetime





# consistency of LDA topic modeling, measured by Jensen-Shannon Divergence
def LDA_consistent():
	path = "/Users/ninawang/Thesis/remote/THESIS2019/example_data/NYT-OPINION2012-2013-processed/"

	start = datetime.datetime(2012, 6, 1)
	end = datetime.datetime(2013, 7, 1)

	articles = lex.get_articles_from_filepath(path,start,end)

	# articles, dictionary, corpus, model = lex.topic_model(articles, LEFT_WORDS+RIGHT_WORDS,num_topics=20)

	# LOAD
	# path = "/Users/ninawang/Thesis/remote/THESIS2019/Jupyter Notebooks/"

	path = "/Users/ninawang/Thesis/remote/THESIS2019/file_transfer/"

	with open(path+"both_tm.pkl","rb") as f:
		model = pickle.load(f)
	with open(path+"both_dict.pkl","rb") as f:
		dictionary = pickle.load(f)
	with open(path+"both_articles.pkl","rb") as f:
		articles = pickle.load(f)
	with open(path+"both_corpus.pkl","rb") as f:
		corpus = pickle.load(f)

	arts = {"2016":[],"2017":[]}
	for i,article in enumerate(articles):
		if article.date_publish is not None:
			arts[str(article.date_publish.year)].append(corpus[i])

	dist1 = ldac.total_topic_distribution(model, arts["2016"])
	dist2 = ldac.total_topic_distribution(model, arts["2017"])

	jsd = ldac.compare_topic_distributions(dist1, dist1)
	print("JS Divergence of topic: %f"%(jsd))


if __name__ == '__main__':
	LDA_consistent()






