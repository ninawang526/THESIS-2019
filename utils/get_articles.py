from newsplease import NewsPlease
import datetime 
import os, re
import pickle
import time

from THESIS2019.utils.base_words import *


## helper method to filter articles by date & length
def get_relevant_articles(articles, start, end, filter_date):
	filtered_articles = []
	for article in articles:
		if article.date_publish is not None:
			if filter_date:
				if article.date_publish < start or article.date_publish >= end:
					continue
		if article.text is None:
			continue
		elif len(article.text.split()) <= 100:
			continue 
		filtered_articles.append(article)
	return filtered_articles


## helper method to retrieve articles from path
def get_articles_from_filepath(path, start, end, filter_date):
	articles = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		for filename in filenames:
			if filename.endswith(".pkl"):
				filepath = os.path.join(dirpath, filename)
				with open(filepath, 'rb') as input_file:
					e = pickle.load(input_file)
					articles.append(e)
	filtered_articles = get_relevant_articles(articles, start, end, filter_date)
	return filtered_articles


def get_articles_year(path, year, filter_date):
	if not os.path.exists(path):
		raise ValueError("path " + path + " does not exist.")
	start = datetime.datetime(year, 1, 1)
	end = datetime.datetime(year+1, 12, 1)
	return get_articles_from_filepath(path,start,end,filter_date)


## gathers all articles containing relevant keyword
def get_articles_by_keywords(articles, keywords):
    needles = set([s.lower() for s in keywords])
    relevant_docs = []
    
    # iterate over documents
    for i in range(len(articles)):
        article = articles[i]
        words = text_cleanup(article.text, filter_support=True)
        haystack = set([s.lower() for s in words])

        if len(haystack.intersection(needles)) > 0:
            filt_words = [w for w in words if w.lower() not in needles]
            relevant_docs.append(article)

    return relevant_docs


def get_articles_outlets(datapath, outlets, year, filter_date=True):
	articles = {}
	for outlet in outlets:
		filename = outlet+str(year)+"-"+str(year+1)#+"-processed"
		path = datapath + filename #+ "-processed/"

		print("getting %s"%(outlet))
		start_time = time.time()

		outlet_articles = get_articles_year(path, year, filter_date)

		print("--- %s seconds for %d articles ---" % (time.time() - start_time, len(outlet_articles)))
	
		articles[filename] = outlet_articles
	return articles








