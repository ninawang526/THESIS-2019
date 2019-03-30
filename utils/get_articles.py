from newsplease import NewsPlease
import datetime, os, re
import pickle
import time

from THESIS2019.utils.base_words import *


## helper method to filter articles by date & length
def get_relevant_articles(articles, start, end):
	filtered_articles = []
	for article in articles:
		if article.date_publish is not None:
			if article.date_publish < start or article.date_publish >= end:
				continue
		if article.text is None:
			continue
		elif len(article.text.split()) <= 100:
			continue 
		filtered_articles.append(article)
	return filtered_articles


## helper method to retrieve articles from path
def get_articles_from_filepath(path, start, end):
	articles = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		for filename in filenames:
			if filename.endswith(".pkl"):
				filepath = os.path.join(dirpath, filename)
				with open(filepath, 'rb') as input_file:
					e = pickle.load(input_file)
					articles.append(e)
	filtered_articles = get_relevant_articles(articles, start, end)
	return filtered_articles


def get_articles_year(path, year):
	if not os.path.exists(path):
		raise ValueError("path " + path + " does not exist.")
	start = datetime.datetime(year, 6, 1)
	end = datetime.datetime(year+1, 7, 1)
	return get_articles_from_filepath(path,start,end)


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


def get_articles_outlets(datapath, outlets, year):
	articles = {}
	for outlet in outlets:
		filename = outlet+str(year)+"-"+str(year+1)
		path = datapath + filename + "-processed/"
		outlet_articles = get_articles_year(path, year)
		articles[filename] = outlet_articles
	return articles








