import csv
import datetime

from THESIS2019.utils import to_lexicon as lex
from THESIS2019.utils.base_words import *
from THESIS2019.utils.get_articles import *

from THESIS2019.PART1.TopicModel.STM.r_stm import run_stm


# split article set into x and y
# determined by whether keyword is in the processed article
def split_articles(articles, x_keywords, y_keywords): 
	x_needles = lex.needles_cleanup(x_keywords)
	y_needles = lex.needles_cleanup(y_keywords) 

	proper_nouns = [wd for wd in (x_keywords+y_keywords) if wd[0].isupper()]

	x_articles= []
	y_articles = []

	both, left, right, none = 0, 0, 0, 0

	for article in articles:
		title, date_publish, text = article.title, article.date_publish, article.text
		words = lex.text_cleanup(text, filter_support=True, proper_nouns=proper_nouns, filter_banned=True) # filter & tokenize words
		
		wordset = set(words)
		
		needles = x_needles.union(y_needles)
		words = [w for w in words if w not in needles]

		if len(wordset.intersection(x_needles)) > 0:
			x_articles.append((article, words))
		if len(wordset.intersection(y_needles)) > 0:
			y_articles.append((article, words))

		if len(wordset.intersection(x_needles)) > 0 and len(wordset.intersection(y_needles)) > 0:
			both+=1
		elif len(wordset.intersection(x_needles)) > 0:
			left+=1
		elif len(wordset.intersection(y_needles)) > 0:
			right+=1
		else:
			none+=1

	# print ("both: %d" %(both))
	# print ("left: %d" %(left))
	# print ("right: %d" %(right))
	# print ("none: %d" %(none))

	return x_articles, y_articles



# determine whether articles are right or left 
# if both, put in both
def to_csv_leaning(articles, outlet, x_keywords, y_keywords, csvwriter):
	x_articles, y_articles = split_articles(articles, x_keywords, y_keywords)
	print("len x = %d, len y = %d" %(len(x_articles),len(y_articles)))

	for article,bow in x_articles:
		title, date = article.title.strip(), article.date_publish
		bow = (" ").join(bow)
		csvwriter.writerow([bow, title, "left", date, outlet])

	for article,bow in y_articles:
		title, date = article.title.strip(), article.date_publish
		bow = (" ").join(bow)
		csvwriter.writerow([bow, title, "right", date, outlet])


# determine whether articles are right or left 
# outlets: {"outlet":[articles]}
def to_csv_media_outlet(filename, outlets, x_keywords, y_keywords):
	with open(filename, "w") as f:
		csvwriter = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
		csvwriter.writerow(["documents","docname","rating","date","outlet"])

		for outlet, articles in outlets.items():
			print("adding %s to csv, len %d" %(outlet,len(articles)))
			to_csv_leaning(articles, outlet, x_keywords, y_keywords, csvwriter)





if __name__ == '__main__':
	datapath = "/Users/ninawang/Thesis/remote/THESIS2019/example_data/"
	# path = "/Users/ninawang/Thesis/remote/THESIS2019/example_data/NYT-OPINION2016-2017-processed/"
	# start = datetime.datetime(2016, 6, 1)
	# end = datetime.datetime(2017, 7, 1)
	# articles = lex.get_articles_from_filepath(path,start,end)

	# year = 2012
	# filename = "NYT-OPINION"+str(year)+"-"+str(year+1)
	# path = datapath + filename + "-processed/"
	# articles = lex.get_articles_year(path, year)
	outlets = ["NYT-OPINION","BREITBART", "CNN", "FOX","MSN","NATIONALREVIEW","NPR","REUTERS-POLITICS","SLATE","WASHINGTONEXAMINER"]
	articles = get_articles_outlets(datapath,outlets,2012)
	print(len(articles.values()))
	filename = "stm_data_media_outlet_2.csv"
	to_csv_media_outlet(filename,articles, LEFT_WORDS, RIGHT_WORDS)

	# csvfile = "/Users/ninawang/Thesis/remote/THESIS2019/PART1/TopicModel/STM/stm_data.csv"
	# run_stm(filename+"-stm", csvfile)

	# print("saved to " +filename+"-stm" )




