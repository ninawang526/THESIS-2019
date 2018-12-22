import datetime, os, re
import pickle
import datetime
from newsplease import NewsPlease
from base_words import *

import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

import gensim
from gensim import corpora
from gensim.models import Phrases


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


## filter out irrelevant parts of speech
def filter_support_words(article):
    tagged = (nltk.pos_tag(nltk.word_tokenize(article)))
    filt = set(["MD","CD","IN"])
    words = [tup[0] for tup in tagged if tup[1] not in filt]
    return words
        

## helper method for text cleanup
def text_cleanup(text, filter_support = False):
	def get_lemma(word):
		return WordNetLemmatizer().lemmatize(word)

	stop_words = set(stopwords.words('english'))
	months = ["january","february","march","april","may","june","july","august","september","october","november","december"]

	if filter_support:
		tokens = filter_support_words(text)
	else:
		tokens = nltk.word_tokenize(text)
	tokens = [token for token in tokens if (token).isalpha() and len(token)>2]
	tokens = [token for token in tokens if token.lower() not in stop_words]
	tokens = [token for token in tokens if token.lower() not in months]
	tokens = [get_lemma(token.lower()) for token in tokens]
	return tokens


# the 15 words around a sentence
def get_collocations(articles, keywords, opposite_keywords,exclude=[]): 
    needles = set([s.lower() for s in keywords])
    anti_needles = set([s.lower() for s in (opposite_keywords+exclude)])

    collocations = []
    # sentences = []
    for article in articles:
        title, date_publish, text = article.title, article.date_publish, article.text
        # sentences = tokenize.sent_tokenize(text)
        words = text_cleanup(text,filter_support=True)
        # tokened_s = [text_cleanup(sent, filter_support=True) for sent in sentences]

        # for n, sent in enumerate(tokened_s):
        #     haystack = set([s.lower() for s in sent])
        #     if len(haystack.intersection(needles)) > 0 and len(haystack.intersection(anti_needles)) == 0:
        #         collocations += sent
        for n, word in enumerate(words):
        	if word in needles:
	        	start_i = max(0, n-15)
	        	end_i = min(n+15, len(words)-1)
	        	surround = words[start_i:end_i]
	        	
	        	haystack = set([s.lower() for s in surround])
	        	if len(haystack.intersection(anti_needles)) == 0:
	        		collocations += [w for w in surround if w.lower() not in needles and w.lower() not in ["president","party"]]
	    
    # collocations = [word for word in collocations if word.lower() not in keywords and not in exclude]
    return collocations


def embeddings(articles, keywords):
	docs = []
	for article in articles:
		sentences = tokenize.sent_tokenize(article.text)
		sentences = [gensim.utils.simple_preprocess(s) for s in sentences]
		docs += sentences

	model = gensim.models.Word2Vec(
		docs,
		size=150,
		window=10,
		min_count=2,
		workers=10)

	model.train(docs, total_examples=len(docs), epochs=10)
	print (model.wv.most_similar(positive="obama",topn=20))
	print (model.wv.most_similar(positive="romney",topn=20))
	# print (keywords[0])
	# print (sim)

	# remember to preserve the similarity weights
	# !!! what would it mean if there are lower similarity weights
	# as in, very few words used in the same context...



# additional words in the body of the documents
# don't need to keep track of word probabilities...just bow
def topic_model(articles, keywords, print_words=False):
	needles = set([s.lower() for s in keywords])

	doc = []
	for article in articles:
		words = text_cleanup(article.text, filter_support=True)
		haystack = set([s.lower() for s in words])
		if len(haystack.intersection(needles)) > 0:
			doc.append(words)

	filt_doc = []
	for bow in doc:
		filt_words = [word for word in bow if word.lower() not in needles]
		filt_doc.append(filt_words)

	doc = filt_doc

	# Add bigrams and trigrams to docs
	bigram = Phrases(doc, min_count=5,threshold=10) #min_count=5, threshold=10
	trigram = Phrases(bigram[doc])

	# build dict & corpus
	updated_docs = [bigram[doc[i]] for i in range(len(doc))]
	dictionary = corpora.Dictionary(updated_docs)
	corpus = [dictionary.doc2bow(text) for text in updated_docs]
	

	NUM_TOPICS =15
	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=200)

	topics = ldamodel.show_topics(num_topics=50, num_words=50, log=False, formatted=False)

	if print_words:
		for idx, topic in topics:
			print ("topic " + str(idx) + ": " + (",  ").join([str(t[0]) for t in topic]))
			print ("\n")

	return topics


if __name__ == '__main__':
	path = "."
	
	start = datetime.datetime(2012, 6, 1)
	end = datetime.datetime(2013, 7, 1)
	
	articles = get_articles_from_filepath(path,start,end)

	print ("finished grabbing articles")

	left_collocations = get_collocations(articles, LEFT_WORDS,RIGHT_WORDS)
	# romney_collocations = get_collocations(articles, ["conservative", "conservatives", "conservatism"])
	
	print (left_collocations[:500])
	print (len(left_collocations))

	# print (romney_collocations[32])
	# print (len(romney_collocations))

	print ("finished collocations")

	# obama_tm = topic_model(articles, ["obama"])
	# romney_tm = topic_model(articles, ["conservative", "conservatives", "conservatism"], print_words=True)
	# print_topics(romney_tm)




	# obama_embedding = embeddings(articles, ["conservative", "conservatives", "conservatism"])
	

	print ()


