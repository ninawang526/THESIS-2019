import datetime, os, re
import pickle
import time
from newsplease import NewsPlease

from THESIS2019.utils.base_words import *
from THESIS2019.utils.get_articles import *

import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import *

import gensim
from gensim import corpora
from gensim.models import Phrases
from gensim.models.ldamulticore import LdaMulticore
from gensim.models.coherencemodel import CoherenceModel
from gensim.test.utils import common_corpus, common_dictionary




## filter out irrelevant parts of speech
def filter_support_words(tokens):
    tagged = (nltk.pos_tag(tokens))
    filt = set(["MD","CD","IN"])
    words = [tup[0] for tup in tagged if tup[1] not in filt]
    return words

def filter_banned_words(tokens):
	words = [tok for tok in tokens if tok not in BANNED]
	return words 

## helper method for text cleanup
def text_cleanup(text, filter_support=False, proper_nouns=[], filter_banned=False):
	stop_words = set(stopwords.words('english'))

	tokens = nltk.word_tokenize(text)
	if filter_support:
		tokens = filter_support_words(tokens)
	
	def filt(token):
		if token not in proper_nouns:
			token = token.lower()
		return (token.isalpha() and len(token)>2 and (token not in stop_words) and (token not in MONTH_NAMES))
	
	## stem & lemmatize tokens
	def lemma_stem(token):
		if token in proper_nouns:
			return token
		token = token.lower()
		stemmer = PorterStemmer()
		lemma = WordNetLemmatizer().lemmatize(token, pos="v") # LOOK AT THIS!! POS??
		return stemmer.stem(lemma)

	tokens = [lemma_stem(token) for token in tokens if filt(token)]

	if filter_banned:
		tokens = filter_banned_words(tokens)
	
	return tokens
 

# helper method to stem all needles except proper nouns
def needles_cleanup(needles):
	def stem(token):
		if token[0].isupper():
			return token
		stemmer = PorterStemmer()
		return stemmer.stem(token)
	return set([stem(wd) for wd in needles])


# the 15 words around a sentence -- organized in articles: [[][]]
# all the collocations inside an article are lumped into one list
def get_collocations(articles, x_keywords, y_keywords): 
	x_needles = needles_cleanup(x_keywords)
	y_needles = needles_cleanup(y_keywords) 
	# exclude = needles_cleanup(exclude)
	# words that if found, don't count; since we only want "romney" in a sentence for right; 
	# if "obama" is also present, we don't know what sentence is referring to.

	proper_nouns = [wd for wd in (x_keywords+y_keywords) if wd[0].isupper()]

	x_collocations = []
	x_articles = []
	y_collocations = []
	y_articles = []

	for article in articles:
		title, date_publish, text = article.title, article.date_publish, article.text
		words = text_cleanup(text, filter_support=True, proper_nouns=proper_nouns, filter_banned=True) # filter & tokenize words
		# recently changed filter banned=true. 
		x_article_collocations = []
		y_article_collocations = []
		
		i = 0
		while i < len(words):
			word = words[i]
			start_i = max(0, i-15)
			end_i = min(i+15, len(words)-1)
			surround = words[start_i:end_i]

			# found a word in x group
			if word in x_needles:
				haystack = set(surround)
				if len(haystack.intersection(y_needles)) == 0:
					x_article_collocations += [w for w in surround if w not in x_needles] #+ ["presid","parti"]]
					i = end_i+15

			# found a word in y group 
			elif word in y_needles:
				haystack = set(surround)
				if len(haystack.intersection(x_needles)) == 0:
					y_article_collocations += [w for w in surround if w not in y_needles] 
					i = end_i+15
			i+=1

		if len(x_article_collocations)>0:
			x_collocations.append((date_publish, x_article_collocations))
			x_articles.append(article)

		if len(y_article_collocations)>0:
			y_collocations.append((date_publish, y_article_collocations))
			y_articles.append(article)

	# collocations = [word for word in collocations if word.lower() not in keywords and not in exclude]
	# return x_collocations, x_articles, y_collocations, y_articles
	return x_collocations, y_collocations





## the 15 words around a sentence -- organized in articles: [[][]]
def get_collocations_without_tokenizing(articles, keywords, opposite_keywords,exclude=[]): 
	stemmer = PorterStemmer() 

	needles = set([stemmer.stem(s.lower()) for s in keywords])
	anti_needles = set([stemmer.stem(s.lower()) for s in (opposite_keywords+exclude)])

	collocations = []
	rel_articles = []

	for article in articles:
		title, date_publish, text = article.title, article.date_publish, article.text
		# sentences = tokenize.sent_tokenize(text)
		words = text.split(" ")
		#words = nltk.word_tokenize(text)
		article_collocations = ""
		
		i = 0
		while i < len(words):
			word = words[i]
			if word in needles:
				start_i = max(0, i-15)
				end_i = min(i+15, len(words)-1)
				surround = words[start_i:end_i]

				haystack = set([s.lower() for s in surround])
				if len(haystack.intersection(anti_needles)) == 0:
					sent = " ".join(surround)
					article_collocations += (sent)
					i = end_i+15
			i+=1

		if len(article_collocations)>0:
			collocations.append(article_collocations)
			rel_articles.append(article)
	# collocations = [word for word in collocations if word.lower() not in keywords and not in exclude]
	return collocations, rel_articles


# # collocations, except preserving article order
# def tokenize_articles(articles, keywords, opposite_keywords, exclude=[]):
#     needles = set([s.lower() for s in keywords])
#     anti_needles = set([s.lower() for s in (opposite_keywords+exclude)])

#     collocations = []
#     for article in articles:
#         title, date_publish, text = article.title, article.date_publish, article.text
#         sentences = tokenize.sent_tokenize(text)
#         tokened_s = [text_cleanup(sent, filter_support=True) for sent in sentences]

#         art = []
#         for n, sent in enumerate(tokened_s):
#             haystack = set([s.lower() for s in sent])
#             if len(haystack.intersection(needles)) > 0 and len(haystack.intersection(anti_needles)) == 0 and len(haystack.intersection(set(exclude))) == 0:
#                 sent = [word for word in sent if word not in keywords]
#                 art += sent
        
#         if len(art)>0:
#         	collocations.append(art)

#     return collocations


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


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model=gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics)
        model_list.append(model)
#         coherence_values.append(ldamodel.log_perplexity(corpus))
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def graph_coherence(dictionary, corpus, texts):
	start=2
	limit=100
	step=3
	model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus, texts=texts, start=start, limit=limit, step=step)

	# Show graph
	# import matplotlib.pyplot as plt
	x = range(start, limit, step)
	# plt.plot(x, coherence_values)

	for i, c in enumerate(coherence_values):
	    print (x[i], c)

	# plt.xlabel("Num Topics")
	# plt.ylabel("Coherence score")
	# plt.legend(("coherence_values"), loc='best')
	# plt.show()


def get_topicmodel_docs(article_set, keywords):
	needles = set([s.lower() for s in keywords])

	# filter articles for presence of keyword 
	filt_articles = {}
	doc = []

	for key,articles in article_set.items():
		filt_article_set = []
		for article in articles:
			words = text_cleanup(article.text, filter_support=True)
			haystack = set([s.lower() for s in words])
			if len(haystack.intersection(needles)) > 0:
				doc.append(words)
				filt_article_set.append(article)
		filt_articles[key]=filt_article_set

	filt_doc = []
	for bow in doc:
		filt_words = [word for word in bow if word.lower() not in needles]
		filt_doc.append(filt_words)

	doc = filt_doc

	# Add bigrams and trigrams to docs
	bigram = Phrases(doc, min_count=5,threshold=10) #min_count=5, threshold=10
	trigram = Phrases(bigram[doc])

	updated_docs = [bigram[doc[i]] for i in range(len(doc))] 
	
	filtered_updated_docs = []
	for bow in updated_docs:
		filtered_updated_docs.append([word for word in bow if word not in BANNED])
	updated_docs = filtered_updated_docs

	# with open("topicmodel_docs2016.pkl","wb") as f:
	# 	pickle.dump(updated_docs, f)

	return filt_articles, updated_docs


# additional words in the body of the documents
# don't need to keep track of word probabilities...just bow
# article set example: {2012:[], 2016:[]}
def LDA(article_set, keywords, num_topics=15, best_coherence=False):
	print ("getting docs...")
	
	start_time = time.time()

	filt_articles, updated_docs = get_topicmodel_docs(article_set,keywords)
	
	# LOAD
	# with open("topicmodel_docs2016.pkl","rb") as f:
		# updated_docs = pickle.load(f)
	
	print("--- %s seconds ---" % (time.time() - start_time))

	# build dict & corpus
	dictionary = corpora.Dictionary(updated_docs)
	corpus = [dictionary.doc2bow(text) for text in updated_docs]

	if best_coherence:
		graph_coherence(dictionary, corpus, updated_docs)

	print ("getting LDA model...")

	start_time = time.time()
	ldamodel = LdaMulticore(corpus, num_topics = num_topics, id2word=dictionary, passes=50, workers=15)#200
	
	with open("ldamodel2.pkl","wb") as f:
		pickle.dump(ldamodel, f)
		
	print("--- %s seconds ---" % (time.time() - start_time))

	return filt_articles, dictionary, corpus, ldamodel 




if __name__ == '__main__':
	path = "../../ARTICLES-2016/processed/NYT-OPINION2016-2017-processed/"
	
	start = datetime.datetime(2016, 6, 1)
	end = datetime.datetime(2017, 7, 1)
	
	articles = get_articles_from_filepath(path,start,end)

	print ("finished grabbing articles")

	# left_collocations, l_arts = get_collocations(articles, LEFT_WORDS,RIGHT_WORDS)
	# romney_collocations = get_collocations(articles, ["conservative", "conservatives", "conservatism"])
	
	# print (left_collocations[:500])
	# print (len(left_collocations))

	# print (romney_collocations[32])
	# print (len(romney_collocations))

	print ("finished collocations")

	articles, dictionary, corpus, model = LDA(articles, LEFT_WORDS+RIGHT_WORDS,num_topics=70)
	# romney_tm = topic_model(articles, ["conservative", "conservatives", "conservatism"], print_words=True)

	# !!!!!!!!! DUMP !!!!!!!!! #
	pickle.dump(model, open("both_tm.pkl","wb"))
	pickle.dump(dictionary, open("both_dict.pkl","wb"))
	pickle.dump(corpus, open("both_corpus.pkl","wb"))
	pickle.dump(articles, open("both_articles.pkl","wb"))

	topics = model.show_topics(num_topics=-1, num_words=25, log=False, formatted=False)

	for idx, topic in topics:
		print ("topic " + str(idx) + ": " + (",  ").join([str(t[0]) for t in topic]))
		print ("\n")


	# obama_embedding = embeddings(articles, ["conservative", "conservatives", "conservatism"])
	

	print ()


