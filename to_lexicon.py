import datetime, os, re
import pickle
import datetime
from newsplease import NewsPlease

import nltk
from nltk import tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

import gensim
from gensim import corpora
from gensim.models import Phrases


## helper method to filter articles by date & length
def get_relevant_articles(articles):
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
def get_articles_from_filepath(path):
	articles = []
	for (dirpath, dirnames, filenames) in os.walk(path):
		for filename in filenames:
			if filename.endswith(".pkl"):
				filepath = os.path.join(dirpath, filename)
				with open(filepath, 'rb') as input_file:
					e = pickle.load(input_file)
					articles.append(e)
	filtered_articles = get_relevant_articles(articles)
	return filtered_articles


## helper method for text cleanup
def text_cleanup(text):
	def get_lemma(word):
		return WordNetLemmatizer().lemmatize(word)

	stop_words = set(stopwords.words('english'))
	months = ["january","february","march","april","may","june","july","august","september","october","november","december"]

	tokens = nltk.word_tokenize(text)
	tokens = [token for token in tokens if (token).isalpha() and len(token)>2]
	tokens = [token for token in tokens if token.lower() not in stop_words]
	tokens = [token for token in tokens if token.lower() not in months]
	tokens = [get_lemma(token) for token in tokens]
	return tokens


# each entry is the words contained in a sentence
def get_collocations(articles, keywords):
    collocations = []
    for article in articles:
        title, date_publish, text = article.title, article.date_publish, article.text
        sentences = tokenize.sent_tokenize(text)
        tokened_s = [text_cleanup(sent) for sent in sentences]

        for n, sent in enumerate(tokened_s):
            haystack = set([s.lower() for s in sent])
            needles = set([s.lower() for s in keywords])
            if len(haystack.intersection(needles)) > 0:
                collocations += sent
    
    collocations = [word for word in collocations if word not in keywords]
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
def topic_model(articles, keywords):
	needles = set([s.lower() for s in keywords])

	doc = []
	for article in articles:
		words = text_cleanup(article.text)
		haystack = set([s.lower() for s in words])
		if len(haystack.intersection(needles)) > 0:
			doc.append(words)

	filt_doc = []
	for bow in doc:
		filt_words = [word for word in bow if word.lower() not in needles]
		filt_doc.append(filt_words)

	doc = filt_doc

	# Add bigrams and trigrams to docs
	bigram = Phrases(doc,min_count=5,threshold=10) #min_count=5, threshold=10
	trigram = Phrases(bigram[doc])

	# build dict & corpus
	updated_docs = [bigram[doc[i]] for i in range(len(doc))]
	dictionary = corpora.Dictionary(updated_docs)
	corpus = [dictionary.doc2bow(text) for text in updated_docs]
	

	NUM_TOPICS =15
	ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=200)

	# print topics
	topics = ldamodel.show_topics(num_topics=50, num_words=50, log=False, formatted=False)
	for idx, topic in topics:
		print ("topic " + str(idx) + ": " + (",  ").join([str(t[0]) for t in topic]))
		print ("\n")

	topics_bow = [tup[0] for topic in topics for tup in topic]
	print (topics_bow)
	return topics_bow


if __name__ == '__main__':
	path = "."
	
	global start
	global end
	start = datetime.datetime(2012, 6, 1)
	end = datetime.datetime(2013, 7, 1)
	
	articles = get_articles_from_filepath(path)

	print ("done1")

	# obama_collocations = get_collocations(articles, ["obama"])
	# romney_collocations = get_collocations(articles, ["romney"])

	print ("done2")

	# obama_tm = topic_model(articles, ["obama"])
	# romney_tm = topic_model(path, ["romney"])

	obama_embedding = embeddings(articles, ["obama"])
	# print (obama_collocations[19])
	# print (len(obama_collocations))

	# print (romney_collocations[32])
	# print (len(romney_collocations))

	print ()


