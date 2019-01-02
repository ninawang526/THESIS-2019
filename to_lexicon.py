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
from nltk.stem.porter import *

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
def filter_support_words(tokens):
    tagged = (nltk.pos_tag(tokens))
    filt = set(["MD","CD","IN"])
    words = [tup[0] for tup in tagged if tup[1] not in filt]
    return words
       

## helper method for text cleanup
def text_cleanup(text, filter_support=False,proper_nouns=[]):
	stop_words = set(stopwords.words('english'))
	months = ["january","february","march","april","may","june","july","august","september","october","november","december"]

	tokens = nltk.word_tokenize(text)
	if filter_support:
		tokens = filter_support_words(tokens)
	
	def filt(token):
		if token not in proper_nouns:
			token = token.lower()
		return (token.isalpha() and len(token)>2 and (token not in stop_words) and (token not in months)) 
	
	## stem & lemmatize tokens
	def lemma_stem(token):
		if token in proper_nouns:
			return token
		token = token.lower()
		stemmer = PorterStemmer()
		lemma = WordNetLemmatizer().lemmatize(token)
		return stemmer.stem(lemma)

	tokens = [lemma_stem(token) for token in tokens if filt(token)]
	
	return tokens
 

def needles_cleanup(needles):
	def stem(token):
		if token[0].isupper():
			return token
		stemmer = PorterStemmer()
		return stemmer.stem(token)
	return set([stem(wd) for wd in needles])


# the 15 words around a sentence -- organized in articles: [[][]]
def get_collocations(articles, keywords, opposite_keywords,exclude=[]): 
	# needles = set([stemmer.stem(s.lower()) for s in keywords])
	# anti_needles = set([stemmer.stem(s.lower()) for s in (opposite_keywords+exclude)])

	needles = needles_cleanup(keywords)
	anti_needles = needles_cleanup(opposite_keywords+exclude)

	proper_nouns = [wd for wd in (keywords+opposite_keywords+exclude) if wd[0].isupper()]
	# print (proper_nouns)

	collocations = []
	rel_articles = []

	for article in articles:
		title, date_publish, text = article.title, article.date_publish, article.text
		words = text_cleanup(text,filter_support=True,proper_nouns=proper_nouns) # filter & tokenize words

		article_collocations = []
		
		i = 0
		while i < len(words):
			word = words[i]
			if word in needles:
				start_i = max(0, i-15)
				end_i = min(i+15, len(words)-1)
				surround = words[start_i:end_i]

				haystack = set(surround)
				if len(haystack.intersection(anti_needles)) == 0:
					article_collocations += [w for w in surround if w not in needles and w not in ["presid","parti"]]
					i = end_i+15

					# print (word)
					# print (" ".join(surround))
					# print ("\n")
			i+=1

		if len(article_collocations)>0:
			collocations.append(article_collocations)
			rel_articles.append(article)
	# collocations = [word for word in collocations if word.lower() not in keywords and not in exclude]
	return collocations, rel_articles


# the 15 words around a sentence -- organized in articles: [[][]]
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


# collocations, except preserving article order
def tokenize_articles(articles, keywords, opposite_keywords, exclude=[]):
    needles = set([s.lower() for s in keywords])
    anti_needles = set([s.lower() for s in (opposite_keywords+exclude)])

    collocations = []
    for article in articles:
        title, date_publish, text = article.title, article.date_publish, article.text
        sentences = tokenize.sent_tokenize(text)
        tokened_s = [text_cleanup(sent, filter_support=True) for sent in sentences]

        art = []
        for n, sent in enumerate(tokened_s):
            haystack = set([s.lower() for s in sent])
            if len(haystack.intersection(needles)) > 0 and len(haystack.intersection(anti_needles)) == 0 and len(haystack.intersection(set(exclude))) == 0:
                sent = [word for word in sent if word not in keywords]
                art += sent
        
        if len(art)>0:
        	collocations.append(art)

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

	# CHANGE!!!!!!
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
	path = "NYT-OPINION2012-2013-processed/"
	
	start = datetime.datetime(2012, 6, 1)
	end = datetime.datetime(2013, 7, 1)
	
	articles = get_articles_from_filepath(path,start,end)

	print ("finished grabbing articles")

	left_collocations, l_arts = get_collocations(articles, LEFT_WORDS,RIGHT_WORDS)
	# romney_collocations = get_collocations(articles, ["conservative", "conservatives", "conservatism"])
	
	# print (left_collocations[:500])
	# print (len(left_collocations))

	# print (romney_collocations[32])
	# print (len(romney_collocations))

	print ("finished collocations")

	# obama_tm = topic_model(articles, ["obama"])
	# romney_tm = topic_model(articles, ["conservative", "conservatives", "conservatism"], print_words=True)
	# print_topics(romney_tm)




	# obama_embedding = embeddings(articles, ["conservative", "conservatives", "conservatism"])
	

	print ()


