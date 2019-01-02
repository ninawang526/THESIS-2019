import FightinWords as fw
import numpy as np
from wordfreq import word_frequency
from sklearn.feature_extraction.text import CountVectorizer as CV

import nltk

# l1 = 'the quick brown fox trot jumps over the lazy pig'
# l2 = 'the lazy purple pig jumps over the fox trot lazier donkey pig pig pig pig pig pig fox trot'
# v = 'purple pig'

# l1 = ["the quick", "quick brown", "brown fox"]#["the","quick", "quick", "brown", "brown fox", "new new"]
# l2 = ["the quick", "not brown", "happy now", "hi"]
# v = ["the quick", "quick brown"]

l1 = ['score low trustworthi', 'low trustworthi question', 'trustworthi question well', 'question well let', 'well let talk']
l2 = ['privat email server', 'state john kerri', 'secretari state john', 'nobel peac prize', 'nation committe email']
v = ['score low trustworthi', 'state john kerri']


# prior = []
# for word in vocab:
# 	prior.append(word_frequency(word, 'en'))

# prior = np.array(prior)

vocab = v

cv = CV(decode_error = 'ignore',
                binary = False, ngram_range=(1,3),
                max_features = 15000, vocabulary=vocab)

X = cv.fit_transform(l1+l2)
print (cv.get_feature_names()[:5])

r = fw.bayes_compare_language(l1, l2, cv=cv)
print (r)

for i in sorted(r,key=lambda x:x[1]):
	print (i)