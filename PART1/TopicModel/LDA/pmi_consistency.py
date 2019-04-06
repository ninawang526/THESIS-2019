import pickle
import datetime

import THESIS2019.utils.to_lexicon as lex
from THESIS2019.utils.base_words import *
import THESIS2019.utils.get_articles as get

import numpy as np
import pandas as pd
from collections import defaultdict
import nltk
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel



def pickle_dump(outlet, articles, dictionary, corpus, model):
    with open("data/"+outlet+"_model2012.pkl","wb") as f:
        pickle.dump(model, f)
    with open("data/"+outlet+"_articles2012.pkl","wb") as f:
        pickle.dump(articles, f)
    with open("data/"+outlet+"_dictionary2012.pkl","wb") as f:
        pickle.dump(dictionary, f)
    with open("data/"+outlet+"_corpus2012.pkl","wb") as f:
        pickle.dump(corpus, f)
        
def pickle_load(outlet):
    with open("data/"+outlet+"_model2012.pkl","rb") as f:
        model = pickle.load(f)
    with open("data/"+outlet+"_dictionary2012.pkl","rb") as f:
        dictionary = pickle.load(f)
    with open("data/"+outlet+"_articles2012.pkl","rb") as f:
        articles = pickle.load(f)
    with open("data/"+outlet+"_corpus2012.pkl","rb") as f:
        corpus = pickle.load(f)
    return articles, dictionary, corpus, model
    

def get_lda_models(article_set, load=True, store=False):
    arts_list, dictionary_list, corpus_list, model_list = defaultdict(lambda:0),defaultdict(lambda:0),defaultdict(lambda:0),defaultdict(lambda:0)
    
    for outlet, articles in article_set.items():
        print("%s has %d articles" %(outlet, len(articles)))
        
        if load:
            arts, dictionary, corpus, model = pickle_load(outlet)
        else:
            art_set = {outlet:articles}
            arts, dictionary, corpus, model = lex.LDA(art_set, LEFT_WORDS+RIGHT_WORDS,num_topics=30)
            if store:
                pickle_dump(outlet, arts, dictionary, corpus, model)
        
        arts_list[outlet]=arts
        dictionary_list[outlet]=dictionary
        corpus_list[outlet]=corpus
        model_list[outlet]=model
        
    return arts_list, dictionary_list, corpus_list, model_list



def print_topics(model):
    topics = model.show_topics(num_topics=-1, num_words=5, log=False, formatted=False)
    for idx, topic in topics:
        print ("topic " + str(idx) + ": " + (",  ").join([str(t[0]) for t in topic]))
    

def pmi(wordslist, texts, corpus, dictionary):
    cm = CoherenceModel(topics=wordslist, corpus=corpus, dictionary=dictionary, coherence='u_mass')#coherence='c_npmi')
    return cm.get_coherence()


# for each topic, compare to other topic
def compare_topics(t1, t2, texts, corpus, dictionary):
    permutations = []
    for w1 in t1:
        for w2 in t2:
            permutations.append([w1,w2])
    return pmi(permutations,texts,corpus,dictionary)


def pairwise_compare_models(model1, model2, texts, corpus, dictionary):
    topics1 = model1.show_topics(num_topics=-1, num_words=10, log=False, formatted=False)
    topics2 = model2.show_topics(num_topics=-1, num_words=10, log=False, formatted=False)
    
    pmis_col = defaultdict(lambda:0)
    for idx1, topic1 in topics1[:5]:
        pmis_row = defaultdict(lambda:0)
        for idx2, topic2 in topics2[:5]:
            t1 = [str(t[0]) for t in topic1]
            t2 = [str(t[0]) for t in topic2]
            print()
            print ("topic " + str(idx1) + ": " + (",  ").join(t1))
            print ("topic " + str(idx2) + ": " + (",  ").join(t2))
    
            pmi = compare_topics(t1, t2, texts, corpus, dictionary)
            print("pmi between t%s and t%s: %6.4f" %(idx1, idx2, pmi))

            pmis_row["b"+str(idx2)]=pmi
        pmis_col["a"+str(idx1)]=pmis_row
    df =pd.DataFrame(pmis_col)
    print(df)
    print(list(df.values)[0])
           

if __name__ == '__main__':
    datapath = "/Users/ninawang/Thesis/remote/THESIS2019/example_data_1000/"
    # outlets = ['BREITBART','NATIONALREVIEW','FOX',
    #              'WASHINGTONEXAMINER','REUTERS','NPR',
    #              'NYT', 'MSN','CNN','SLATE']
    outlets = ['NYT', 'CNN']
    articles = get.get_articles_outlets(datapath,outlets,2012,filter_date=False)

    arts_list, dictionary_list, corpus_list, model_list = get_lda_models(articles, load=True, store=False)

    outlet_list = list(articles.keys())
    compare = []
    for i in range(len(outlet_list)):
        for j in range(i+1, len(outlet_list)):
            outlet1 = outlet_list[i]
            outlet2 = outlet_list[j]
            # get models
            model1 = model_list[outlet1]
            model2 = model_list[outlet2]
            # get joint texts, dictionary, corpus
            texts = list(arts_list[outlet1].values())[0]+list(arts_list[outlet2].values())[0]
            dictionary = corpora.Dictionary(texts)
            corpus = [dictionary.doc2bow(text) for text in texts]
            # compare models
            print("comparing %s and %s"%(outlet1, outlet2))
            print_topics(model1)
            print_topics(model2)


            pairwise_compare_models(model1, model2, texts, corpus, dictionary)
    




# get_coh([["olymp","sport"]],texts,corpus,dictionary)











