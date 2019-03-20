# evaluation of how distinct topic models are

from THESIS2019.PART1.TopicModel.DTM import dtm_utils as dtm
import THESIS2019.utils.to_lexicon as lex

# Do different parties talk about different topics?
# 1) LDA and Jensen-Shannon Divergence
path = "../../ARTICLES-2016/processed/NYT-OPINION2016-2017-processed/"
start = datetime.datetime(2016, 6, 1)
end = datetime.datetime(2017, 7, 1)
articles = get_articles_from_filepath(path,start,end)

print ("finished grabbing articles")

filt_articles, dictionary, corpus, ldamodel = lex.LDA()





# Do different parties talk about the same topics differently?
# 1) DTM



