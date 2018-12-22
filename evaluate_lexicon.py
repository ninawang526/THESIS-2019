

def word_distributions()


### evaluating collocates
path = "."

start = datetime.datetime(2012, 6, 1)
end = datetime.datetime(2013, 7, 1)

articles = get_articles_from_filepath(path, start, end)

print ("finished grabbing articles")

left_collocations = get_collocations(articles, LEFT_WORDS)
# romney_collocations = get_collocations(articles, ["conservative", "conservatives", "conservatism"])

print (left_collocations[:500])
print (len(left_collocations))

# print (romney_collocations[32])
# print (len(romney_collocations))

print ("finished collocations")

# obama_tm = topic_model(articles, ["obama"])
# romney_tm = topic_model(articles, ["conservative", "conservatives", "conservatism"], print_words=True)
# print_topics(romney_tm)
