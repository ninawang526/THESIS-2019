import zipfile, re, pickle, os, sys
from newsplease import NewsPlease


total = 0
distinct = 0

opinion = 0
world = 0
politics = 0

seen = []

data_path = sys.argv[1]

with zipfile.ZipFile(data_path,"r") as f:
    for name in f.namelist():
        if name.endswith(".snapshot"):

            
            filename = name.split("/")[-2].split(".")[0]
            total += 1
            
            if filename in seen:
                continue
            else:
                seen.append(filename)
            
            html = f.read(name)
            try:
                article = NewsPlease.from_html(html, url=None)
            except e as ArticleException:

  
            if re.search("/opinion/",name):
                opinion += 1
            elif re.search("/politics/",name):
                politics += 1
            elif re.search("/world/",name):
                world += 1

            distinct +=1

            if (distinct % 1000) == 0:
                print (distinct)
            
            # SAVE
            savename = data_path.split("/")[-1].split(".")[0] + "-processed/" + ("/").join(name.split("/")[1:-2]) 
            print (savename)

            if not os.path.exists(savename):
                os.makedirs(savename)

            with open(savename + "/" + filename +".pkl", 'wb') as output:
                pickle.dump(article, output, pickle.HIGHEST_PROTOCOL)

            # print (savename + "/" + filename +".pkl", 'wb')
            # print (article.title)

print ("total", total)
print ("distinct", distinct)
print ("opinion", opinion)
print ("politics", politics)
print ("world", world)

