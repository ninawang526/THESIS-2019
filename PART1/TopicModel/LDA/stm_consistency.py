import pickle
import datetime
import time
import numpy as np
from collections import defaultdict

from THESIS2019.utils.base_words import *
import THESIS2019.utils.get_articles as get
# import THESIS2019.PART1.TopicModel.STM.stm as stm

import dit
from dit.divergences import jensen_shannon_divergence



if __name__ == '__main__':
	print("here")
	# datapath = "/Users/ninawang/Thesis/remote/THESIS2019/example_data/"
	datapath = "/n/fs/thesis-ninaw/ARTICLES-2012/processed/"

	outlets = ['BREITBART','NATIONALREVIEW','FOX',
			'WASHINGTONEXAMINER','REUTERS','NPR',
			'NYT', 'MSN','CNN','SLATE']

	print("getting articles")
	start_time = time.time()
	
	articles = get.get_articles_outlets(datapath,outlets,2012,filter_date=False)
	
	print("--- %s seconds ---" % (time.time() - start_time))

	print("writing csv")
	start_time = time.time()

	# writing all files to csv - labeled with left/right and outlet.
	folder = "stm_data/"
	csv_filename = folder+"stm_data.csv"
	csvfile = stm.to_csv_media_outlet(csv_filename,articles, LEFT_WORDS, RIGHT_WORDS)
	print("--- %s seconds ---" % (time.time() - start_time))
	

	# run stm
	# stm.run_stm(folder, csv_filename)

	# sum frequencies
