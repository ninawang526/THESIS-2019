import csv
import pandas as pd
import subprocess

import THESIS2019.PART1.TopicModel.STM.stm as stm
from THESIS2019.utils.base_words import *
from THESIS2019.utils.get_articles import *


def get_outlet_ind(outlets, row_outlet):
	for ind, outlet in enumerate(outlets):
		if outlet in row_outlet:
			return ind


# transforms a dataframe into proper cclda input form
def to_cclda_input(df, outlets, filename):    
	with open(filename,"w") as f:
		for index, row in df.iterrows():
			ind = get_outlet_ind(outlets, row["outlet"])
			f.write("%d %s\n" %(ind, row["documents"]))


# execute cclda; optional if csv file is already available
def cclda(articles, outlets, filename="cclda", csvfile=None):
	path = "data/"+filename
	# write results to csv and reads into dataframe
	if csvfile is None:
		stm.to_csv_media_outlet(path+".csv", articles, LEFT_WORDS, RIGHT_WORDS)
		df = pd.read_csv(path+".csv")
	else:
		df = pd.read_csv(csvfile)
	# turns dataframe into proper cclda input form
	to_cclda_input(df, outlets, path+".txt")
	# executes the cclda code
	subprocess.call(["./cclda.sh"])
	# write a legend for the outlets
	with open("data/output_topwords_cclda.txt","a") as f:
		f.write("\nLEGEND:\n")
		for i,outlet in enumerate(outlets):
			f.write("%s: %d\n" %(outlet, i))


if __name__ == '__main__':
	# note: in later iterations, see if we can use an already created file. 
	datapath = "/Users/ninawang/Thesis/remote/THESIS2019/example_data/"
	outlets = ["NYT-OPINION","BREITBART", "CNN", "FOX","MSN","NATIONALREVIEW","NPR","REUTERS-POLITICS","SLATE","WASHINGTONEXAMINER"]
	articles = get_articles_outlets(datapath,outlets,2012)
	
	cclda(articles, outlets, filename="cclda",csvfile="data/cclda.csv")



