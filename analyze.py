import pickle

fp = "NYT-POL2012-2013-processed-polar.pkl"

with open(fp, 'rb') as f: 
	polar = pickle.load(f)
	for p in polar[:50]:
		phr,c = p[0].split(" "),p[1]
		print ("{}.{}.{}\t\t{}".format(phr[0],phr[1],phr[2],c))
	print ("\n")