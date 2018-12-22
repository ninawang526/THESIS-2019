for (dirpath, dirnames, filenames) in os.walk("."):
	if len(filenames) > 1 and filenames[0].endswith("snapshot"):
		for i in range(1,len(filenames)):
			p = os.path.join(dirpath, filenames[i])
			os.remove(p)
