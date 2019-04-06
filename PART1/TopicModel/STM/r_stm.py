import pandas as pd

# r
from rpy2.robjects import r
import rpy2.robjects as robjects


# saves files:
#	- stm plot
#	- stm summary 
# 	- cov plot 
# 	- cov summary
# 	- topic-document frequency matrix
def run_stm(folder, csvfile):
	r('library(\'stm\')')
	r('poliblogs<-read.csv(\"'+csvfile+'\",header=TRUE)')

	# ---------------------------- RUN MODELS ---------------------------- # 
	# preprocess documents
	r("processed <- textProcessor(poliblogs$documents, metadata = poliblogs)")
	r("out <- prepDocuments(processed$documents, processed$vocab, processed$meta)")
	r("docs <- out$documents")
	r("vocab <- out$vocab")
	r("meta <-out$meta")

	# create STM
	r("First_STM <- stm(documents = out$documents, vocab = out$vocab," \
				"K = 30, prevalence =~ rating," \
				"max.em.its = 75, data = out$meta," \
				"init.type = \"Spectral\", verbose = FALSE)")

	# analyze results of covariates
	r("predict_topics<-estimateEffect(formula = ~ rating, stmobj = First_STM," \
		"metadata = out$meta, uncertainty = \"Global\")")

	# ---------------------------- SAVE FILES ---------------------------- # 
	print("saving files")
	r("dir.create(\""+folder+"\",showWarnings=FALSE,mode=\"0777\")")
	# save workspace
	r("save.image(file=\""+folder+"/my_work_space.RData\")")
	# saving stm plot
	r("pdf(file=\""+folder+"/"+"stm_plot.pdf\")")
	r("plt = plot(First_STM)")
	r("dev.off()")
	# saving stm summary
	r("capture.output(summary(First_STM), file=\""+folder+"/"+"stm_summary.txt\", append=FALSE)")
	# saving covariate plot 
	r("pdf(\""+folder+"/"+"cov_plot.pdf\")")
	r("plot(predict_topics, covariate = \"rating\", topics = 1:30," \
			"model = First_STM, method = \"difference\"," \
			"cov.value1 = \"left\", cov.value2 = \"right\"," \
			"xlab = \"More Conservative ... More Liberal\"," \
			"main = \"Effect of Liberal vs. Conservative\"," \
			"labeltype = \"custom\"," \
			"custom.labels = paste(\"Topic\",1:30))")
	r("dev.off()")
	# saving covariate summary
	r("capture.output(summary(predict_topics), file=\""+folder+"/"+"cov_summary.txt\", append=FALSE)")
	
	# ---------------------------- SAVE FREQ DATAFRAME ---------------------------- # 
	# saving topic-document frequency dataframe as csv
	r("freqs = data.frame(First_STM$theta)")
	r("write.csv(freqs, \""+folder+"/"+"td_frequencies.csv\")")

	# ---------------------------- SAVE COV DATAFRAME ---------------------------- # 
	# return dataframe of cov results
	r("summ = summary(predict_topics)")
	r("est <- predict_topics")

	r("coef <- se <- pval <- rep(NA, 30)")
	r("for (i in 1:30){" \
		"coef[i] <- est$parameters[[i]][[1]]$est[2]" \
		"\nse[i] <- est$parameters[[i]][[1]]$vcov[2,2]" \
		"\npval[i] <- summ$tables[[i]][8]}")

	r("df <- data.frame(topic = 1:30, coef=coef, se=se, pval=pval)")
	r("df <- df[order(df$coef),]")
	r("df[order(df$coef),]")

	r("write.csv(df, \""+folder+"/"+"covariates.csv\")")


	# df = robjects.r['df']
	# return df





if __name__ == '__main__':
	run_stm("sample")





