require(optparse)
require(Hmisc)
require(ggcorrplot)
library(gplots)
library(plyr)
options(stringsAsFactors=FALSE) 
option_list = list(
  make_option(c("-b", "--b_path"), action="store", default='./tvar_all.score', 
              type='character', help="back_table")
)
opt = parse_args(OptionParser(option_list=option_list))
df_b <- read.table(opt$b_path,sep="\t",header=TRUE)
df_b <- df_b[ ,3:50]
#heatmap.2(as.matrix(df_b), dendrogram="col")
res <- rcorr(as.matrix(df_b), type = c("pearson"))
corr <- as.matrix(res$r)
corr
corr_mean <- mean(corr)
corr_mean
#corr
# ggcorrplot(corr, method = "circle", legend.title = "Correlation")
