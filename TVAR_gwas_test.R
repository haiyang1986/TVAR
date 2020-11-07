require(optparse)
options(stringsAsFactors=FALSE) 
option_list = list(
  make_option(c("-p", "--p_path"), action="store", default='./score_all/I25_pos.score', 
              type='character', help="back_table"),
  make_option(c("-q", "--q_path"), action="store", default='./score_all/I25_neg.score', 
              type='character', help="back_table")
)
opt = parse_args(OptionParser(option_list=option_list))
ta <- read.table(opt$p,sep="\t",header=FALSE)
a <- ta[,ncol(ta)]
tb <- read.table(opt$q,sep="\t",header=FALSE)
b <- tb[,ncol(tb)]
htest = wilcox.test(a, b, alternative = "greater")
cat(paste(opt$p,"\n",sep=''),file=stdout())
cat(paste("p-value of t test TVAR socres of GWAS and background: ",signif(htest$p.value,3),"\n",sep=''),file=stdout())