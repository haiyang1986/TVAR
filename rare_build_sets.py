import argparse
import sys
import os
import numpy as np

np.random.seed(1)
import random
import h5py
import re
import string
import tempfile
import gzip
from tempfile import mkdtemp
from subprocess import check_output
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from os.path import splitext, basename, exists, abspath, isfile, getsize

class RARE_TEST(object):
    def __init__(self, out_path ='./output/', model_path='./model/', input='F20', nb_threads=1):
        self.nb_threads = nb_threads
        self.model_path = model_path
        self.out_path = out_path
        self.model = None
        self.tmp_dir = './tmp/'
        self.input = input
        self.b_decomposition = False
        # for the python map function
        self.anno_todo_list = []
        # './res/ensembl/splice_sites_final.bed']
        self.hg19 = None
        # bed file for the annotation
        self.df_bed = None
        # hash used to map tabix chr_pos to index
        self.bed_file = None
        # deep learning related parameters
        self.batch_size = 64
        self.nb_epoch = 30
        #self.model = linear_model.LogisticRegression(n_jobs=self.nb_threads)
        # tempfile.tempdir = './tmp'

    # make chunks for multi-threads
    def chunks(self, l, n):
        n = max(1, n)
        chunks_list = []
        nb_list = len(l)
        for i in range(0, n):
            start = int(nb_list / n) * i
            chunk_len = int(nb_list / n)
            if i == n - 1:
                chunk_len = nb_list - start
            chunks_list.append(l[start:start + chunk_len])
        return chunks_list



    def vcf2df(self, vcf):
        chr_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                     '19',
                     '20', '21', '22', 'X', 'Y']
        if vcf.endswith('.raw'):
            df = pd.read_table(vcf, comment='#', low_memory=False)
        elif vcf.endswith('.csv'):
            df = pd.read_csv(vcf, comment='#', low_memory=False, index_col=0)
        elif vcf.endswith('.pandas'):
            df = pd.read_feather(vcf, nthreads=2)
        elif vcf.endswith('.gz'):
            df = pd.read_csv(vcf, sep='\t', compression='gzip')
        else:
            df = pd.read_csv(vcf, sep='\t', header=None, comment='#', low_memory=False)
        df = df.iloc[:, [0, 1, 2, 3]]
        df.columns = ['chr', 'pos', 'ref', 'alt']
        df = df.astype(dtype={"chr": str, "pos": int, "ref": str, "alt": str})
        df.index = list(map(lambda x: 'v' + str(x), range(df.shape[0])))
        print(df.head(10))
        df['chr'] = df['chr'].apply(lambda x: str(x).replace("chr", ""))
        return df

    def df2bed(self, df_ref, flank=500):
        tmp_dir = mkdtemp()
        tmp_bed = tmp_dir + '/df.bed'
        tmp_vcf = tmp_dir + '/df.vcf'
        # df to tabix input file
        df = df_ref.copy()
        # df to bed
        df.insert(1, 'start', df['pos'] - 1)
        df['chr'] = df['chr'].apply(lambda x: 'chr'+str(x).replace('chr',''))
        df['id'] = df.index.values
        df.to_csv(tmp_vcf, index=False, header=False, sep='\t')
        if not flank == -1:
            cmd = "bedtools slop -i %s -g ./hg19.genome -l %d -r %d >%s" % (tmp_vcf, flank - 2, flank, tmp_bed)
            check_output(cmd, shell=True)
            return tmp_bed
        else:
            return tmp_vcf


    def fea_load(self, filename):
        h5f = h5py.File(filename, 'r')
        fea = h5f['fea'][:]
        label = h5f['label'][:]
        h5f.close()
        return fea, label


    def fea_save(self, filename, fea, label):
        h5f = h5py.File(filename, 'w')
        h5f.create_dataset('fea', data=fea)
        h5f.create_dataset('label', data=label)
        h5f.close()


    def rare_variants(self, disease):
        rare_dict={}
        for line in gzip.open('/data/soft/annovar/humandb/1kg_le5.txt.gz', 'rt'):
            txt = line.rstrip().split('\t')[4]
            rare_dict[txt] = 1
        top_n = 200
        if disease =='Heart_Left_Ventricle':
            rare_path = './eval/cad.UKBB.txt'
        elif disease =='Brain_Anterior_cingulate_cortex_BA24':
            rare_path = './eval/szc.pgc.txt'
        elif disease == 'Pancreas':
            rare_path = './eval/t2d.ng.txt'
        elif disease == 'Breast_Mammary_Tissue':
            rare_path = './eval/bc.Michailidou.txt'
        elif disease == 'Whole_Blood':
            rare_path = './env/eval/ibd.ng.txt'
        df = pd.read_csv(rare_path, sep='\t')
        df = df.loc[df['rsid'].isin(rare_dict), :]
        df = df.nsmallest(top_n, 'pval')
        print(df)
        rare_list = df.loc[:, 'rsid'].tolist()
        uk_rare_list =[]
        for rs in rare_list:
            if rs in rare_dict:
                uk_rare_list.append(rs)
        return uk_rare_list

    # feature extract
    def build_sets(self, file_input, neg_input, annovar_path):
        chr_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                     '19',
                     '20', '21', '22']
        df_train = pd.read_csv(file_input, sep='\t', compression='gzip', dtype={'chr': str, 'pos': int}).iloc[:, 0:2]
        #df_train.to_csv('./tmp/train_tmp.gz', index=False, header=True, sep='\t',compression='gzip')
        #df_train = pd.read_csv('./tmp/train_tmp.gz', sep='\t', compression='gzip', dtype={'chr': str, 'pos': int})
        train_bed = self.df2bed(df_train)
        base_dt = dict.fromkeys(['A', 'T', 'G', 'C'], 1)
        print(self.input)
        #['coronary artery disease', 'breast cancer']
        gwas_snps = self.rare_variants(self.input)
        print(len(gwas_snps))
        base_name = self.input
        rs_pos = self.out_path + base_name + '_rare.rs'
        file_annovar = self.out_path + base_name + '_rare_annovar.vcf'
        file_pos_all = self.out_path + base_name + '_rare_tmp.input'
        file_pos = self.out_path + base_name + '_rare_pos.gz'
        file_neg = self.out_path + base_name + '_rare_neg.gz'
        file_pos_bed = self.out_path + base_name + '_rare_pos.bed'
        file_neg_bed = self.out_path + base_name + '_rare_neg.bed'
        fp = open(rs_pos, 'w')
        for key in gwas_snps:
            fp.write("%s\n" % key)
        fp.close()
        cmd = 'perl %s/convert2annovar.pl -format rsid %s -dbsnpfile %s/humandb/hg19_snp142.txt > %s' % (
            annovar_path, rs_pos, annovar_path, file_annovar)
        check_output(cmd, shell=True)
        rs_dict = {}
        fp = open(file_pos_all, 'w')
        for line in open(file_annovar, 'rt'):
            txt = line.rstrip().split('\t')
            if txt[3] in base_dt and txt[4] in base_dt and txt[5] not in rs_dict:
                rs_dict[txt[5]] = 1
                chr_name = txt[0].replace("chr", "")
                nb_pos = int(txt[2])
                pos_end = nb_pos
                fp.write("%s\t%d\t%s\t%s\n" % (chr_name, pos_end, txt[3], txt[4]))
        fp.close()
        pos_tab = pd.read_csv(file_pos_all, sep='\t', header=None, dtype={0: str, 1: int})
        print(pos_tab.shape)
        pos_tab.columns = ['chr', 'pos', 'ref', 'alt']
        pos_tab = pos_tab.drop_duplicates(subset=['chr', 'pos'], keep='first')
        pos_bed = self.df2bed(pos_tab, -1)
        cmd = "bedtools intersect -a %s -b %s -v >%s" % (pos_bed, train_bed, file_pos_bed)
        check_output(cmd, shell=True)
        pos_tab = pd.read_csv(file_pos_bed, sep='\t', header=None, dtype={0: str, 1: int}).iloc[:, 0:5]
        pos_tab.columns = ['chr', 'start', 'pos', 'ref', 'alt']
        pos_tab.drop('start', axis=1, inplace=True)
        print(pos_tab.shape)
        pos_tab = pos_tab.sort_values(by=['chr', 'pos'], ascending=[True, True])
        nb_pos = pos_tab.shape[0]
        nb_neg = nb_pos * 3
        neg_tab = pd.read_csv(neg_input, sep='\t', header=None, dtype={0: str, 1: int})
        neg_tab.columns = ['chr', 'pos', 'ref', 'alt', 'rs']
        neg_tab = neg_tab.sample(nb_neg, random_state=1)
        neg_tab = neg_tab.drop_duplicates(subset=['chr', 'pos'], keep='first')
        print(neg_tab.shape)
        neg_bed = self.df2bed(neg_tab, -1)
        cmd = "bedtools intersect -a %s -b %s -v >%s" % (neg_bed, train_bed, file_neg_bed)
        check_output(cmd, shell=True)
        neg_tab = pd.read_csv(file_neg_bed, sep='\t', header=None).iloc[:, 0:5]
        neg_tab.columns = ['chr', 'start', 'pos', 'ref', 'alt']
        neg_tab.drop('start', axis=1, inplace=True)
        print(neg_tab.shape)
        neg_tab = neg_tab.merge(pos_tab.iloc[:, 0:2], on=['chr', 'pos'], how='left', indicator=True)
        neg_tab = neg_tab.loc[neg_tab['_merge'] == 'left_only', ['chr', 'pos', 'ref', 'alt']]
        print(neg_tab.shape)
        if neg_tab.shape[0] > nb_pos:
            neg_tab = neg_tab.sample(nb_pos, random_state=0)
        print(neg_tab.shape)
        neg_tab = neg_tab.sort_values(by=['chr', 'pos'], ascending=[True, True])
        neg_tab.index = range(len(neg_tab.index))
        pos_tab['chr'] = pos_tab['chr'].apply(lambda x: str(x).replace("chr", ""))
        neg_tab['chr'] = neg_tab['chr'].apply(lambda x: str(x).replace("chr", ""))
        pos_tab.to_csv(file_pos, header=True, index=False, sep='\t', compression='gzip')
        neg_tab.to_csv(file_neg, header=True, index=False, sep='\t', compression='gzip')
        cmd4 = "rm -f %s %s %s" % (pos_bed, neg_bed, train_bed)
        check_output(cmd4, shell=True)
        return True


    def impute(self, X):
        for i in range(X.shape[1]):
            X[np.isnan(X[:, i]), i] = self.nan_list[i]
        return X



    def fit(self, X, y):
        self.model.fit(X, y)
        fp = open(self.model_path + '/EHR.model', 'w')
        pickle.dump(self.model, fp)
        fp.close()
        return True



    def distance_score(self, X, mean):
        # sample number
        score = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            # mix number
            for j in range(mean.shape[0]):
                score[i] += la.norm(X[i, :] - mean[j, :], 2)
        return score

def norm_0_1(a):
    stra = str(a)
    if stra.startswith('1|0') or stra.startswith('0|1') or stra.startswith('1|1'):
        return 1
    else:
        return 0

def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='GWAS_test v0.01.')
    parser.add_argument("-g", dest='genotype', default="/scratch/cgg/weiq1/result/BioVU-06162017/vcf/merge_imputed",
                        help="genotype input")
    parser.add_argument("-r", dest='annovar_path', default="/data/soft/annovar/",
                        help="annotation input")
    parser.add_argument('-e', dest='extend_len', type=int, default=50, help="vcf flop value")
    parser.add_argument("-i", dest='input', default="Whole_Blood", help="run_mode: train, test")
    parser.add_argument("-m", dest='run_mode', default="fea", help="run_mode: train, test")
    parser.add_argument("-f", dest='file_input', default="./input/tvar_labels.gz", help="file input")
    parser.add_argument("-n", dest='neg_input', default="/data/soft/annovar/humandb/1kg_le5.txt.gz", help="file input")
    parser.add_argument("-o", dest='out_path', default="./eval/", help="file output")
    parser.add_argument("-t", dest='threads_num', type=int, default=1, help="threads num")
    args = parser.parse_args()
    model_path = './model/'
    tmp_dir ='./tmp/'
    rare = RARE_TEST(args.out_path, model_path, args.input)
    # feature extract
    if args.run_mode == 'fea':
        rare.build_sets(args.file_input, args.neg_input, args.annovar_path)


if __name__ == "__main__":
    main()
