import argparse
import sys
import numpy as np

np.random.seed(1)
import string
import os
from tempfile import mkdtemp
from subprocess import check_output
import matplotlib.pyplot as plt
from pathos.pools import ProcessPool, ThreadPool
import h5py
import re
import pandas as pd
import pickle
from os.path import splitext, basename, exists, abspath, isfile, getsize
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
from sklearn import preprocessing
from itertools import cycle
from scipy import interp
from scipy import stats
from scipy.sparse import csc_matrix, save_npz, load_npz


class TVAR(object):
    def __init__(self, nb_threads=1, model_path='./model/', flop_len=1000, cv_len=5):
        self.nb_threads = nb_threads
        self.model_path = model_path
        self.score_path = './score'
        # self.score_path = '/fs0/yangh8/tmp/'
        self.k_mer_len = 6
        self.fa_ref = './hg19.fa'
        self.cv_len = cv_len
        self.gene_top_n = 2
        self.gene_cutoff = 100000
        # for the python map function
        # for the python map function
        self.anno_todo_list = []
        self.chr_todo_list = []
        # mean of the whole genome score list for impute
        self.nan_list = [1.79467, -0.0721653, 0.0878948, 0.128081, 0.101462, 0.0357264, 0.0451349, 0.105336, 4.62506]
        # TSS source
        self.other_list = ['../data/gerp.bed']
        self.gene_exp_file = '../data/GTEx_RNASeq/GTEx_RNASeq.txt'
        self.dis_list = [
            '../data/res/encode/TSS_human_with_gencodetss_notlow_ext50eachside_merged_withgenctsscoord_andgnlist.sorted.bed']
        # '/fs0/yangh8/env/for_wangq_revisit/res/ensembl/splice_sites_final.bed']
        self.score_list = ['../data/scores/Gerp/gerp_scores.tsv.gz',
                           '../data/scores/phastCons/phastCons46way.placental.tsv.gz',
                           '../data/scores/phastCons/phastCons46way.primates.tsv.gz',
                           '../data/scores/phastCons/phastCons100way.tsv.gz',
                           '../data/scores/phyloP/phyloP46way.placental.tsv.gz',
                           '../data/scores/phyloP/phyloP46way.primate.tsv.gz',
                           '../data/scores/phyloP/phyloP100way.tsv.gz',
                           '../data/scores/SiPhy/SiPhy_scores.tsv.gz']

        # for feature train
        self.roadmap_path = '../../data/roadmap/'
        self.fantom5_path = './input/FANTOM5.list'
        self.chromhmm_path = './input/chromhmm.list'
        self.gtex_path = './input/gtex.list'
        self.compare_path = './input/uk_labels.score'
        self.hg19 = None
        # bed file for the annotation
        self.df_bed = None
        # hash used to map tabix chr_pos to index
        self.bed_file = None
        # deep learning related parameters
        self.epochs = 30
        self.mat_len = 500
        self.batch_size = 64

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
        if df.shape[1] >= 4:
            df = df.iloc[:, [0, 1, 2, 3]]
            df.columns = ['chr', 'pos', 'ref', 'alt']
            df = df.astype(dtype={"chr": str, "pos": int, "ref": str, "alt": str})
        else:
            df = df.iloc[:, [0, 1]]
            df.columns = ['chr', 'pos']
            df = df.astype(dtype={"chr": str, "pos": int})
        df.index = list(map(lambda x: 'v' + str(x), range(df.shape[0])))
        print(df.head(10))
        df['chr'] = df['chr'].apply(lambda x: str(x).replace("chr", ""))
        return df

    # vcf slop to bed with chrom size check
    def df2files(self, df_ref, flank=500):
        df = df_ref.copy()
        tmp_dir = mkdtemp()
        tmp_bed = tmp_dir + '/input.bed'
        tmp_bed_vcf = tmp_dir + '/input2.bed'
        tmp_vcf = tmp_dir + '/vcf.bed'
        tmp_tabix = tmp_dir + '/tmp.tab'
        tmp_fa = tmp_dir + '/input.fa'
        df.to_csv(tmp_tabix, index=False, header=False, sep='\t')
        df.insert(1, 'start', df['pos'] - 1)
        df['chr'] = df['chr'].apply(lambda x: 'chr' + str(x))
        df['id'] = df.index.values
        df.to_csv(tmp_vcf, index=False, header=False, sep='\t')
        if not flank == -1:
            cmd = "bedtools slop -i %s -g ./hg19.genome -l %d -r %d >%s" % (tmp_vcf, flank - 2, flank, tmp_bed)
            check_output(cmd, shell=True)
            cmd2 = "bedtools slop -i %s -g ./hg19.genome -l %d -r %d >%s" % (
                tmp_vcf, flank / 2 - 2, flank / 2, tmp_bed_vcf)
            check_output(cmd2, shell=True)
        # df to fa
        cmd = "bedtools getfasta -fi %s -bed %s -fo %s -name" % (self.fa_ref, tmp_bed_vcf, tmp_fa)
        # check_output(cmd, shell=True)
        return tmp_bed, tmp_fa, tmp_vcf, tmp_tabix

    def build_roadmap_matrix(self):
        roadmap_anno_list = []
        tissue_dict = {'E017': 'IMR90', 'E002': 'ES cell', 'E008': 'ES cell', 'E001': 'ES cell', 'E015': 'ES cell',
                       'E014': 'ES cell', 'E016': 'ES cell', 'E003': 'ES cell', 'E024': 'ES cell', 'E020': 'iPSC',
                       'E019': 'iPSC', 'E018': 'iPSC', 'E021': 'iPSC', 'E022': 'iPSC', 'E007': 'ES-deriv',
                       'E009': 'ES-deriv', 'E010': 'ES-deriv', 'E013': 'ES-deriv', 'E012': 'ES-deriv',
                       'E011': 'ES-deriv', 'E004': 'ES-deriv', 'E005': 'ES-deriv', 'E006': 'ES-deriv',
                       'E062': 'Blood &T cell', 'E034': 'Blood &T cell', 'E045': 'Blood &T cell',
                       'E033': 'Blood &T cell', 'E044': 'Blood &T cell', 'E043': 'Blood &T cell',
                       'E039': 'Blood &T cell', 'E041': 'Blood &T cell', 'E042': 'Blood &T cell',
                       'E040': 'Blood &T cell', 'E037': 'Blood &T cell', 'E048': 'Blood &T cell',
                       'E038': 'Blood &T cell', 'E047': 'Blood &T cell', 'E029': 'HSC &B cell',
                       'E031': 'HSC &B cell', 'E035': 'HSC &B cell', 'E051': 'HSC &B cell', 'E050': 'HSC &B cell',
                       'E036': 'HSC &B cell', 'E032': 'HSC &B cell', 'E046': 'HSC &B cell', 'E030': 'HSC &B cell',
                       'E026': 'Mesench.', 'E049': 'Mesench.', 'E025': 'Mesench.', 'E023': 'Mesench.',
                       'E052': 'Myosat.', 'E055': 'Epithelial', 'E056': 'Epithelial', 'E059': 'Epithelial',
                       'E061': 'Epithelial',
                       'E057': 'Epithelial', 'E058': 'Epithelial', 'E028': 'Epithelial', 'E027': 'Epithelial',
                       'E054': 'Neurosph.', 'E053': 'Neurosph.', 'E112': 'Thymus', 'E093': 'Thymus', 'E071': 'Brain',
                       'E074': 'Brain', 'E068': 'Brain', 'E069': 'Brain', 'E072': 'Brain', 'E067': 'Brain',
                       'E073': 'Brain',
                       'E070': 'Brain', 'E082': 'Brain', 'E081': 'Brain', 'E063': 'Adipose', 'E100': 'Muscle',
                       'E108': 'Muscle', 'E107': 'Muscle', 'E089': 'Muscle', 'E090': 'Muscle', 'E083': 'Heart',
                       'E104': 'Heart', 'E095': 'Heart', 'E105': 'Heart', 'E065': 'Heart',
                       'E078': 'Smooth muscle', 'E076': 'Smooth muscle', 'E103': 'Smooth muscle',
                       'E111': 'Smooth muscle',
                       'E092': 'Digestive', 'E085': 'Digestive', 'E084': 'Digestive', 'E109': 'Digestive',
                       'E106': 'Digestive',
                       'E075': 'Digestive', 'E101': 'Digestive', 'E102': 'Digestive', 'E110': 'Digestive',
                       'E077': 'Digestive', 'E079': 'Digestive', 'E094': 'Digestive', }
        for key in tissue_dict.keys():
            core1_file = self.roadmap_path + '-'.join([key, 'H3K4me3.narrowPeak.gz'])
            core2_file = self.roadmap_path + '-'.join([key, 'H3K4me1.narrowPeak.gz'])
            core3_file = self.roadmap_path + '-'.join([key, 'H3K27me3.narrowPeak.gz'])
            core4_file = self.roadmap_path + '-'.join([key, 'H3K9me3.narrowPeak.gz'])
            core5_file = self.roadmap_path + '-'.join([key, 'H3K36me3.narrowPeak.gz'])
            dnase_file = self.roadmap_path + '-'.join([key, 'DNase.macs2.narrowPeak.gz'])
            roadmap_anno_list += [core1_file, core2_file, core3_file, core4_file, core5_file, dnase_file]
        return roadmap_anno_list

    def bed_annotate(self, threads_id=-1):
        df = None
        tmp_dir = mkdtemp()
        # multi-thread support
        if not threads_id == -1:
            tmp_bed = tmp_dir + str(threads_id) + '_tmp.bed'
            todo_list = self.anno_todo_list[threads_id]
        else:
            tmp_bed = tmp_dir + 'tmp.bed'
            todo_list = self.anno_todo_list
        series = []
        for file in todo_list:
            if isfile(file):
                cmd = "bedtools intersect -wb -a %s -b %s >%s" % (self.bed_file, file, tmp_bed)
                check_output(cmd, shell=True)
                if not isfile(tmp_bed) or getsize(tmp_bed) == 0:
                    dt_val = dict.fromkeys(['v0'], 0)
                else:
                    df = pd.read_table(tmp_bed, header=None)
                    dt_val = dict((k, v) for k, v in zip(df.iloc[:, 3].values.tolist(), df.iloc[:, -1].values.tolist()))
            else:
                dt_val = dict.fromkeys(['v0'], 0)
            series.append(pd.Series(dt_val, name=splitext(basename(file))[0]))
        df = pd.concat(series, axis=1, sort=True)
        cmd_del = "rm -fr %s" % tmp_dir
        check_output(cmd_del, shell=True)
        # check_output(cmd_del, shell=True)
        return df

    # annotation with bed files
    def norm01_annotate(self, threads_id=-1):
        df = None
        tmp_dir = mkdtemp()
        # multi-thread support
        if not threads_id == -1:
            tmp_bed = tmp_dir + str(threads_id) + '_tmp.bed'
            todo_list = self.anno_todo_list[threads_id]
        else:
            tmp_bed = tmp_dir + 'tmp.bed'
            todo_list = self.anno_todo_list
        series = []
        for file in todo_list:
            if isfile(file):
                cmd = "bedtools intersect -wb -a %s -b %s >%s" % (self.bed_file, file, tmp_bed)
                check_output(cmd, shell=True)
                if not isfile(tmp_bed) or getsize(tmp_bed) == 0:
                    dt_val = dict.fromkeys(['v0'], 0)
                else:
                    df = pd.read_table(tmp_bed, header=None)
                    dt_val = dict.fromkeys(df.iloc[:, 3].values, 1)
            else:
                dt_val = dict.fromkeys(['v0'], 0)
            series.append(pd.Series(dt_val, name=splitext(basename(file))[0]))
        df = pd.concat(series, axis=1, sort=True)
        cmd_del = "rm -fr %s" % tmp_dir
        check_output(cmd_del, shell=True)
        # check_output(cmd_del, shell=True)
        return df

    def gtex_annotation(self, bed_file):
        file_todo = []
        file_todo_paths = []
        for file in open(self.gtex_path):
            file_todo_paths.append(file.rstrip())
            file_todo.append(splitext(basename(file.rstrip()))[0])
        threads_num = min(self.nb_threads, len(file_todo_paths))
        self.anno_todo_list = self.chunks(file_todo_paths, threads_num)
        pool = ThreadPool(threads_num)
        self.bed_file = bed_file
        ano_mat = pool.map(self.bed_annotate, range(threads_num))
        df = pd.concat(ano_mat, axis=1, sort=True)
        del ano_mat
        return df

    # annotation with other resources
    def chromhmm_annotation(self, bed_file):
        file_todo = []
        file_todo_paths = []
        hmm_dict = {'E15': 0.0, 'E9': 1.0 / 14, 'E5': 2.0 / 14, 'E14': 3.0 / 14, 'E8': 4.0 / 14, 'E4': 5.0 / 14,
                    'E7': 6.0 / 14, 'E6': 7.0 / 14, 'E2': 8.0 / 14,
                    'E3': 9.0 / 14, 'E1': 10.0 / 14, 'E13': 11.0 / 14, 'E12': 12.0 / 14, 'E11': 13.0 / 14, 'E10': 1.0}
        for file in open(self.chromhmm_path):
            file_todo_paths.append(file.rstrip())
            file_todo.append(splitext(basename(file.rstrip()))[0])
        threads_num = min(self.nb_threads, len(file_todo_paths))
        self.anno_todo_list = self.chunks(file_todo_paths, threads_num)
        pool = ThreadPool(threads_num)
        self.bed_file = bed_file
        ano_mat = pool.map(self.bed_annotate, range(threads_num))
        df = pd.concat(ano_mat, axis=1, sort=True)
        df = df.applymap(lambda x: 0.0 if (pd.isnull(x)) else hmm_dict[x])
        del ano_mat
        return df

    # annotation with other resources
    def fantom5_annotation(self, bed_file):
        file_todo = []
        file_todo_paths = []
        for file in open(self.fantom5_path):
            file_todo_paths.append(file.rstrip())
            file_todo.append(splitext(basename(file.rstrip()))[0])
        threads_num = min(self.nb_threads, len(file_todo_paths))
        self.anno_todo_list = self.chunks(file_todo_paths, threads_num)
        pool = ThreadPool(threads_num)
        self.bed_file = bed_file
        ano_mat = pool.map(self.bed_annotate, range(threads_num))
        df = pd.concat(ano_mat, axis=1, sort=True)
        del ano_mat
        return df

    # roadmap annotation
    def roadmap_annotate(self, bed_file):
        file_todo = []
        file_todo_paths = self.build_roadmap_matrix()
        for file in file_todo_paths:
            file_todo.append(splitext(basename(file.rstrip()))[0])
        threads_num = min(self.nb_threads, len(file_todo_paths))
        self.anno_todo_list = self.chunks(file_todo_paths, threads_num)
        pool = ThreadPool(threads_num)
        self.bed_file = bed_file
        ano_mat = pool.map(self.norm01_annotate, range(threads_num))
        df = pd.concat(ano_mat, axis=1, sort=True)
        del ano_mat
        return df

    def score_load(self, filename):
        h5f = h5py.File(filename, 'r')
        y = h5f['data'][:]
        h5f.close()
        return y

    def score_save(self, filename, y):
        h5f = h5py.File(filename, 'w')
        h5f.create_dataset('data', data=y)
        h5f.close()

    def data_load(self, filename):
        h5f = h5py.File(filename, 'r')
        y = h5f['data'][:]
        h5f.close()
        return y

    def data_save(self, filename, y):
        h5f = h5py.File(filename, 'w')
        h5f.create_dataset('data', data=y)
        h5f.close()

    # annotation with bed files
    def bed_annotate_s(self, df_ref, bed_file, ref_files, b_decomposition=False):
        file_labels = []
        labels_done = []
        file_todo = []
        file_todo_paths = []
        df_ori = None
        for filename in open(ref_files):
            base_file = splitext(basename(filename))[0]
            file_labels.append(base_file)
            file_todo.append(base_file)
            file_todo_paths.append(filename.rstrip())
        threads_num = min(self.nb_threads, len(file_todo_paths))
        self.anno_todo_list = self.chunks(file_todo_paths, threads_num)
        pool = ThreadPool(threads_num)
        self.bed_file = bed_file
        ano_mat = pool.map(self.norm01_annotate, range(threads_num))
        df = df_ref.join(ano_mat)
        del ano_mat
        df = df.loc[:, file_todo].fillna(0)
        if b_decomposition:
            mat = df.values.astype(int)
            fp = open(self.model_path + '/fea1.model', 'rb')
            fea1_model = pickle.load(fp, encoding='latin1')
            fp.close()
            mat = fea1_model.transform(mat)
            df = pd.DataFrame(data=mat, index=df.index,
                              columns=map(lambda x: 'annotation' + str(x), range(mat.shape[1])))
        return df

    def score_annotation_s(self, df_ref, tabix_file):
        tmp_dir = mkdtemp()
        file_labels = []
        labels_done = []
        file_todo = []
        file_todo_paths = []
        df = None
        for filename in self.score_list:
            base_file = splitext(basename(filename))[0]
            file_labels.append(base_file)
            file_todo.append(base_file)
            file_todo_paths.append(filename.rstrip())
        if len(file_todo_paths) > 0:
            threads_num = min(self.nb_threads, len(file_todo_paths))
            self.bed_file = tabix_file
            self.score_list = file_todo_paths
            pool = ThreadPool(threads_num)
            ano_mat = pool.map(self.bed_intersect_tib, range(threads_num))
            df = pd.concat(ano_mat, axis=1, sort=True)
            df.reset_index(inplace=True)
            df.rename(columns={0: 'chr', 1: 'pos'}, inplace=True)
            df = df.astype({"chr": str, "pos": int})
            df_tmp = df_ref.copy()
            df_tmp['vid'] = df_ref.index.values
            df = df.merge(df_tmp, on=['chr', 'pos'], how='right')
            df.index = df['vid'].values
            df.drop(columns=['chr', 'pos', 'vid'], inplace=True)
            # df.loc[ids == -1, :] = np.nan
            df.iloc[:, :] = self.impute(df.values)
            df.columns = list(map(lambda x: 'conservation' + str(x), range(df.shape[1])))
            # df.index = df_ref.index.values
        cmd_del = "rm -fr %s" % tmp_dir
        check_output(cmd_del, shell=True)
        return df

    # annotation with other resources
    def other_annotation_s(self, df_ref, vcf_file):
        dfs = []
        self.bed_file = vcf_file
        for file in self.dis_list:
            dfs.append(self.genome_distance(file))
        df = df_ref.join(dfs)
        df = df.fillna(0).iloc[:, 2:]
        return df

    # annotation with bed files
    def gene_annotation_s(self, df_ref, vcf_file):
        chr_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                     '19',
                     '20', '21', '22', 'X', 'Y']
        self.bed_file = vcf_file
        threads_num = min(self.nb_threads, len(chr_names))
        self.chr_todo_list = self.chunks(chr_names, threads_num)
        pool = ThreadPool(threads_num)
        ano_mat = pool.map(self.gene_annotation, range(threads_num))
        df = pd.concat(ano_mat, axis=0)
        del ano_mat
        return df

    def gene_annotation(self, threads_id=-1):
        if not threads_id == -1:
            todo_list = self.chr_todo_list[threads_id]
        else:
            todo_list = self.chr_todo_list
        df_genes_all = pd.read_csv(self.gene_exp_file, sep='\t', dtype={0: str, 1: str, 2: int})
        all_list = list(df_genes_all)[4:]
        drop_list = ['Bladder', 'Cervix - Ectocervix', 'Cervix - Endocervix', 'Fallopian Tube', 'Kidney - Cortex']
        anno_list = [x for x in all_list if x not in drop_list]
        df_genes = {}
        df_vcf_all = pd.read_csv(self.bed_file, sep='\t', dtype={0: str, 1: int, 2: int}, header=None)
        df_vcf_all.columns = ['chr', 'start', 'pos', 'id']
        df_vcf_all['chr'] = df_vcf_all['chr'].apply(lambda x: str(x).replace("chr", ""))
        df_vcf_all = df_vcf_all[df_vcf_all['chr'].isin(todo_list)]
        for chr in todo_list:
            df_tmp = df_genes_all.loc[df_genes_all['chr'] == chr, :]
            df_genes[chr] = df_tmp
        df_out = pd.DataFrame(data=np.zeros((df_vcf_all.shape[0], len(anno_list)), dtype=float), columns=anno_list,
                              index=df_vcf_all['id'].values)
        nb_line = 0
        for idx, row in df_vcf_all.iterrows():
            chr = row['chr']
            df_dis = (df_genes[chr]['pos'] - row['pos']).abs()
            # df_dis = df_dis[df_dis < self.gene_cutoff]
            # if df_dis.shape[0] ==0:
            #     exps = np.zeros((len(anno_list)))
            # else:
            #     df_sort = df_genes[chr].iloc[df_dis.argsort()[: 1]]
            #     exps = df_sort.loc[:, anno_list].mean(axis=0).values
            # df_dis = df_dis[df_dis.where(df_dis < self.gene_cutoff)]
            # print(df_dis.shape[0])
            df_sort = df_genes[chr].iloc[df_dis.argsort()[: min(self.gene_top_n, df_dis.shape[0])]]
            df_sort = df_sort.loc[:, anno_list]
            exps = df_sort.mean(axis=0).values
            for i in range(len(exps)):
                df_out.iat[nb_line, i] = exps[i]
            nb_line += 1
        return df_out

    # annotation with tibix file
    def bed_intersect_tib(self, threads_id=-1):
        tmp_dir = mkdtemp()
        df = None
        # multi-thread support
        if not threads_id == -1:
            tmp_bed = tmp_dir + str(threads_id) + '_tmp.tab'
            file = self.score_list[threads_id]
            cmd = "tabix %s -R %s >%s" % (file, self.bed_file, tmp_bed)
            check_output(cmd, shell=True)
            if not isfile(tmp_bed) or getsize(tmp_bed) == 0:
                df_tmp = pd.read_csv(self.bed_file, sep='\t', header=None, low_memory=False)
                if threads_id == 0:
                    df_tmp['score1'] = self.nan_list[threads_id]
                    df_tmp['score2'] = self.nan_list[threads_id + 1]
                else:
                    df_tmp['score1'] = self.nan_list[threads_id + 1]
                df_tmp.to_csv(tmp_bed, index=False, header=False, sep='\t')
            df_tmp = pd.read_csv(tmp_bed, sep='\t', comment='N', header=None, index_col=[0, 1], low_memory=False)
            df = df_tmp[~df_tmp.index.duplicated(keep='first')]
        else:
            tmp_bed = tmp_dir + 'tmp.tab'
            dfs = []
            nb_score_file = 0
            for file in self.score_list:
                cmd = "tabix %s -R %s >%s" % (file, self.bed_file, tmp_bed)
                check_output(cmd, shell=True)
                if not isfile(tmp_bed) or getsize(tmp_bed) == 0:
                    df_tmp = pd.read_csv(self.bed_file, sep='\t', header=None, low_memory=False)
                    if nb_score_file == 0:
                        df_tmp['score1'] = self.nan_list[nb_score_file]
                        df_tmp['score2'] = self.nan_list[nb_score_file + 1]
                    else:
                        df_tmp['score1'] = self.nan_list[nb_score_file + 1]
                    df_tmp.to_csv(tmp_bed, index=False, header=False, sep='\t')
                df_tmp = pd.read_csv(tmp_bed, sep='\t', comment='N', header=None, dtype={0: str, 1: int})
                df_tmp = df_tmp.drop_duplicates(subset=['chr', 'pos'], keep='first')
                df_tmp = df_tmp[~df_tmp.index.duplicated(keep='first')]
                dfs.append(df_tmp)
            df = pd.concat(dfs, axis=1, sort=True)
        cmd_del = "rm -fr %s" % tmp_dir
        check_output(cmd_del, shell=True)
        return df

    def genome_distance(self, b_file):
        tmp_dir = mkdtemp()
        sort_bed = tmp_dir + '/sorted.bed'
        tmp_bed = tmp_dir + '/out.bed'
        cmd1 = 'sort -k1,1 -k2,2n %s -o %s' % (self.bed_file, sort_bed)
        check_output(cmd1, shell=True)
        cmd2 = "bedtools closest -a %s -b %s -D b > %s" % (sort_bed, b_file, tmp_bed)
        check_output(cmd2, shell=True)
        if not isfile(tmp_bed) or getsize(tmp_bed) == 0:
            dt_val = dict.fromkeys(['v0'], 0)
        else:
            df = pd.read_table(tmp_bed, header=None)
            dt_val = dict(zip(df.iloc[:, 3].values, df.iloc[:, -1].values.astype(float)))
        cmd_del = "rm -fr %s" % tmp_dir
        check_output(cmd_del, shell=True)
        return pd.Series(dt_val, name=splitext(basename(b_file))[0])

    # feature extract
    def fea_extract(self, df_ori, save_file, annotation_list='./input/beds.list'):
        df = df_ori.loc[:, ['chr', 'pos']]
        bed, fa, vcf, tabix = self.df2files(df, flank=int(self.mat_len))
        X1 = self.bed_annotate_s(df, bed, annotation_list, True)
        X2 = self.score_annotation_s(df, tabix)
        X3 = self.other_annotation_s(df, vcf)
        X4 = self.gene_annotation_s(df, vcf)
        # X4 =self.chromhmm_annotation(bed)
        # X5 = self.fasta2mat(fa)
        fea = df_ori.join([X1, X2, X3, X4])
        # fea = df_ori.join([X4])
        # print(fea.shape)
        # fea = df_ori.join([X5])
        fea.iloc[:, 4:] = fea.iloc[:, 4:].fillna(0)
        fea.index = range(len(fea.index))
        print(df_ori.shape[0], fea.shape)
        fea.to_hdf(save_file, 'data', mode='w')
        cmd4 = "rm -fr %s" % os.path.dirname(bed)
        check_output(cmd4, shell=True)
        print("feature extract finished!")
        return True

    def impute(self, X):
        for i in range(X.shape[1]):
            X[np.isnan(X[:, i]), i] = self.nan_list[i]
        return X

    def fea1_train(self, X):
        fea1_model = PCA(n_components=96)
        fea1_model.fit(X)
        fp = open(self.model_path + '/fea1.model', 'wb')
        pickle.dump(fea1_model, fp)
        fp.close()
        return True

    def compare_cv(self, dvar_file, divan_file, y, ids_dict, ids_ori_dict, cv_labels=None, file_save='compare_out.png',
                   b_plot=True):
        plt.figure(1)
        task_plt_ids = [221, 222, 223, 224]
        df_dvar = pd.read_csv(dvar_file, sep='\t')
        df_divan = pd.read_csv(divan_file, sep='\t')
        score_dvar = {'cadd13': 'CADD', 'dann': 'DANN', 'eigen': 'Eigen'}
        score_ids = {'I25': 'CAD', '20001_1002': 'BC', '20002_1223': 'T2D', 'F20': 'SC'}
        disease_ids = {'I25': 'Coronary Artery Disease', '20001_1002': 'Breast Cancer', '20002_1223': 'Type 2 Diabetes',
                       'F20': 'Schizophrenia'}
        scores = {}
        k = cv_labels.shape[1]
        methods_all = ['NVAR', 'DIVAN', 'CADD', 'Eigen', 'DANN']
        mean_fpr = {}
        mean_tpr = {}
        for method in methods_all:
            mean_fpr[method] = np.linspace(0, 1, 100)
        for method in score_dvar.keys():
            scores[score_dvar[method]] = df_dvar.loc[:, method].values.astype(float)
        plt_id = 0
        for disease in ids_dict.keys():
            id = ids_dict[disease]
            id_ori = ids_ori_dict[disease]
            scores['DIVAN'] = df_divan.loc[:, score_ids[disease]].values.astype(float)
            probas = {}
            tprs = {}
            for method in methods_all:
                tprs[method] = []
            for i in range(1, k + 1):
                res_save = self.score_path + '/scores.cv' + str(i)
                ix = cv_labels[:, i - 1] == 0
                y_test = y[ix, id]
                probas['NVAR'] = self.score_load(res_save)[:, id_ori]
                for key in scores.keys():
                    probas[key] = scores[key][ix]
                for method in probas.keys():
                    if np.count_nonzero(y_test) > 10:
                        fpr, tpr, thresholds = roc_curve(y_test, probas[method])
                        tprs[method].append(interp(mean_fpr[method], fpr, tpr))
                        tprs[method][-1][0] = 0.0
            for method in probas.keys():
                mean_tpr[method] = np.mean(tprs[method], axis=0)
                mean_tpr[method][-1] = 1.0
            if b_plot:
                self.show_roc_auc(mean_fpr, mean_tpr, methods_all, plt, task_plt_ids[plt_id], disease_ids[disease])
            plt_id += 1
        plt.tight_layout()
        plt.show()
        # plt.savefig(file_save)

    def auc_roc(self, y, score_file='', file_save='clinvar_out.png', b_plot=True):
        res_save = score_file
        base_file = splitext(basename(score_file))[0].upper()
        probas_ = self.score_load(res_save)
        # probas_ = load_npz(score_file).toarray()
        probas_ = np.max(probas_, axis=1)
        fpr, tpr, thresholds = roc_curve(y, probas_)
        auc_val = auc(fpr, tpr)
        print(auc_val)
        if b_plot:
            lw = 2
            plt.plot(fpr, tpr, color='darkorange',
                     lw=lw, label='NVAR (AUC = %0.3f)' % auc_val)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--', label='Luck (AUC = 0.500)')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(base_file)
            plt.legend(loc="lower right")
            plt.savefig(file_save)

    def eval(self, y, thrs, file_save='clinvar_out.png', b_plot=True):
        res_save = self.score_path + '/uk_labels.score'
        thr_save = self.score_path + '/uk_labels.thr.npy'
        norm_save = self.score_path + '/uk_labels.norm'
        probas_ = self.score_load(res_save)
        probas_norm = np.zeros(probas_.shape, dtype=int)
        nb_phewas = probas_.shape[1]
        aucs = []
        auc_norms = []
        thresholds = []
        thrs = np.load(thr_save)
        for j in range(y.shape[1]):
            aucs.append([])
            auc_norms.append([])
            thresholds.append([])
        for j in range(nb_phewas):
            # probas = probas_[:, j]
            # y_phewas = y[:, j]
            # aucs[j] = roc_auc_score(y_phewas, probas)
            # fpr, tpr, threshold = roc_curve(y_phewas, probas)
            # thresholds[j] = threshold[np.argmax(tpr - fpr)]
            # thresholds[j] = thrs[j]
            # i = np.arange(len(tpr))
            # roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
            # roc_t = roc.ix[(roc.tf - 0).abs().argsort()[:1]]
            # thresholds[j] = list(roc_t['threshold'])[0]
            # i = np.arange(len(tpr))
            # roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
            # roc_t = roc.ix[(roc.tf - 0).abs().argsort()[:1]]
            # thresholds[j] = roc_t['threshold']
            probas_tmp = np.where(probas_[:, j] > thrs[j], 1, 0)
            probas_norm[:, j] = probas_tmp
            # auc_norms[j] = roc_auc_score(y_phewas, probas_tmp)
            # print(thresholds[j], aucs[j], auc_norms[j])
        # print(np.min(aucs), np.max(aucs), np.mean(aucs))
        # print(np.min(auc_norms), np.max(auc_norms), np.mean(auc_norms))

        self.score_save(norm_save, probas_norm)

    # Cross Validation and shows the performances

    def fit_cv(self, X, y, k, labels=None, thr_file=None, file_save='train_out.png', b_plot=False):
        n = y.shape[0]
        tprs = []
        all_aucs = []
        auc_mean = []
        mean_fpr = np.linspace(0, 1, 100)
        if labels is not None:
            k = labels.shape[1]
        aucs = []
        auc_out = []
        thrs = []
        thr_out = np.zeros((y.shape[1]))
        for j in range(y.shape[1]):
            aucs.append([])
            thrs.append([])
        for i in range(1, k + 1):
            ix = labels[:, i - 1] == 0
            y_test = y[ix]
            res_save = self.score_path + '/scores.cv' + str(i)
            probas_ = self.score_load(res_save)
            # print(probas_.shape)
            nb_phewas = probas_.shape[1]
            for j in range(nb_phewas):
                probas = probas_[:, j]
                y_phewas = y_test[:, j]
                if np.count_nonzero(y_phewas) > 10:
                    fpr_a, tpr_a, threshold_a = roc_curve(y_phewas, probas)
                    thrs[j].append(threshold_a[np.argmax(tpr_a - fpr_a)])
                    aucs[j].append(roc_auc_score(y_phewas, probas))
            y_test = y_test.reshape((y_test.size))
            probas_ = probas_.reshape((probas_.size))
            # auc_mean.append(roc_auc_score(y_test, probas_))
            fpr, tpr, thresholds = roc_curve(y_test, probas_)
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
        for j in range(y.shape[1]):
            auc_out.append(np.mean(aucs[j]))
            thr_out[j] = np.mean(thrs[j])
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        print(np.min(auc_out), np.max(auc_out), mean_auc)
        print(auc_out)
        i_top = np.argsort(auc_out)[::-1]
        label_file = './input/tvar_labels.gz'
        df_label = pd.read_csv(label_file, sep='\t', compression='gzip')
        label_list = list(df_label)[4:]
        fp = open("auc.log", 'wt')
        for j in range(y.shape[1]):
            fp.write("\'%s\' (AUC=%.3f)\n" % (label_list[i_top[j]], auc_out[i_top[j]]))
        fp.close()
        # print(thr_out)
        # if thr_file is not None:
        #     np.save(thr_file, thr_out)
        # if b_plot:
        #     plt.plot([0, 1], [0, 1], '--', color='navy', label='Luck')
        #     plt.plot(mean_fpr, mean_tpr, 'k--', color='darkorange',
        #              label='(AUC = %0.3f)' % mean_auc, lw=2)
        #     plt.ylim([0.0, 1.05])
        #     plt.xlim([0.0, 1.0])
        #     plt.xlabel('False Positive Rate')
        #     plt.ylabel('True Positive Rate')
        #     plt.title('Variants prediction (ROC) with 4-fold cross validation')
        #     plt.legend(loc="lower right")
        #     plt.savefig(file_save)

    # show the performances of different approaches
    def show_pr_auc(self, y_list, label_list, name_list, plt, sub_id, title_txt='Variants prediction (ROC)'):
        colors = cycle(['blue', 'darkorange', 'seagreen', 'red', 'cyan', 'indigo'])
        plt.subplot(sub_id)
        for i in range(len(y_list)):
            fpr, tpr, thresholds = precision_recall_curve(label_list[i].ravel(), y_list[i].ravel())
            roc_all = average_precision_score(label_list[i], y_list[i])
            plt.plot(tpr, fpr, color=colors.next(), label='%s (AUC = %0.3f)' % (name_list[i], roc_all))
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(title_txt)
        plt.legend(loc="lower right", fontsize=10, fancybox=True).get_frame().set_alpha(0.5)
        plt.text(-0.14, 1.075, string.ascii_lowercase[sub_id - 221], fontsize=14, weight='bold')

    # show the performances of different approaches
    def show_roc_auc(self, fpr, tpr, methods_list, plt, sub_id, title_txt='Variants prediction (ROC)'):
        # for i in methods_list:
        #     roc_all = auc(fpr[i], tpr[i])
        #     print(i, roc_all)
        colors = cycle(['black', 'blue', 'darkorange', 'seagreen', 'red', 'cyan', 'indigo'])
        plt.subplot(sub_id)
        for i in methods_list:
            roc_all = auc(fpr[i], tpr[i])
            plt.plot(fpr[i], tpr[i], color=next(colors), label='%s (AUC = %0.3f)' % (i, roc_all))
        plt.xlim([0, 1.0])
        plt.ylim([0, 1.1])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title_txt)
        plt.legend(loc="lower right", fontsize=10, fancybox=True).get_frame().set_alpha(0.5)
        plt.text(-0.14, 1.075, string.ascii_lowercase[sub_id - 221], fontsize=14, weight='bold')


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='NVAR v0.1.')
    parser.add_argument("-m", dest='run_mode', default="eval", help="run_mode: train, test")
    parser.add_argument("-i", dest='file_input', default="./input/tvar_labels.gz", help="file input")
    parser.add_argument("-o", dest='output_path', default="./score/", help="file output")
    parser.add_argument('-e', dest='extend_len', type=int, default=1000, help="vcf flop value")
    parser.add_argument("-c", dest='cv', help="Cross Validation k")
    parser.add_argument("-b", dest='base', default="dvar", help="Cross Validation k")
    parser.add_argument("-t", dest='threads_num', type=int, default=8, help="threads num")
    parser.add_argument("-v", dest='cv_out', default="./input/tvar_cv.np", help="clinvar_pos")
    args = parser.parse_args()
    model_path = './model/'
    tvar = TVAR(args.threads_num, model_path, args.extend_len, args.cv)

    if args.run_mode == 'fea':
        base_file = splitext(basename(args.file_input))[0]
        fea_save_file = './fea/' + base_file + '.fea'
        # df need to be sorted first
        df = tvar.vcf2df(args.file_input)
        tvar.fea_extract(df, fea_save_file)

    # pca train
    elif args.run_mode == 'fea_train':
        base_file = splitext(basename(args.file_input))[0]
        # fea_save_file = './fea/' + base_file + '.fea'
        df = tvar.vcf2df(args.file_input)
        df = df.loc[:, ['chr', 'pos']]
        bed, fa, vcf, tabix = tvar.df2files(df, flank=int(tvar.mat_len))
        X1 = tvar.bed_annotate_s(df, bed, './input/beds.list', False)
        tvar.fea1_train(X1.values)


    elif args.run_mode == 'cv':
        print('CV started!')
        base_file = splitext(basename(args.file_input))[0]
        fea_file = './fea/' + base_file + '.fea'
        thr_file = './score/' + base_file + '.thr.npy'
        if isfile(args.file_input) and isfile(fea_file):
            print('CV data loading...')
            df_label = pd.read_csv(args.file_input, sep='\t', compression='gzip')
            print('CV data loaded!')
        else:
            print('run fea model first!')
            return
        y = df_label.iloc[:, 4:].as_matrix().astype(int)
        del df_label
        cv_labels = tvar.data_load(args.cv_out)
        print(cv_labels.shape)
        print(y.shape)
        tvar.fit_cv(None, y, 5, cv_labels, thr_file)

    elif args.run_mode == 'norm':
        print('Eval started!')
        base_file = splitext(basename(args.file_input))[0]
        fea_file = './fea/' + base_file + '.fea'
        thr_file = './score/' + base_file + '.thr.npy'
        thrs = np.load(thr_file)
        if isfile(args.file_input) and isfile(fea_file):
            print('Eval data loading...')
            df_label = pd.read_csv(args.file_input, sep='\t', compression='gzip')
            print('Eval data loaded!')
        else:
            print('run fea model first!')
            return
        y = df_label.iloc[:, 4:].as_matrix().astype(int)
        tvar.eval(y, thrs)

    elif args.run_mode == 'clinvar':
        print('Eval clinvar started!')
        base_file = splitext(basename(args.file_input))[0]
        fea_file = './fea/' + base_file + '.fea'
        score_file = './score/' + base_file + '.score'
        if isfile(args.file_input) and isfile(fea_file):
            print('Eval data loading...')
            df_label = pd.read_csv(args.file_input, sep='\t', compression='gzip')
            print('Eval data loaded!')
        else:
            print('run fea model first!')
            return
        y = df_label.iloc[:, 4:].as_matrix().astype(int)
        tvar.auc_roc(y, score_file)

    elif args.run_mode == 'compare':
        ICD10s = ['I25', '20001_1002', '20002_1223', 'F20']
        # only keep 4 diseases
        ids = {'I25': 0, '20001_1002': 1, '20002_1223': 2, 'F20': 3}
        # original 2000 diseases
        ids_ori = {}
        print('compare started!')
        base_file = splitext(basename(args.file_input))[0]
        dvar_file = './' + base_file + '.score'
        divan_file = './' + base_file + '.divan'
        if isfile(args.file_input):
            print('CV data loading...')
            df_label = pd.read_csv(args.file_input, sep='\t', compression='gzip')
            label_list = list(df_label)[4:]
            print('CV data loaded!')
        else:
            print('run fea model first!')
            return
        y = df_label.loc[:, ICD10s].values.astype(int)
        del df_label
        for icd in ICD10s:
            for i in range(len(label_list)):
                if label_list[i] == icd:
                    ids_ori[icd] = i
        cv_labels = tvar.data_load(args.cv_out)
        print(y.shape, cv_labels.shape)
        tvar.compare_cv(dvar_file, divan_file, y, ids, ids_ori, cv_labels)

    elif args.run_mode == 'ToR':
        file_in = '/fs0/yangh8/env/TNET/env/input/uk_labels.gz'
        base_file = splitext(basename(file_in))[0]
        score_file = '/fs0/yangh8/env/TNET/env/score/' + base_file + '.norm'
        label_out = '/fs0/yangh8/env/TNET/env/eval/' + base_file + '.label'
        score_out = '/fs0/yangh8/env/TNET/env/eval/' + base_file + '.score'
        print(isfile(file_in), isfile(score_file))
        if isfile(file_in) and isfile(score_file):
            df_label = pd.read_csv(file_in, sep='\t', compression='gzip')
            np_scores = tvar.score_load(score_file)
        else:
            print('Loading error!')
            return
        y = df_label.iloc[:, 4:]
        y = y.iloc[:, -50:]
        del df_label
        # y_score = pd.DataFrame(data=np_scores, dtype=np.float, columns=list(y))
        y_score = pd.DataFrame(data=np_scores[:, -50:], dtype=np.float, columns=list(y))
        y.to_csv(label_out, index=False, header=True, sep='\t')
        y_score.to_csv(score_out, index=False, header=True, sep='\t')


if __name__ == "__main__":
    main()
