import argparse
import sys
import numpy as np

np.random.seed(1)
import os, time
import math
import re
import pickle
import h5py
import pandas as pd
import gzip
from subprocess import check_output
from os.path import splitext, basename, exists, abspath, isfile, getsize


def cv(y, k):
    y_index = np.arange(y.shape[0])
    v_all = set(y_index.tolist())
    ids = []
    test_set = np.zeros((y.shape[0], k), dtype=int)
    for i in range(k):
        ids.append(set())
    col_names = list(y)
    # marks_all: all the positive samples
    marks_all = set()
    for i in range(y.shape[1]):
        # new positive samples
        marks_new = set(y_index[y[:, i] == 1])
        marks = np.array(list(marks_new - marks_all))
        n = len(marks)
        if n == 0:
            continue
        marks_all = marks_all | set(marks)
        assignments = np.array((n // k + 1) * list(range(1, k + 1)))
        assignments = assignments[:n]
        for j in range(k):
            ids[j] = ids[j] | set(marks[assignments != j + 1])
    v_neg = np.array(list(v_all - marks_all))
    n = len(v_neg)
    for i in range(k):
        assignments = np.array((n // k + 1) * list(range(1, k + 1)))
        assignments = assignments[:n]
        #merge pos and neg samples
        select_ids = list(ids[i] | set(v_neg[assignments != i + 1]))
        test_set[select_ids, i] = 1
    return test_set


def data_load(filename):
    h5f = h5py.File(filename, 'r')
    y = h5f['data'][:]
    h5f.close()
    return y


def data_save(filename, y):
    h5f = h5py.File(filename, 'w')
    h5f.create_dataset('data', data=y)
    h5f.close()


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='iVariant v0.01.')
    parser.add_argument("-i", dest='input', default="./tvar_pos.label.gz",
                        help="annotation input")
    parser.add_argument("-r", dest='list', default="/data/soft/annovar/humandb/1kg_gt5.txt",
                        help="annotation input")
    parser.add_argument("-o", dest='out', default="./tvar_labels.gz", help="clinvar_pos")
    parser.add_argument("-v", dest='cv_out', default="./tvar_cv.np", help="clinvar_pos")
    args = parser.parse_args()
    chr_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                 '19',
                 '20', '21', '22', 'X', 'Y', ]
    nb_line = 0
    tmp_dir = './tmp/'
    out_path = './tmp/'
    df_pos = pd.read_csv(args.input, sep='\t', dtype={'chr': str, 'pos':int})
    df_pos = df_pos.drop_duplicates(subset=['chr', 'pos'], keep='first')
    print(df_pos.shape)
    nb_pos = df_pos.shape[0]
    nb_neg = nb_pos * 2
    neg_tab = pd.read_csv(args.list, sep='\t', header=None, dtype={0: str, 1: int})
    neg_tab.columns = ['chr', 'pos', 'ref', 'alt', 'rs']
    neg_tab = neg_tab.sample(nb_neg, random_state=1)
    neg_tab = neg_tab.drop_duplicates(subset=['chr', 'pos'], keep='first')
    neg_tab = neg_tab.merge(df_pos.iloc[:, 0:2], on=['chr', 'pos'], how='left', indicator=True)
    neg_tab= neg_tab.loc[neg_tab['_merge'] == 'left_only',['chr', 'pos', 'ref', 'alt']]
    print(neg_tab.shape)
    if neg_tab.shape[0] > nb_pos:
        neg_tab = neg_tab.sample(nb_pos, random_state=0)
    neg_tab.index = range(len(neg_tab.index))
    print(neg_tab.shape)
    neg_tab_labels = pd.DataFrame(data=np.zeros(df_pos.shape, dtype=int), columns=list(df_pos)).iloc[:,4:]
    df_neg = pd.concat([neg_tab, neg_tab_labels], axis=1)
    print(df_neg.shape)
    df_all = pd.concat([df_pos, df_neg], axis=0)
    df_all['chr'] = df_all['chr'].apply(lambda x: str(x).replace("chr", ""))
    df_all = df_all.loc[df_all['chr'].isin(chr_names), :]
    df_all = df_all.sort_values(by=['chr', 'pos'], ascending=[True, True])
    print(df_all.shape)
    # df_labels = df_all.iloc[:, 4:]
    # df_labels = df_labels.loc[:, df_labels.sum().sort_values(ascending=True).index]
    # df_all.iloc[:, 4:] = df_labels
    df_all = df_all.drop_duplicates(subset=['chr', 'pos'], keep='first')
    print(df_all.shape)
    df_all.index = range(len(df_all.index))
    df_all.to_csv(args.out, header=True, index=False, sep='\t', compression='gzip')
    df_all = pd.read_csv(args.out, sep='\t', compression='gzip')
    y = df_all.iloc[:, 4:].as_matrix().astype(int)
    del df_all
    labels = cv(y, 5)
    print((labels.shape))
    # print labels
    data_save(args.cv_out, labels)


if __name__ == "__main__":
    main()
