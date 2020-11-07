import argparse
import sys
import numpy as np

np.random.seed(1)
import os, time
import math
import re
import pandas as pd
import gzip
from subprocess import check_output
from os.path import splitext, basename, exists, abspath, isfile, getsize
import pickle


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='iVariant v0.01.')
    parser.add_argument("-i", dest='input', default="./gtex.dvar",
                        help="annotation input")
    parser.add_argument("-l", dest='list', default="./gtex.list",
                        help="annotation input")
    parser.add_argument('-q', dest='cutoff', type=float, default=0.05, help="cutoff threshold")
    parser.add_argument('-s', dest='scut', type=float, default=0.6, help="cutoff threshold")
    parser.add_argument("-o", dest='out', default="./tvar_pos.label.gz", help="clinvar_pos")
    parser.add_argument('-t', dest='top', type=int, default=1000, help="cutoff threshold")
    args = parser.parse_args()
    chr_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                 '19',
                 '20', '21', '22', 'X', 'Y', ]
    nb_line = 0
    df_input = pd.read_csv(args.input, sep='\t', header=None, index_col=4)
    df_input.columns = ['chr', 'pos', 'ref', 'alt', 'DVAR']
    df_input = df_input[~df_input.index.duplicated(keep='first')]
    df_input['chr'] = df_input['chr'].apply(lambda x: str(x).replace("chr", ""))
    df_dict = df_input.loc[:, ['DVAR']]
    df_dict = df_dict.loc[:, ['DVAR']].fillna(0)
    df_high_score = df_dict.loc[df_dict['DVAR'] > args.scut, :]
    high_dict = dict([(i, 1) for i in df_high_score.index.values.tolist()])
    df_input = df_input.loc[:, ['chr', 'pos', 'ref', 'alt']]
    tissues_dict = {}
    for line in open(args.list, 'rt'):
        filename = line.rstrip()
        id = basename(filename).replace('.gz', '')
        tissues_dict[id] = 1
    var_set = set()
    var_all_dict = {}
    for line in open(args.list, 'rt'):
        line = line.rstrip()
        id = basename(line).replace('.gz', '')
        df = pd.read_csv(line, sep='\t', header=None, index_col=0)
        df = df.iloc[:, [0]]
        df.columns = ['p']
        df = df.loc[df.index.isin(high_dict), :]
        df = df.loc[df['p'] < args.cutoff, :]
        df['score'] = df_dict.loc[df.index.values, 'DVAR']
        ori_shape = df.shape[0]
        if df.shape[0] > args.top:
            df = df.nlargest(args.top, 'score')
        var_set = var_set | set(df.index.values.tolist())
        var_all_dict[id] = df.index.values.tolist()
        nb_line += 1
        print((nb_line, ori_shape, len(var_set)))
    labels = np.zeros((len(var_set), len(list(tissues_dict.keys()))), dtype=int)
    df_label = pd.DataFrame(data=labels, index=list(var_set), columns=list(tissues_dict.keys()))
    var_dict = dict([(i, 1) for i in var_set])
    for id in var_all_dict:
        for ix in var_all_dict[id]:
            df_label.at[ix, id] = 1
    df_input = df_input.loc[df_label.index, :]
    df_label = df_input.join([df_label])
    df_label = df_label.sort_values(by=['chr', 'pos'], ascending=[True, True])
    print((df_label.shape))
    df_label.to_csv(args.out, header=True, index=False, sep='\t', compression='gzip')
    # test
    # df_label = pd.read_csv(args.out, sep='\t', dtype={'chr': str})
    # mat_label = df_label.iloc[:, 4:].values
    # mat_label_shape = mat_label.shape
    # print mat_label_shape
    # # shuffle 1
    # # mat_label = mat_label.reshape((mat_label.size))
    # # np.random.shuffle(mat_label)
    # # mat_label = mat_label.reshape(mat_label_shape)
    # # # shuffle 2
    # for i in xrange(mat_label_shape[0]):
    #     vec_tmp = mat_label[i,:]
    #     np.random.shuffle(vec_tmp)
    #     mat_label[i,:] = vec_tmp
    # # # shuffle 3
    # # for i in xrange(mat_label_shape[1]):
    # #     vec_tmp = mat_label[:,i]
    # #     np.random.shuffle(vec_tmp)
    # #     mat_label[:,i] = vec_tmp
    # df_label.iloc[:, 4:] = mat_label
    # out_path = './uk_shuffled.label.gz'
    # df_label.to_csv(out_path, header=True, index=False, sep='\t', compression='gzip')


if __name__ == "__main__":
    main()
