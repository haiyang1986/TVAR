import argparse
import sys
import os
import re
import tempfile
import gzip
import pandas as pd
import gzip
from os.path import splitext, basename, exists, abspath, isfile, getsize
from subprocess import check_output


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='iVariant v0.01.')
    parser.add_argument("-i", dest='path',
                        default="./score_test/eQTL_GTEx_pos_non_coding.tvar",
                        help="annotation input")
    parser.add_argument("-r", dest='ref',
                        default="./score/eQTL_GTEx_pos_non_coding.tvar",
                        help="annotation input")
    parser.add_argument("-o", dest='out_path',
                        default="./score/",
                        help="annotation input")
    args = parser.parse_args()
    df_ori = pd.read_csv(args.ref, sep='\t', header=0, index_col=[0, 1])
    df = pd.read_csv(args.path, sep='\t', header=None, index_col=[0, 1])
    cols = []
    for l in list(df_ori):
        cols.append(l)
    df.columns = cols
    df = df.T
    df.columns = list(map(lambda x: 'v' + str(x), range(df.shape[1])))
    out_file = args.ref +'_R'
    df.to_csv(out_file, header=True, index=True, sep='\t')


if __name__ == "__main__":
    main()
