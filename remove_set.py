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
                        default="./score/fourset.input",
                        help="annotation input")
    parser.add_argument("-r", dest='ref',
                        default="./input/tvar_labels.gz",
                        help="annotation input")
    parser.add_argument("-o", dest='out_path',
                        default="./score_test",
                        help="annotation input")
    args = parser.parse_args()
    df = pd.read_csv(args.ref, sep='\t', index_col=[0, 1])
    df_tmp = df[~df.index.duplicated(keep='first')]

    for line in open(args.path, 'rt'):
        file_name = line.rstrip()
        file_name = file_name.replace("cadd13","cadd")
        print(file_name)
        out_file = './score_test/' + basename(file_name)
        if file_name.endswith('tvar'):
            df_ori = pd.read_csv(file_name, sep='\t', header=0)
        else:
            df_ori = pd.read_csv(file_name, sep='\t', header=None)
        #df_ori.reset_index(inplace=True)
        if not file_name.endswith('tvar'):
            df_ori.columns = ['chr', 'pos', 'score']
        df_ori['chr'] = df_ori['chr'].apply(lambda x: str(x).replace("chr", ""))
        df_ori.set_index(['chr', 'pos'], inplace=True)
        #print(df_ori.head(5))
        #print(df_ori.shape)
        df_ori = df_ori.loc[~df_ori.index.isin(df_tmp.index)]
        df_ori.to_csv(out_file, header=False, index=True, sep='\t')
        print(df_ori.shape)


if __name__ == "__main__":
    main()
