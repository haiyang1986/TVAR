import argparse
import sys
import os
import re
import tempfile
import gzip
import pandas as pd
from os.path import splitext, basename, exists, abspath, isfile, getsize


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='iVariant v0.01.')
    parser.add_argument("-i", dest='path',
                        default="/scratch/cgg/chenr6/Database/Summary_statistics/IBD/ibd_build37_59957_20161107.txt.fmt.rsid1",
                        help="annotation input")
    parser.add_argument("-o", dest='out_path',
                        default="/fs0/yangh8/env/TVAR/env/eval/ibd.ng.txt",
                        help="annotation input")
    args = parser.parse_args()
    df = pd.read_csv(args.path, sep='\t').loc[:,['snp','pvalue']]
    df.columns = ['rsid', 'pval']
    df = df.loc[df['pval'] < 0.05, :]
    df = df.sort_values(by=['pval'], ascending=[True])
    df.drop_duplicates(['rsid'], keep='first', inplace=True)
    df = df[df['rsid'].apply(lambda x: x.startswith('rs'))]
    df.to_csv(args.out_path, index=False, header=True, sep='\t')


if __name__ == "__main__":
    main()
