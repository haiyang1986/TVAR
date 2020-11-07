import argparse
import sys
import numpy as np

np.random.seed(1)
import os, time
import math
import re
import pandas as pd
import gzip
from os.path import splitext, basename, exists, abspath, isfile, getsize
from subprocess import check_output

def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='iVariant v0.01.')
    parser.add_argument("-i", dest='input', default="./GTEx_all.list",
                        help="annotation input")
    parser.add_argument("-o", dest='out', default="../../data/GTEx/", help="clinvar_pos")
    parser.add_argument('-c', dest='cutoff', type=float, default=0.1, help="cutoff threshold")
    args = parser.parse_args()
    chr_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                 '19',
                 '20', '21', '22', 'X', 'Y', ]
    nb_line =0
    tmp_dir = './tmp/'
    out_path = './tmp/'
    for line in open(args.input, 'rt'):
        filename = line.rstrip()
        id = basename(filename).replace('.v8.egenes.txt.gz', '')
        df = pd.read_csv(filename, sep='\t')
        df  =df.loc[:, ['rs_id_dbSNP151_GRCh38p7','qval']]
        df.columns = ['id', 'qval']
        df = df.sort_values('qval', ascending=True).drop_duplicates('id')
        out_path = args.out + id
        print(out_path)
        df.to_csv(out_path, header=False, index=False, sep='\t')
        cmd2 = "gzip -f %s" % (out_path)
        check_output(cmd2, shell=True)




if __name__ == "__main__":
    main()
