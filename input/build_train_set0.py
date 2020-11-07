import argparse
import sys
import numpy as np

np.random.seed(1)
import os, time
import math
import re
import tempfile
from tempfile import mkdtemp
from subprocess import Popen, check_output
import pandas as pd
import gzip
from os.path import splitext, basename, exists, abspath, isfile, getsize


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='iVariant v0.01.')
    parser.add_argument("-r", dest='annovar_path', default="/data/soft/annovar/",
                        help="annotation input")
    parser.add_argument("-i", dest='input', default="./gtex.list",
                        help="annotation input")
    parser.add_argument("-o", dest='out_path', default="./rs_all.list", help="clinvar_pos")
    parser.add_argument('-p', dest='pvalue', type=float, default=1e-5, help="pvalue threshold")
    parser.add_argument('-c', dest='cutoff', type=float, default=0.1, help="cutoff threshold")
    args = parser.parse_args()
    chr_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                 '19',
                 '20', '21', '22', 'X', 'Y', ]
    nts = ['A', 'C', 'G', 'T']
    numbers =[]
    set_all = None
    nb_line =0
    tmp_dir = './tmp/'
    out_path = './tmp/'
    for line in open(args.input, 'rt'):
        file = line.rstrip()
        df = pd.read_csv(file, sep='\t', header=None)
        nb_line += 1
        df.columns = ['rs', 'qval']
        #df = df[df['qval'] < args.cutoff]
        set_tissue = set(df['rs'].values.tolist())
        if nb_line == 1:
            set_all = set_tissue
        else:
            set_all = set_all | set_tissue
    print(len(set_all))
    base_name = splitext(basename(args.input))[0]
    rs_file = out_path + base_name + '.rs'
    file_av = out_path + base_name + '.avinput'
    file_us = out_path + base_name + '.input'
    fp = open(rs_file, 'w')
    for key in set_all:
        fp.write("%s\n" % (key))
    fp.close()
    # anno variants
    cmd = 'perl %s/convert2annovar.pl -format rsid %s -dbsnpfile %s/humandb/hg19_snp142.txt > %s' % (
        args.annovar_path, rs_file, args.annovar_path, file_av)
    check_output(cmd, shell=True)
    df_vcf = pd.read_csv(file_av, sep='\t', header=None, usecols=[0,1,2,3,4,5])
    df_vcf.columns = ['chr', 'pos', 'end', 'ref', 'alt', 'rs']
    df_vcf.drop('end', inplace=True, axis=1)
    df_vcf['chr'] = df_vcf['chr'].apply(lambda x: str(x).replace("chr", ""))
    df_vcf = df_vcf[df_vcf['chr'].isin(chr_names)]
    df_vcf = df_vcf[df_vcf['ref'].isin(nts)]
    df_vcf = df_vcf[df_vcf['alt'].isin(nts)]
    df_vcf = df_vcf.sort_values(by=['chr', 'pos'], ascending=[True, True])
    df_vcf.to_csv(file_us, header=False, index=False, sep='\t')


if __name__ == "__main__":
    main()
