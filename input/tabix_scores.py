import argparse
import sys
import numpy as np

np.random.seed(1)
import os, time
import re
import tempfile
from tempfile import mkdtemp
import pickle
from subprocess import Popen, check_output
import pandas as pd
import gzip
from os.path import splitext, basename, exists, abspath, isfile, getsize


def exec_commands(cmds):
    if not cmds: return

    def done(p):
        return p.poll() is not None

    def success(p):
        return p.returncode == 0

    def fail():
        sys.exit(1)

    processes = []
    while True:
        while cmds:
            task = cmds.pop()
            processes.append(Popen(task, shell=True))
        for p in processes:
            if done(p):
                if success(p):
                    processes.remove(p)
                else:
                    fail()
        if not processes and not cmds:
            break
        else:
            time.sleep(0.05)


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='iVariant v0.01.')
    parser.add_argument("-i", dest='input', default="./tmp/gtex.avinput",
                        help="annotation input")
    parser.add_argument("-o", dest='out_path', default="./gtex.dvar", help="clinvar_pos")
    parser.add_argument("-l", dest='LINSIGHT_path', default=".../../data/scores/LINSIGHT_norm.score.gz",
                        help="annotation input")
    parser.add_argument("-d", dest='DVAR_path', default="../../data/scores/hg19_DVAR.score.gz",
                        help="annotation input")
    args = parser.parse_args()
    tmp_dir = './tmp/'
    chr_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
                 '19',
                 '20', '21', '22', 'X', 'Y', ]
    base_dt = dict.fromkeys(['A', 'T', 'G', 'C'], 1)
    base_name = splitext(basename(args.input))[0]
    tmp_file_us = tmp_dir + '/' + base_name + '.input'
    file_out = args.out_path
    tmp_out1 = "%s/%s.hg19_%s_dropped" % (tmp_dir, base_name, 'LINSIGHT')
    tmp_out2 = "%s/%s.hg19_%s_dropped" % (tmp_dir, base_name, 'dvar')
    df = pd.read_csv(args.input, sep='\t', header=None)
    df.columns = ['chr', 'start', 'pos', 'ref', 'alt', 'rs']
    df['chr'] = df['chr'].apply(lambda x: "chr" + str(x).replace("chr", ""))
    fp_us = open(tmp_file_us, 'w')
    for line in open(args.input):
        txt = line.rstrip().split('\t')
        chr_txt = txt[0].replace("chr", "")
        fp_us.write("%s\t%s\t%s\t%s\t%s\t%s\n" % (chr_txt, txt[2], txt[2], txt[3], txt[4], txt[5]))
    fp_us.close()
    cmd1 = 'sort -k1,1 -k2,2n %s -o %s' % (tmp_file_us, tmp_file_us)
    check_output(cmd1, shell=True)
    cmd1 = 'tabix %s -R %s >%s' % (
        args.LINSIGHT_path, tmp_file_us, tmp_out1
    )
    cmd2 = 'tabix %s -R %s >%s' % (
        args.DVAR_path, tmp_file_us, tmp_out2
    )
    check_output(cmd1, shell=True)
    check_output(cmd2, shell=True)
    df_dvar = pd.read_csv(tmp_out2, sep='\t', comment='#', header=None, low_memory=False)
    df_dvar.columns = ['chr', 'pos', 'DVAR']
    df_dvar['chr'] = df_dvar['chr'].apply(lambda x: "chr" + str(x).replace("chr", ""))
    df_LINSIGHT = pd.read_csv(tmp_out1, sep='\t', comment='#', header=None, low_memory=False)
    df_LINSIGHT.columns = ['chr', 'pos', 'LINSIGHT']
    df_LINSIGHT['chr'] = df_LINSIGHT['chr'].apply(lambda x: "chr" + str(x).replace("chr", ""))
    df_tvar = pd.merge(left=df_dvar, right=df_LINSIGHT, how='left', on=['chr', 'pos'])
    df_tvar = df_tvar.drop_duplicates(subset=['chr', 'pos'], keep='first')
    df_all = pd.merge(left=df, right=df_tvar, how='left', on=['chr', 'pos'])
    df_all = df_all.drop_duplicates(subset=['chr', 'pos', 'ref', 'alt'], keep='first')
    df_all['score'] = df_all[['DVAR', 'LINSIGHT']].mean(axis=1)
    df_all = df_all.loc[:, ['chr', 'pos', 'ref', 'alt', 'rs','LINSIGHT']]
    df_all.to_csv(file_out, header=False, index=False, sep='\t')
    cmd = "rm -f %s %s %s" % (tmp_out1, tmp_out2, tmp_file_us)
    # check_output(cmd, shell=True)


if __name__ == "__main__":
    main()
