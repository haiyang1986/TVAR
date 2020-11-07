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


def filesize(filename):
    if os.path.isfile(filename):
        return getsize(filename)
    else:
        return -1


def test_sub_finished(filename, keyword):
    while not isfile(filename):
        time.sleep(60)
    while True:
        with open(filename) as f:
            all_lines = f.readlines()
            if len(all_lines) > 0:
                last = all_lines[-1]
                if keyword in last:
                    break
        time.sleep(60)


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='iVariant v0.01.')
    parser.add_argument("-m", dest='run_mode', default="all",
                        help="annotation input")
    parser.add_argument("-i", dest='input', default="",
                        help="annotation input")
    args = parser.parse_args()
    old_path = abspath("./")
    env_path = './'
    gpu_path = './'
    eval_path = './'
    if args.input == '':
        fea_input = '%s/input/input.list' % (env_path)
    else:
        fea_input = args.input
    log_file = './tvar.log'
    cmd = 'rm -f %s' % (log_file)
    check_output(cmd, shell=True)
    #
    # if args.run_mode == 'pca' or args.run_mode == 'all':
    #     background_list = '/fs0/yangh8/DVAR/input/train.input'
    #     cmd1 = 'python TVar_cpu.py -m fea_train -i %s -t 8' % (background_list)
    #     print(cmd1)
    #     check_output(cmd1, shell=True)

    if args.run_mode == 'fea' or args.run_mode == 'all':
        for line in open(fea_input):
            cmd = 'python TVar_cpu.py -m fea -i %s -t 8' % (line.rstrip())
            # check_output(cmd, shell=True)

    # if args.run_mode == 'cv' or args.run_mode == 'all':
    #     cmds = []
    #     train_input = './input/train.list'
    #     for line in open(train_input):
    #         cmd = 'python TVar_cpu.py -m fea -i %s -t 8' % (line.rstrip())
    #         check_output(cmd, shell=True)
    #     cmd1 = "python TVar_gpu.py -m cv"
    #     cmds.append(cmd1)
    #     for cmd in cmds:
    #         check_output(cmd, shell=True)
    #     cmd = 'python TVar_cpu.py -m cv'
    #     check_output(cmd, shell=True)
    #     print("CV OK!")
    #     return

    if args.run_mode == 'train' or args.run_mode == 'all' or args.run_mode == 'train_score':
        cmds = []
        train_input = './input/train.list'
        for line in open(train_input):
            cmd = 'python TVar_cpu.py -m fea -i %s -t 8' % (line.rstrip())
            # check_output(cmd, shell=True)
        cmd1 = "python TVar_gpu.py -m train"
        # check_output(cmd1, shell=True)

    if args.run_mode == 'score' or args.run_mode == 'all' or args.run_mode == 'train_score':
        cmds = []
        for line in open(fea_input):
            cmd = "python TVar_gpu.py -m score -i %s" % (line.rstrip())
            cmds.append(cmd)
        # for cmd in cmds:
        #     check_output(cmd, shell=True)

    # if args.run_mode == 'rare' or args.run_mode == 'all' or args.run_mode == 'train_score':
    #     clinvar_input = './input/rare.input'
    #     for line in open(clinvar_input):
    #         cmd = 'python TVar_cpu.py -m fea -i %s -t 8' % (line.rstrip())
    #         #check_output(cmd, shell=True)
    #     cmds = []
    #     for line in open(clinvar_input):
    #         tissue = basename(line.rstrip()).replace("_rare_neg.gz", "")
    #         tissue = tissue.replace("_rare_pos.gz", "")
    #         cmd = "python TVar_gpu.py -m rare -n %s -i %s" % (tissue, line.rstrip())
    #         cmds.append(cmd)
    #     for cmd in cmds:
    #         check_output(cmd, shell=True)
    #     compare_files = []
    #     cmd = "rm -f ./eval/rare.log"
    #     check_output(cmd, shell=True)
    #     for line in open(clinvar_input):
    #         base_file = splitext(basename(line.rstrip()))[0]
    #         score_file = './score/' + base_file + '.tvar'
    #         compare_files.append(score_file)
    #         if len(compare_files) == 2:
    #             cmd = 'Rscript TVAR_gwas_test.R -p %s -q %s >> ./eval/rare.log' % (compare_files[0], compare_files[1])
    #             check_output(cmd, shell=True)
    #             compare_files.clear()
    #
    # if args.run_mode == 'gwas' or args.run_mode == 'all' or args.run_mode == 'train_score':
    #     clinvar_input = './input/gwas.input'
    #     for line in open(clinvar_input):
    #         cmd = 'python TVar_cpu.py -m fea -i %s -t 8' % (line.rstrip())
    #         #check_output(cmd, shell=True)
    #     cmds = []
    #     for line in open(clinvar_input):
    #         tissue = basename(line.rstrip()).replace("_gwas_neg.gz", "")
    #         tissue = tissue.replace("_gwas_pos.gz", "")
    #         cmd = "python TVar_gpu.py -m gwas -n %s -i %s" % (tissue, line.rstrip())
    #         cmds.append(cmd)
    #     for cmd in cmds:
    #         check_output(cmd, shell=True)
    #     compare_files = []
    #     cmd = ''
    #     check_output(cmd, shell=True)
    #     cmd = "rm -f ./eval/gwas.log"
    #     check_output(cmd, shell=True)
    #     for line in open(clinvar_input):
    #         base_file = splitext(basename(line.rstrip()))[0]
    #         score_file = './score/' + base_file + '.tvar'
    #         compare_files.append(score_file)
    #         if len(compare_files) == 2:
    #             cmd = 'Rscript TVAR_gwas_test.R -p %s -q %s >> ./eval/gwas.log' % (compare_files[0], compare_files[1])
    #             check_output(cmd, shell=True)
    #             compare_files.clear()

    # if args.run_mode == 'eval' or args.run_mode == 'all' or args.run_mode == 'merge':
    #     cmd = 'python remove_set.py'
    #     check_output(cmd, shell=True)
    #     cmd0 = 'python compare.py -m eval > four_sets.log'
    #     check_output(cmd0, shell=True)

    if args.run_mode == 'comp' or args.run_mode == 'all' or args.run_mode == 'train_score':
        clinvar_input = './input/rare.input'
        compare_files = []
        log_file = './eval/rare.log'
        cmd = 'rm -f %s' % (log_file)
        check_output(cmd, shell=True)
        for line in open(clinvar_input):
            base_file = splitext(basename(line.rstrip()))[0]
            score_file = './score/' + base_file + '.deepsea'
            compare_files.append(score_file)
            if len(compare_files) == 2:
                cmd = 'Rscript TVAR_gwas_test.R -p %s -q %s >> %s' % (compare_files[0], compare_files[1], log_file)
                check_output(cmd, shell=True)
                compare_files.clear()


if __name__ == "__main__":
    main()
