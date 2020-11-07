import argparse
import sys
import os, time
import gzip
import h5py
import numpy as np
import math
np.random.seed(1)
import fileinput
from subprocess import Popen, check_output
from collections import OrderedDict, Mapping, namedtuple, Sized
from functools import partial, reduce
import operator
from itertools import zip_longest, product
from os.path import splitext, basename, exists, abspath, isfile, getsize


def grouper(n, iterable, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(fillvalue=fillvalue, *args)


class ParameterGrid(object):

    def __init__(self, param_grid):
        if isinstance(param_grid, Mapping):
            param_grid = [param_grid]
        self.param_grid = param_grid

    def __iter__(self):

        for p in self.param_grid:
            # Always sort the keys of a dictionary
            items = sorted(p.items())
            if not items:
                yield {}
            else:
                keys, values = zip(*items)
                for v in product(*values):
                    params = dict(zip(keys, v))
                    yield params

    def __len__(self):
        """Number of points on the grid."""
        product = partial(reduce, operator.mul)
        return sum(product(len(v) for v in p.values()) if p else 1
                   for p in self.param_grid)

    def __getitem__(self, ind):
        for sub_grid in self.param_grid:
            if not sub_grid:
                if ind == 0:
                    return {}
                else:
                    ind -= 1
                    continue

            # Reverse so most frequent cycling parameter comes first
            keys, values_lists = zip(*sorted(sub_grid.items())[::-1])
            sizes = [len(v_list) for v_list in values_lists]
            total = np.product(sizes)

            if ind >= total:
                # Try the next grid
                ind -= total
            else:
                out = {}
                for key, v_list, n in zip(keys, values_lists, sizes):
                    ind, offset = divmod(ind, n)
                    out[key] = v_list[offset]
                return out
        raise IndexError('ParameterGrid index out of range')


class parallel_do(object):
    def __init__(self, mode='local', cmd='', nb_threads=1, b_run=False):
        self.nb_threads = nb_threads
        self.b_run = b_run
        self.grid_params = {}
        self.mode = mode
        self._ori_cmd = cmd
        self.cmd = ""
        self.input = ""
        self.tmp_files = []
        self.out_path = ""
        self.output = []
        self.tmp_dir = './tmp/'
        self.out_label = '-o'
        self.in_label = 'i'

    def local_run(self, cmds):
        if not cmds: return

        def done(p):
            return p.poll() is not None

        def success(p):
            return p.returncode == 0

        processes = []
        while True:
            while cmds and len(processes) < self.nb_threads:
                task = cmds.pop()
                processes.append(Popen(task, shell=True))
            for p in processes:
                if done(p):
                    if success(p):
                        processes.remove(p)
                    else:
                        processes.remove(p)
            if not processes and not cmds:
                break
            else:
                time.sleep(0.05)

    def chunks_file(self, file, n):
        base_name = basename(file)
        with open(file) as f:
            list = f.readlines()
        n = max(1, n)
        chunks_list = []
        nb_list = len(list)
        for i in range(0, n):
            out_file = self.tmp_dir + base_name + "." + str(i)
            start = int(nb_list / n) * i
            chunk_len = int(nb_list / n)
            if i == n - 1:
                chunk_len = nb_list - start
            with open(out_file, 'w') as f:
                sub_list = list[start:start + chunk_len]
                f.write(','.join(sub_list))
        return True

    def chunks(self, list, n):
        n = max(1, n)
        chunks_list = []
        nb_list = len(list)
        for i in range(0, n):
            start = int(nb_list / n) * i
            chunk_len = int(nb_list / n)
            if i == n - 1:
                chunk_len = nb_list - start
            chunks_list.append(list[start:start + chunk_len])
        return chunks_list

    def h5_load(self, filename):
        h5f = h5py.File(filename, 'r')
        X = h5f['data'][:]
        h5f.close()
        return X

    def h5_save(self, filename, X):
        h5f = h5py.File(filename, 'w')
        h5f.create_dataset('data', data=X)
        h5f.close()

    def split_file(self, filepath):
        path, filename = os.path.split(filepath)
        basename, ext = os.path.splitext(filename)
        num_lines = sum(1 for line in open(filepath))
        # calculate the chunk number
        n = int(math.ceil(float(num_lines)/ self.nb_threads))
        file_list = []
        with open(filepath) as f:
            for i, g in enumerate(grouper(n, f, fillvalue=''), 1):
                s_file = self.tmp_dir + '{}{}_{}'.format(basename, ext, i)
                if self.b_run:
                    with open(s_file, 'w') as fout:
                        fout.writelines(g)
                file_list.append(s_file)
        return file_list

    def merge_file(self):
        with open(self.out_path, 'w') as file_out:
            input_lines = fileinput.input(self.output)
            file_out.writelines(input_lines)
        for file in self.tmp_files:
            cmd = "rm -f %s" % (file)
            check_output(cmd, shell=True)

    def get_commands(self):
        self.parse_comand()
        if self.mode == 'list':
            grid_dict = {}
            grid_dict[self.in_label] = [line.rstrip() for line in open(self.input)]
        elif self.mode == 'file':
            path, filename = os.path.split(self.out_path)
            basename, ext = os.path.splitext(filename)
            self.output = []
            for i in range(1, self.nb_threads + 1):
                self.output.append(self.tmp_dir + '{}{}_{}'.format(basename, ext, i))
            file_list = self.split_file(self.input)
            self.tmp_files = file_list + self.output
            grid_dict = {}
            grid_dict[self.in_label] = file_list
        p_list = list(ParameterGrid(grid_dict))
        cmd_added = []
        result_added = []
        all_outs = []
        for p in p_list:
            cmd0 = self.cmd
            for key in p:
                self.grid_params[key] = 1
                cmd0 += ' -' + key + ' ' + p[key]
            cmd_added.append(cmd0)
        for i in range(len(cmd_added)):
            if self.mode == 'list':
                all_outs.append(cmd_added[i] + ' ' + self.out_label + ' ' + self.out_path)
            elif self.mode == 'file':
                all_outs.append(cmd_added[i] + ' ' + self.out_label + ' ' + self.output[i])
        return all_outs

    def parse_comand(self):
        cmd_txts = self._ori_cmd.split(' ')
        tmp_txt = []
        b_skip = False
        for i in range(len(cmd_txts)):
            if b_skip:
                b_skip = False
                continue
            if cmd_txts[i].startswith('-'):
                p_i = cmd_txts[i][1:]
                if p_i == 'i':
                    self.input = cmd_txts[i + 1]
                    b_skip = True
                    continue
                elif p_i == 'o':
                    self.out_path = cmd_txts[i + 1]
                    b_skip = True
                    continue
                else:
                    tmp_txt.append(cmd_txts[i])
            elif cmd_txts[i].startswith('>'):
                self.mode = 'file'
                self.out_path = cmd_txts[i + 1]
                self.out_label = cmd_txts[i]
                b_skip = True
                continue
            else:
                tmp_txt.append(cmd_txts[i])
        self.cmd = ' '.join(tmp_txt)


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='parallel util v0.01.')
    parser.add_argument("-m", dest='run_mode', default="file", help="run_mode: file or list")
    parser.add_argument("-c", dest='command', default="ls",
                        help="annotation input")
    parser.add_argument("-t", dest='threads_num', type=int, default=1, help="threads num")
    parser.add_argument('--r', dest='b_run', action='store_true')
    parser.add_argument('--no-r', dest='b_run', action='store_false')
    parser.set_defaults(b_run=False)
    args = parser.parse_args()
    tmp_dir = './tmp/'
    pdo = parallel_do(mode=args.run_mode, cmd=args.command, nb_threads=args.threads_num, b_run=args.b_run)
    all_commands = pdo.get_commands()
    for cmd in all_commands:
        if not args.b_run:
            print(cmd)
    if args.b_run:
        pdo.local_run(all_commands)
        if pdo.mode == 'file':
            pdo.merge_file()


if __name__ == "__main__":
    main()
