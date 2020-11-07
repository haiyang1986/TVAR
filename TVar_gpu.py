import argparse
import time
import sys
import os

os.environ['PYTHONHASHSEED'] = '0'

import random

random.seed(1)

import numpy as np

np.random.seed(1)

import tensorflow as tf

tf.set_random_seed(1)
import tensorlayer as tl

import string
import h5py
import pandas as pd
import pickle
from os.path import splitext, basename, exists, abspath, isfile, getsize
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy import interp
from scipy import stats
from sklearn.utils.class_weight import compute_class_weight
tl.logging.set_verbosity(tl.logging.INFO)

class DLF(object):
    def __init__(self):
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        self.n_epoch = 60
        self.batch_size = 32
        self.learning_rate = 0.0001
        self.print_freq = 5
        self.dim_feature = 0
        self.dim_label = 49
        self.model = None
        self.x = None
        self.y_ = None

    def __del__(self):
        self.sess.close()
        del self.model
        self.model = None

    def initPlaceHolder(self, x_data, y_data=None):
        self.dim_feature = x_data.shape[1]
        if y_data is not None:
            self.dim_label = y_data.shape[1]
        with tf.name_scope("inputs"):
            self.x = tf.placeholder(tf.float32, shape=[None, self.dim_feature], name='x_in')
            self.y_ = tf.placeholder(tf.float32, shape=[None, self.dim_label], name='y_in')
        return

    # define the network
    def train(self, sess, network, train_op, cost, X_train, y_train, x, y_, acc=None, batch_size=100, n_epoch=100, print_freq=5,
            X_val=None, y_val=None, eval_train=True, early_stop= False, save_path =''):
        assert X_train.shape[0] >= batch_size, "Number of training examples should be bigger than the batch size"
        min_loss = float("inf")
        print("Start training the network ...")
        start_time_begin = time.time()
        for epoch in range(n_epoch):
            start_time = time.time()
            loss_ep = 0
            n_step = 0
            for X_train_a, y_train_a in tl.utils.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(network.all_drop)  # enable noise layers
                loss, _ = sess.run([cost, train_op], feed_dict=feed_dict)
                loss_ep += loss
                n_step += 1
            loss_ep = loss_ep / n_step

            if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
                if (X_val is not None) and (y_val is not None):
                    print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
                    if eval_train is True:
                        train_loss, train_acc, n_batch = 0, 0, 0
                        for X_train_a, y_train_a in tl.utils.iterate.minibatches(X_train, y_train, batch_size,
                                                                                 shuffle=True):
                            dp_dict = tl.utils.dict_to_one(network.all_drop)  # disable noise layers
                            feed_dict = {x: X_train_a, y_: y_train_a}
                            feed_dict.update(dp_dict)
                            if acc is not None:
                                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                                train_acc += ac
                            else:
                                err = sess.run(cost, feed_dict=feed_dict)
                            train_loss += err
                            n_batch += 1
                        print("   train loss: %f" % (train_loss / n_batch))
                        if acc is not None:
                            print("   train acc: %f" % (train_acc / n_batch))
                    val_loss, val_acc, n_batch = 0, 0, 0
                    for X_val_a, y_val_a in tl.utils.iterate.minibatches(X_val, y_val, batch_size, shuffle=True):
                        dp_dict = tl.utils.dict_to_one(network.all_drop)  # disable noise layers
                        feed_dict = {x: X_val_a, y_: y_val_a}
                        feed_dict.update(dp_dict)
                        if acc is not None:
                            err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                            val_acc += ac
                        else:
                            err = sess.run(cost, feed_dict=feed_dict)
                        val_loss += err
                        n_batch += 1
                    total_loss = val_loss / n_batch
                    if early_stop:
                        if total_loss < min_loss or epoch + 1 == n_epoch:
                            min_loss = total_loss
                            tl.files.save_npz(network.all_params, name=save_path)
                        else:
                            break
                    print("   val loss: %f" % (total_loss))
                    if acc is not None:
                        print("   val acc: %f" % (val_acc / n_batch))
                else:
                    print(
                        "Epoch %d of %d took %fs, loss %f" % (epoch + 1, n_epoch, time.time() - start_time, loss_ep))
        print("Total training time: %fs" % (time.time() - start_time_begin))


    def score(self, X, x, y_op, batch_size=None):
        if batch_size is None:
            dp_dict = tl.utils.dict_to_one(self.model.all_drop)  # disable noise layers
            feed_dict = {
                x: X,
            }
            feed_dict.update(dp_dict)
            return self.sess.run(y_op, feed_dict=feed_dict)
        else:
            result = []
            for X_a, _ in tl.iterate.minibatches(X, X, batch_size, shuffle=False):
                dp_dict = tl.utils.dict_to_one(self.model.all_drop)
                feed_dict = {
                    x: X_a,
                }
                feed_dict.update(dp_dict)
                result_a = self.sess.run(y_op, feed_dict=feed_dict)
                result.append(result_a)
            if len(result) == 0:
                if len(X) % batch_size != 0:
                    dp_dict = tl.utils.dict_to_one(self.model.all_drop)
                    feed_dict = {
                        x: X[-(len(X) % batch_size):, :],
                    }
                    feed_dict.update(dp_dict)
                    result_a = self.sess.run(y_op, feed_dict=feed_dict)
                    result.append(result_a)
            else:
                if len(X) != len(result) and len(X) % batch_size != 0:
                    dp_dict = tl.utils.dict_to_one(self.model.all_drop)
                    feed_dict = {
                        x: X[-(len(X) % batch_size):, :],
                    }
                    feed_dict.update(dp_dict)
                    result_a = self.sess.run(y_op, feed_dict=feed_dict)
                    result.append(result_a)
            return np.concatenate(result, axis=0)

    def net(self, x, is_train=True, reuse=False):
        with tf.variable_scope("network", reuse=reuse):
            network = tl.layers.InputLayer(x, name='input')
            network = tl.layers.DenseLayer(network, n_units=512, act=tf.nn.relu, name='relu1')
            network = tl.layers.BatchNormLayer(network, name='bn1')
            network = tl.layers.DropoutLayer(network, keep=0.85, name='drop1')
            network = tl.layers.DenseLayer(network, n_units=512, act=tf.nn.relu, name='relu2')
            network = tl.layers.BatchNormLayer(network, name='bn2')
            network = tl.layers.DropoutLayer(network, keep=0.85, name='drop2')
            network = tl.layers.DenseLayer(network, n_units=512, act=tf.nn.relu, name='relu3')
            network = tl.layers.BatchNormLayer(network, name='bn3')
            network = tl.layers.DropoutLayer(network, keep=0.85, name='drop3')
            network = tl.layers.DenseLayer(network, n_units=512, act=tf.nn.relu, name='relu4')
            network = tl.layers.BatchNormLayer(network, name='bn4')
            network = tl.layers.DropoutLayer(network, keep=0.85, name='drop4')
            network = tl.layers.DenseLayer(network, n_units=512, act=tf.nn.relu, name='relu5')
            network = tl.layers.BatchNormLayer(network, name='bn5')
            network = tl.layers.DropoutLayer(network, keep=0.85, name='drop5')
            network = tl.layers.DenseLayer(network, n_units=1024, act=tf.nn.relu, name='relu6')
            network = tl.layers.BatchNormLayer(network, name='bn6')
            network = tl.layers.DropoutLayer(network, keep=0.85, name='drop6')
            network = tl.layers.DenseLayer(network, n_units=2048, act=tf.nn.relu, name='relu7')
            network = tl.layers.BatchNormLayer(network, name='bn7')
            network = tl.layers.DropoutLayer(network, keep=0.85, name='drop7')
            network = tl.layers.DenseLayer(network, n_units=self.dim_label, act=tf.identity, name='output')
        return network

    def save(self, path):
        tl.files.save_npz(self.model.all_params, name=path)

    def load(self, path):
        if self.x is None:
            print('Please set the input data first!')
            return
        if self.model is None:
            self.model = self.net(self.x, is_train=False, reuse=False)
            tl.layers.initialize_global_variables(self.sess)
        load_params = tl.files.load_npz(name=path)
        tl.files.assign_params(self.sess, load_params, self.model)
        print('model loaded!')

    def fit(self, X_train, y_train, b_shuffle=True):
        self.dim_feature = X_train.shape[1]
        self.dim_label = y_train.shape[1]
        self.model = self.net(self.x, is_train=True, reuse=False)
        y_o = self.model.outputs
        cost = tl.cost.sigmoid_cross_entropy(y_o, self.y_, name='xentropy')
        correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(y_o)), tf.round(self.y_))
        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
        tl.layers.initialize_global_variables(self.sess)
        self.model.print_params()
        self.model.print_layers()
        print(self.model.count_params())
        print('learning_rate: %f, batch_size: %d' % (self.learning_rate, self.batch_size))
        tl.utils.fit(self.sess, self.model, train_op, cost, X_train, y_train, self.x, self.y_,
                     acc=acc, batch_size=self.batch_size, n_epoch=self.n_epoch, print_freq=self.print_freq,
                     X_val=None, y_val=None, eval_train=False)

    def predict(self, X):
        y_op = tf.sigmoid(self.model.outputs)
        return self.score(X, self.x, y_op,batch_size=64)


class TVAR(object):
    def __init__(self, nb_threads=1, model_path='./model/', flop_len=1000, cv_len=5):
        self.nb_threads = nb_threads
        self.model_path = model_path
        self.score_path = './score/'
        self.k_mer_len = 6
        self.cv_len = cv_len

        self.hg19 = None
        # bed file for the annotation
        self.df_bed = None
        # hash used to map tabix chr_pos to index
        self.bed_file = None
        # deep learning related parameters
        self.batch_size = 128
        self.epochs = 30
        self.mat_len = flop_len
        self.batch_size = 64
        self.df_GTExs = []

    def score_load(self, filename):
        h5f = h5py.File(filename, 'r')
        y = h5f['data'][:]
        h5f.close()
        return y

    def score_save(self, filename, y):
        h5f = h5py.File(filename, 'w')
        h5f.create_dataset('data', data=y)
        h5f.close()

    def data_load(self, filename):
        h5f = h5py.File(filename, 'r')
        y = h5f['data'][:]
        h5f.close()
        return y

    def data_save(self, filename, y):
        h5f = h5py.File(filename, 'w')
        h5f.create_dataset('data', data=y)
        h5f.close()

    def fit(self, X, y):
        n = X.shape[0]
        # weights = self.calculating_class_weights(y_train)
        scaler = preprocessing.MinMaxScaler()
        X = scaler.fit_transform(X)
        fp = open(self.model_path + '/scaler.model', 'wb')
        pickle.dump(scaler, fp)
        fp.close()
        model = DLF()
        model.initPlaceHolder(X, y)
        save_path = self.model_path + '/model.all.npz'
        model.fit(X, y)
        model.save(save_path)
        del model

    # Cross Validation and shows the performances
    def fit_cv(self, X, y, k, labels=None, file_save='train_out.png', b_plot=False):
        n = X.shape[0]
        if labels is not None:
            k = labels.shape[1]
        for i in range(1, k + 1):
            ix = labels[:, i - 1] == 0
            y_train = y[~ix]
            y_test = y[ix]
            X_train = X[~ix, :]
            X_test = X[ix, :]
            print(X_test.shape)
            # weights = self.calculating_class_weights(y_train)
            scaler = preprocessing.MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            model = DLF()
            model.initPlaceHolder(X_train, y_train)
            model_save = self.model_path + '/weights.cv' + str(i) + '.npz'
            res_save = self.score_path + '/scores.cv' + str(i)
            model.fit(X_train, y_train)
            model.save(model_save)
            model.load(model_save)
            probas_ = model.predict(X_test)
            print(probas_.shape)
            self.score_save(res_save, probas_)
            del model, y_train, X_train, X_test, probas_

    def impute(self, X):
        for i in range(X.shape[1]):
            X[np.isnan(X[:, i]), i] = self.nan_list[i]
        return X

    def predict(self, X):
        model = DLF()
        model.initPlaceHolder(X, None)
        model_save = self.model_path + '/model.all.npz'
        model.load(model_save)
        scaler = pickle.load(open(self.model_path + '/scaler.model', 'rb'))
        X = scaler.transform(X)
        return model.predict(X)

    def score(self, X):
        model = DLF()
        model.initPlaceHolder(X, None)
        model_save = self.model_path + '/model.all.npz'
        model.load(model_save)
        scaler = pickle.load(open(self.model_path + '/scaler.model', 'rb'))
        X = scaler.transform(X)
        return model.predict(X)


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='NVAR v0.1.')
    parser.add_argument("-m", dest='run_mode', default="cv", help="run_mode: train, test")
    parser.add_argument("-i", dest='file_input', default="./input/tvar_labels.gz", help="file input")
    parser.add_argument("-o", dest='output_path', default="./score/", help="file output")
    parser.add_argument('-e', dest='extend_len', type=int, default=1000, help="vcf flop value")
    parser.add_argument("-c", dest='cv', help="Cross Validation k")
    parser.add_argument("-t", dest='threads_num', type=int, default=1, help="threads num")
    parser.add_argument("-v", dest='cv_out', default="./input/tvar_cv.np", help="clinvar_pos")
    parser.add_argument("-n", dest='name_tissue', default="Heart_Left_Ventricle", help="clinvar_pos")
    args = parser.parse_args()
    model_path = './model/'
    tvar = TVAR(args.threads_num, model_path, args.extend_len, args.cv)

    if args.run_mode == 'cv':
        print('Cross Validation started!')
        base_file = splitext(basename(args.file_input))[0]
        fea_file = './fea/' + base_file + '.fea'
        if isfile(args.file_input) and isfile(fea_file):
            print('Training data loading...')
            df_fea = pd.read_hdf(fea_file, 'data')
            df_label = pd.read_csv(args.file_input, sep='\t', compression='gzip')
            print('Training data loaded!')
        else:
            print('run fea model first!')
            return
        X = df_fea.iloc[:, 4:].as_matrix()
        y = df_label.iloc[:, 4:].as_matrix().astype(int)
        cv_labels = tvar.data_load(args.cv_out)
        del df_fea, df_label
        print(X.shape, y.shape)
        tvar.fit_cv(X, y, 5, cv_labels)
        print('CV finished!')

    # supervised testing
    elif args.run_mode == 'train':
        print('Train started!')
        base_file = splitext(basename(args.file_input))[0]
        fea_file = './fea/' + base_file + '.fea'
        if isfile(fea_file):
            print('Training data loading...')
            df_fea = pd.read_hdf(fea_file, 'data')
            df_label = pd.read_csv(args.file_input, sep='\t', compression='gzip')
            print('Training data loaded!')
        else:
            print('run fea model first!')
            return
        X = df_fea.iloc[:, 4:].as_matrix()
        y = df_label.iloc[:, 4:].as_matrix().astype(int)
        #shuffle
        b_shuffle = False
        if b_shuffle:
            y_shape = y.shape
            y = y.reshape((y_shape))
            np.random.shuffle(y)
            y = y.reshape(y_shape)
        del df_fea
        print(X.shape, y.shape)
        tvar.fit(X, y)
        print('Train finished!')


    elif args.run_mode == 'score':
        label_file = './input/tvar_labels.gz'
        base_file = splitext(basename(args.file_input))[0]
        fea_file = './fea/' + base_file + '.fea'
        score_file = './score/' + base_file + '.tvar'
        if isfile(fea_file):
            df_fea = pd.read_hdf(fea_file, 'data')
        else:
            print('run fea model first!')
            return
        df_label = pd.read_csv(label_file, sep='\t', compression='gzip')
        label_list = list(df_label)[4:]
        X = df_fea.iloc[:, 4:].as_matrix()
        scores = tvar.score(X)
        #X_one = df_fea.iloc[:, 0:2]
        #X_one['tvar'] = np.mean(scores, axis=1)
        #X_one.to_csv(score_file, index=False, header=False, sep='\t')
        X_one = df_fea.iloc[:, 0:2]
        X_score =pd.DataFrame(data=scores, dtype=np.float, columns=label_list)
        X_all = pd.concat([X_one, X_score], axis=1)
        print(X_all.shape, scores.shape)
        #print (X_all.shape)
        del df_fea
        X_all.to_csv(score_file, index=False, header=True, sep='\t')
        print('Score finished!')

    elif args.run_mode == 'gwas':
        tissue = args.name_tissue
        label_file = './input/tvar_labels.gz'
        base_file = splitext(basename(args.file_input))[0]
        fea_file = './fea/' + base_file + '.fea'
        score_file = './score/' + base_file + '.tvar'
        if isfile(fea_file):
            df_fea = pd.read_hdf(fea_file, 'data')
        else:
            print('run fea model first!')
            return
        df_label = pd.read_csv(label_file, sep='\t', compression='gzip')
        label_list = list(df_label)[4:]
        X = df_fea.iloc[:, 4:].as_matrix()
        scores = tvar.score(X)
        #X_one = df_fea.iloc[:, 0:2]
        #X_one['tvar'] = np.mean(scores, axis=1)
        #X_one.to_csv(score_file, index=False, header=False, sep='\t')
        X_one = df_fea.iloc[:, 0:2]
        # if tissue =='Brain_Caudate_basal_ganglia':
        #     tissue ='Brain_Anterior_cingulate_cortex_BA24'
        X_score =pd.DataFrame(data=scores, dtype=np.float, columns=label_list).loc[:,tissue]
        X_all = pd.concat([X_one, X_score], axis=1)
        print(X_all.shape, scores.shape)
        #print (X_all.shape)
        del df_fea
        X_all.to_csv(score_file, index=False, header=False, sep='\t')
        print('gwas finished!')

    elif args.run_mode == 'rare':
        tissue = args.name_tissue
        label_file = './input/tvar_labels.gz'
        base_file = splitext(basename(args.file_input))[0]
        fea_file = './fea/' + base_file + '.fea'
        score_file = './score/' + base_file + '.tvar'
        if isfile(fea_file):
            df_fea = pd.read_hdf(fea_file, 'data')
        else:
            print('run fea model first!')
            return
        df_label = pd.read_csv(label_file, sep='\t', compression='gzip')
        label_list = list(df_label)[4:]
        print(label_list)
        X = df_fea.iloc[:, 4:].as_matrix()
        scores = tvar.score(X)
        #X_one = df_fea.iloc[:, 0:2]
        #X_one['tvar'] = np.mean(scores, axis=1)
        #X_one.to_csv(score_file, index=False, header=False, sep='\t')
        X_one = df_fea.iloc[:, 0:2]
        # if tissue =='Brain_Caudate_basal_ganglia':
        #     tissue ='Brain_Anterior_cingulate_cortex_BA24'
        X_score =pd.DataFrame(data=scores, dtype=np.float, columns=label_list).loc[:,tissue]
        X_all = pd.concat([X_one, X_score], axis=1)
        print(X_all.shape, scores.shape)
        #print (X_all.shape)
        del df_fea
        X_all.to_csv(score_file, index=False, header=False, sep='\t')
        print('Rare finished!')


if __name__ == "__main__":
    main()
