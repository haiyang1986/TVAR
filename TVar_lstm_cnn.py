import argparse
import os
import pickle
import random
import sys
from os.path import splitext, basename, isfile

import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam


def set_global_determinism(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def CNN_model(x_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same',
                     input_shape=(x_shape[1], x_shape[2], x_shape[3])))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(rate=0.2))
    model.add(Dense(49, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.summary()
    return model


def LSTM_model(x_shape):
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, input_shape=(x_shape[1], x_shape[2])))
    model.add(LSTM(32, return_sequences=True))
    model.add(LSTM(32))
    model.add(Flatten())
    model.add(Dropout(rate=0.2))
    model.add(Dense(49, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    model.summary()
    return model



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
        self.batch_size = 32
        self.epochs = 60
        self.mat_len = flop_len
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

    def fit(self, X, y, method='LSTM'):
        n = X.shape[0]
        X = np.concatenate((X, np.zeros((X.shape[0], 13 * 13 - X.shape[1]))), axis=1)
        scaler = preprocessing.MinMaxScaler()
        X = scaler.fit_transform(X)
        fp = open(self.model_path + '/scaler_169.model', 'wb')
        pickle.dump(scaler, fp)
        fp.close()
        if method == 'CNN':
            X = X.reshape(X.shape[0], 13, 13, 1)
            model = CNN_model(X.shape)
            save_path = self.model_path + '/model.CNN.h5'
        elif method == 'LSTM':
            X = X.reshape(X.shape[0], 13, 13)
            model = LSTM_model(X.shape)
            save_path = self.model_path + '/model.LSTM.h5'
        model.fit(X, y, batch_size=self.batch_size, epochs=self.epochs, verbose=2)
        model.save(save_path)
        del model

    # Cross Validation and shows the performances
    def fit_cv(self, X, y, k, labels=None, method='LSTM'):
        X = np.concatenate((X, np.zeros((X.shape[0], 13 * 13 - X.shape[1]))), axis=1)
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
            if method == 'CNN':
                X_train = X_train.reshape(X_train.shape[0], 13, 13, 1)
                X_test = X_test.reshape(X_test.shape[0], 13, 13, 1)
                model = CNN_model(X_train.shape)
            elif method == 'LSTM':
                X_train = X_train.reshape(X_train.shape[0], 13, 13)
                X_test = X_test.reshape(X_test.shape[0], 13, 13)
                model = LSTM_model(X_train.shape)
            model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=2,
                      validation_data=(X_test, y_test))
            model_save = self.model_path + '/weights.cv' + str(i) + '.h5'
            res_save = self.score_path + '/scores.cv' + str(i)
            model.save(model_save)
            # model = load_model(model_save)
            probas_ = model.predict(X_test, batch_size=self.batch_size, verbose=0)
            print(probas_.shape)
            self.score_save(res_save, probas_)
            del model, y_train, X_train, X_test, probas_

    def impute(self, X):
        for i in range(X.shape[1]):
            X[np.isnan(X[:, i]), i] = self.nan_list[i]
        return X

    def predict(self, X, method ='LSTM'):
        save_path = ''
        X = np.concatenate((X, np.zeros((X.shape[0], 13 * 13 - X.shape[1]))), axis=1)
        scaler = pickle.load(open(self.model_path + '/scaler_169.model', 'rb'))
        X = scaler.transform(X)
        if method == 'CNN':
            X = X.reshape(X.shape[0], 13, 13, 1)
            save_path = self.model_path + '/model.CNN.h5'
        elif method == 'LSTM':
            X = X.reshape(X.shape[0], 13, 13)
            save_path = self.model_path + '/model.LSTM.h5'
        model = load_model(save_path)
        return model.predict(X, verbose=0)


    def score(self, X, method ='LSTM'):
        save_path = ''
        X = np.concatenate((X, np.zeros((X.shape[0], 13 * 13 - X.shape[1]))), axis=1)
        scaler = pickle.load(open(self.model_path + '/scaler_169.model', 'rb'))
        X = scaler.transform(X)
        if method == 'CNN':
            X = X.reshape(X.shape[0], 13, 13, 1)
            save_path = self.model_path + '/model.CNN.h5'
        elif method == 'LSTM':
            X = X.reshape(X.shape[0], 13, 13)
            save_path = self.model_path + '/model.LSTM.h5'
        model = load_model(save_path)
        return model.predict(X, verbose=0)


def main(argv=sys.argv):
    parser = argparse.ArgumentParser(description='TVAR v1.0')
    parser.add_argument("-m", dest='run_mode', default="cv", help="run_mode: train, test")
    parser.add_argument("-i", dest='file_input', default="./input/tvar_labels.gz", help="file input")
    parser.add_argument("-o", dest='output_path', default="./score/", help="file output")
    parser.add_argument('-e', dest='extend_len', type=int, default=1000, help="vcf flop value")
    parser.add_argument("-c", dest='cv', help="Cross Validation k")
    parser.add_argument("-t", dest='threads_num', type=int, default=1, help="threads num")
    parser.add_argument("-v", dest='cv_out', default="./input/tvar_cv.np", help="clinvar_pos")
    parser.add_argument("-n", dest='name_tissue', default="Heart_Left_Ventricle", help="clinvar_pos")
    parser.add_argument("-d", dest='deep_model', default="LSTM", help="clinvar_pos")
    args = parser.parse_args()
    set_global_determinism()
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
        X = df_fea.iloc[:, 4:].values
        y = df_label.iloc[:, 4:].values.astype(int)
        cv_labels = tvar.data_load(args.cv_out)
        del df_fea, df_label
        print(X.shape, y.shape)
        tvar.fit_cv(X, y, 5, cv_labels, method= args.deep_model)
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
        X = df_fea.iloc[:, 4:].values
        y = df_label.iloc[:, 4:].values.astype(int)
        # shuffle
        b_shuffle = False
        if b_shuffle:
            y_shape = y.shape
            y = y.reshape((y_shape))
            np.random.shuffle(y)
            y = y.reshape(y_shape)
        del df_fea
        print(X.shape, y.shape)
        tvar.fit(X, y, method= args.deep_model)
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
        X = df_fea.iloc[:, 4:].values
        scores = tvar.score(X, method= args.deep_model)
        # X_one = df_fea.iloc[:, 0:2]
        # X_one['tvar'] = np.mean(scores, axis=1)
        # X_one.to_csv(score_file, index=False, header=False, sep='\t')
        X_one = df_fea.iloc[:, 0:2]
        X_score = pd.DataFrame(data=scores, dtype=np.float, columns=label_list)
        X_all = pd.concat([X_one, X_score], axis=1)
        print(X_all.shape, scores.shape)
        # print (X_all.shape)
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
        X = df_fea.iloc[:, 4:].values
        scores = tvar.score(X, method= args.deep_model)
        # X_one = df_fea.iloc[:, 0:2]
        # X_one['tvar'] = np.mean(scores, axis=1)
        # X_one.to_csv(score_file, index=False, header=False, sep='\t')
        X_one = df_fea.iloc[:, 0:2]
        # if tissue =='Brain_Caudate_basal_ganglia':
        #     tissue ='Brain_Anterior_cingulate_cortex_BA24'
        X_score = pd.DataFrame(data=scores, dtype=np.float, columns=label_list).loc[:, tissue]
        X_all = pd.concat([X_one, X_score], axis=1)
        print(X_all.shape, scores.shape)
        # print (X_all.shape)
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
        X = df_fea.iloc[:, 4:].values
        scores = tvar.score(X, method= args.deep_model)
        # X_one = df_fea.iloc[:, 0:2]
        # X_one['tvar'] = np.mean(scores, axis=1)
        # X_one.to_csv(score_file, index=False, header=False, sep='\t')
        X_one = df_fea.iloc[:, 0:2]
        # if tissue =='Brain_Caudate_basal_ganglia':
        #     tissue ='Brain_Anterior_cingulate_cortex_BA24'
        X_score = pd.DataFrame(data=scores, dtype=np.float, columns=label_list).loc[:, tissue]
        X_all = pd.concat([X_one, X_score], axis=1)
        print(X_all.shape, scores.shape)
        # print (X_all.shape)
        del df_fea
        X_all.to_csv(score_file, index=False, header=False, sep='\t')
        print('Rare finished!')


if __name__ == "__main__":
    main()
