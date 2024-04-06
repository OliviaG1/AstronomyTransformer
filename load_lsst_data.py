# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 09:58:54 2024

@author: Olivia
"""

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler


def load_lsst_data_npy2(data_name1, data_name2=None, data_scale=False, data_norm=False, val_ratio=0.2):
    id_lab_map = { 6: 0, 15: 1, 16: 2, 42: 3, 52: 4, 53: 5, 62: 6, 
                  64: 7, 65: 8, 67: 9, 88: 10, 90: 11, 92: 12, 95: 13}

    if data_name2 == None:
        X = np.load(data_name1 + '_X.npy')
        y = np.load(data_name1 + '_y.npy')
        Z = np.load(data_name1 + '_Z.npy')
    else:        
        X1 = np.load(data_name1 + '_X.npy')
        y1 = np.load(data_name1 + '_y.npy')
        Z1 = np.load(data_name1 + '_Z.npy')
        
        X2 = np.load(data_name2 + '_X.npy')
        y2 = np.load(data_name2 + '_y.npy')
        Z2 = np.load(data_name2 + '_Z.npy')
        
        N1 = X1.shape[0]
        N2 = X2.shape[0]
        X = np.zeros((N1+N2, X1.shape[1], X1.shape[2]))
        X[:N1, :, :] = X1
        X[N1:, :, :] = X2
        
        y = np.vstack((y1, y2))
        Z = np.vstack((Z1, Z2))
    
    X = np.transpose(X, (0, 2, 1))
    
    Z = np.repeat(np.expand_dims(Z, axis=1), 6, axis=1)
    
    #Scaler for each passband
    X = np.sign(X) * np.sqrt(np.sqrt(np.abs(X))) # sqrt twice to reduce range
        
    y = np.squeeze(y)
    y = np.array([id_lab_map[x] for x in y])
    
    if data_norm:
        for i in range(X.shape[0]):
            mean = np.mean(X[i, :, :])
            std = np.std(X[i, :, :])
            X[i, :, :] = (X[i, :, :] - mean) / std
        # seq_len = X.shape[2]
        # mean = np.mean(np.mean(X, axis=2), axis=0)
        # std = np.max(np.std(X, axis=2), axis=0)
        # mean = np.repeat(mean, seq_len).reshape(X.shape[1], seq_len)
        # std = np.repeat(std, seq_len).reshape(X.shape[1], seq_len)
        # X = (X - mean) / std

    X = np.dstack((X, Z))
    
    if data_scale:
        scaler = RobustScaler()
        for i in range(6):
            scaler = scaler.fit(X[:, i, :])
            X[:, i, :] = scaler.transform(X[:, i, :])

    if val_ratio > 0:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=1234)
        train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(y)), y=y))
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
    else:
        X_train, y_train = X, y
        X_val, y_val = [None, None]

    return X_train, y_train, X_val, y_val


def load_lsst_data_npy_half(data_type='all', data_scale=False, data_norm=False, val_ratio=0.2):
    id_lab_map = { 6: 0, 15: 1, 16: 2, 42: 3, 52: 4, 53: 5, 62: 6, 
                  64: 7, 65: 8, 67: 9, 88: 10, 90: 11, 92: 12, 95: 13}

    data_name1 = '../Dataset/LSST/training_set_gp_{}'.format(data_type)
    
    y = np.load(data_name1 + '_y.npy')
    Z = np.load(data_name1 + '_Z.npy')
    for i in range(1, 10):
        data_name2 = '../Dataset/LSST/test_set_batch{}_gp_{}'.format(i, data_type)
        y2 = np.load(data_name2 + '_y.npy')
        Z2 = np.load(data_name2 + '_Z.npy')
        y = np.vstack((y, y2))
        Z = np.vstack((Z, Z2))
    
    X1 = np.load(data_name1 + '_X.npy')
    N = y.shape[0]
    X = np.zeros((N, X1.shape[1], X1.shape[2]))
    N1 = 0
    N2 = X1.shape[0]
    X[N1:N2, :, :] = X1
    for i in range(1, 7):
        data_name2 = '../Dataset/LSST/test_set_batch{}_gp_{}'.format(i, data_type)
        X2 = np.load(data_name2 + '_X.npy')
        N1 = N2
        N2 = N1 + X2.shape[0]
        X[N1:N2, :, :] = X2
        
    X = np.transpose(X, (0, 2, 1))
    
    Z = np.repeat(np.expand_dims(Z, axis=1), 6, axis=1)
    
    #Scaler for each passband
    X = np.sign(X) * np.sqrt(np.sqrt(np.abs(X))) # sqrt twice to reduce range
        
    y = np.squeeze(y)
    y = np.array([id_lab_map[x] for x in y])
    
    if data_norm:
        for i in range(X.shape[0]):
            mean = np.mean(X[i, :, :])
            std = np.std(X[i, :, :])
            X[i, :, :] = (X[i, :, :] - mean) / std
        # seq_len = X.shape[2]
        # mean = np.mean(np.mean(X, axis=2), axis=0)
        # std = np.max(np.std(X, axis=2), axis=0)
        # mean = np.repeat(mean, seq_len).reshape(X.shape[1], seq_len)
        # std = np.repeat(std, seq_len).reshape(X.shape[1], seq_len)
        # X = (X - mean) / std

    X = np.dstack((X, Z))
    
    if data_scale:
        scaler = RobustScaler()
        for i in range(6):
            scaler = scaler.fit(X[:, i, :])
            X[:, i, :] = scaler.transform(X[:, i, :])

    if val_ratio > 0:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=1234)
        train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(y)), y=y))
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_val = X[val_indices]
        y_val = y[val_indices]
    else:
        X_train, y_train = X, y
        X_val, y_val = [None, None]

    return X_train, y_train, X_val, y_val