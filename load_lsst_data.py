# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 10:21:30 2024

@author: Olivia
"""

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import RobustScaler


def load_lsst_data_npy(data_name, data_scale=True, data_norm=False, val_ratio=0.2):
    id_lab_map = { 6: 0, 15: 1, 16: 2, 42: 3, 52: 4, 53: 5, 62: 6, 
                  64: 7, 65: 8, 67: 9, 88: 10, 90: 11, 92: 12, 95: 13}

    X = np.load(data_name + '_X.npy')
    y = np.load(data_name + '_y.npy')
    Z = np.load(data_name + '_Z.npy')
    
    X = np.transpose(X, (0, 2, 1))
    
    Z = np.repeat(np.expand_dims(Z, axis=1), 6, axis=1)
    
    #Scaler for each passband
    if data_scale:
        X = np.sign(X) * np.sqrt(np.sqrt(np.abs(X))) # sqrt twice to reduce range
    
    y = np.squeeze(y)
    y = np.array([id_lab_map[x] for x in y])
    
    if data_norm:
        for i in range(X.shape[0]):
            mean = np.mean(X[i, :, :])
            std = np.std(X[i, :, :])
            X[i, :, :] = (X[i, :, :] - mean) / std

    X = np.dstack((X, Z))
    
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

