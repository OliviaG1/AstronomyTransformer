# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 14:30:31 2024

@author: Olivia
"""

import numpy as np
import pandas as pd


def create_dataset(X, y, time_steps=100, step=100):
    Xs, ys = [], []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i : (i + time_steps)].values
        labels = y.iloc[i : i + time_steps].values
        Xs.append(v)
        if labels[0] != labels[-1]:
            print('Warning: label mismatch at {}'.format(i))
            continue
        ys.append(labels[0])
    return np.array(Xs), np.array(ys).reshape(-1, 1)



# f_input_csv = 'training_set'
# f_meta = 'training_set_metadata.csv'
# data_dir = 'D:/Kaggle/PLAsTiCC/data/'
# detected_only = True

# if detected_only:
#     f_output_csv = data_dir + f_input_csv + '_gp_detected_only.csv'
# else:
#     f_output_csv = data_dir + f_input_csv + '_gp_all.csv'

# df = pd.read_csv(f_output_csv)

# timesteps = 100
# step = 100


# # Save X, y. No OneHot
# scale_columns = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]
# Xs, ys = create_dataset(df[scale_columns], df.target, timesteps, step)
# np.save(f_output_csv[:-4]+'_X.npy', Xs)
# np.save(f_output_csv[:-4]+'_y.npy', ys)


# # Save Z
# zcols = ["hostgal_photoz", "hostgal_photoz_err"]
# ZXs, zys = create_dataset(df[zcols], df.target, timesteps, step)
# ZX = []
# for z in range(0, len(ZXs)):
#     ZX.append(ZXs[z][0])
# ZX = np.array(ZX)
# np.save(f_output_csv[:-4]+'_Z.npy', ZX)






f_meta = 'unblinded_test_set_metadata.csv'

for f_input_csv in ['test_set_batch7', 'test_set_batch10', 'test_set_batch11']:
    data_dir = 'D:/Kaggle/PLAsTiCC/data/'
    detected_only = True
    
    if detected_only:
        f_output_csv = data_dir + f_input_csv + '_gp_detected_only.csv'
    else:
        f_output_csv = data_dir + f_input_csv + '_gp_all.csv'
    
    df = pd.read_csv(f_output_csv)
    
    timesteps = 100
    step = 100
    
    
    # Save X, y. No OneHot
    scale_columns = ["lsstg", "lssti", "lsstr", "lsstu", "lssty", "lsstz"]
    Xs, ys = create_dataset(df[scale_columns], df.target, timesteps, step)
    np.save(f_output_csv[:-4]+'_X.npy', Xs)
    np.save(f_output_csv[:-4]+'_y.npy', ys)
    
    
    # Save Z
    zcols = ["hostgal_photoz", "hostgal_photoz_err"]
    ZXs, zys = create_dataset(df[zcols], df.target, timesteps, step)
    ZX = []
    for z in range(0, len(ZXs)):
        ZX.append(ZXs[z][0])
    ZX = np.array(ZX)
    np.save(f_output_csv[:-4]+'_Z.npy', ZX)
