# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 10:21:30 2024

@author: Olivia
"""

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from torchsummary import summary

from load_lsst_data import load_lsst_data_npy2

from copy import deepcopy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

from AstronomyTransformer import AstronomyTransformer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class dataset_class(Dataset):

    def __init__(self, data, label):
        super(dataset_class, self).__init__()

        self.feature = data
        self.labels = label.astype(np.int32)

    def __getitem__(self, ind):
        x = self.feature[ind]
        x = x.astype(np.float32)
        y = self.labels[ind]  # (num_labels,) array
        data = torch.tensor(x)
        label = torch.tensor(y)
        return data, label, ind

    def __len__(self):
        return len(self.labels)




def load_model(model, model_path, optimizer=None, resume=False, change_output=False,
               lr=None, lr_step=None, lr_factor=None):
    start_epoch = 0
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = deepcopy(checkpoint['state_dict'])
    if change_output:
        for key, val in checkpoint['state_dict'].items():
            if key.startswith('output_layer'):
                state_dict.pop(key)
    model.load_state_dict(state_dict, strict=False)
    print('Loaded model from {}. Epoch: {}'.format(model_path, checkpoint['epoch']))

    # resume optimizer parameters
    if optimizer is not None and resume:
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
            start_lr = lr
            for i in range(len(lr_step)):
                if start_epoch >= lr_step[i]:
                    start_lr *= lr_factor[i]
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr', start_lr)
        else:
            print('No optimizer parameters in checkpoint.')
    if optimizer is not None:
        return model, optimizer, start_epoch
    else:
        return model





# Load Data
data_type='detected'

data_name_train1 = '../Dataset/LSST/training_set_gp_{}'.format(data_type)
data_name_train2 = None

X_train, y_train, X_val, y_val = load_lsst_data_npy2(
    data_name1=data_name_train1, data_name2=data_name_train2, data_scale=False, data_norm=False, val_ratio=0.2)

input_shape = (X_train.shape[1], X_train.shape[2])


# Load Best Model
# Creat Transformer Model
model_id = 'best_{}'.format(data_type)

best_model = AstronomyTransformer(input_shape=input_shape, embedding_size=32, heads_num=8, 
                fnn_num=256, num_classes=14, dropout=0.3)

best_model_name = 'Results/model_{}.pth'.format(model_id)
best_model.load_state_dict(torch.load(best_model_name))
best_model.to(device)
best_model.eval()

summary(best_model, input_shape)


# Run Test
batch_size = 64

torch.no_grad()

# Evalue Testing Data - All Batches
per_batch = {'targets': [], 'predictions': [], 'IDs': []}
for batch_id in range(7, 12):
    data_name_test1 = '../Dataset/LSST/test_set_batch{}_gp_{}'.format(batch_id, data_type)
    data_name_test2 = None
    
    X_test, y_test, _, _ = load_lsst_data_npy2(
        data_name1=data_name_test1, data_name2=data_name_test2, data_scale=False, data_norm=False, val_ratio=0)
    test_dataset = dataset_class(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    for i, batch in enumerate(test_loader):
        X, targets, IDs = batch
        predictions = best_model(X.to(device))
        
        targets = targets.type(torch.LongTensor)
        targets = targets.to(device)
        loss = F.cross_entropy(predictions, targets)
    
        per_batch['targets'].append(targets.cpu().numpy())
        predictions = predictions.detach()
        per_batch['predictions'].append(predictions.cpu().numpy())
        per_batch['IDs'].append(IDs)

    print('Batch {} done.'.format(batch_id))

predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
probs = torch.nn.functional.softmax(predictions, dim=1)  # (total_samples, num_classes) est. prob. for each class and sample
predictions = torch.argmax(probs, dim=1).cpu().numpy()  # (total_samples,) int class index for each sample
probs = probs.cpu().numpy()
targets = np.concatenate(per_batch['targets'], axis=0).flatten()
ConfMatrix_Test = confusion_matrix(targets, predictions)
print('Testing Data Done.')





'''
https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
'''
def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    #print(cm)
    plt.figure(figsize=[20, 18], dpi=90)
    if normalize:
        plt.imshow(cm, interpolation='nearest', vmin=0.0, vmax=1.0, cmap=cmap)
    else:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=24)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=16)
    plt.yticks(tick_marks, classes, fontsize=16)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize=14,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=20)
    plt.xlabel('Predicted label', fontsize=20)
    plt.tight_layout();
    

# plot confusion matrix with classes name
target_map = { 6: 'u-Lensing', 
              15: 'TDE', 
              16: 'EBE', 
              42: 'SN II', 
              52: 'SN Iax', 
              53: 'Mira variable', 
              62: 'SN Ibc', 
              64: 'Kilonova', 
              65: 'M dwarf', 
              67: 'SN Ia-91bg', 
              88: 'AGN', 
              90: 'SN Ia', 
              92: 'RR Lyrae', 
              95: 'SLSN'}
classes = list(target_map.values())


title = 'Confusion matrix: {}-{} for Test Batch 7~11'.format('Transformer', model_id)
plot_confusion_matrix(ConfMatrix_Test, classes, normalize=False, title=title)
plot_confusion_matrix(ConfMatrix_Test, classes, normalize=True, title=title)


# Reorder classes
classes_new = ['SN Ia', 
               'SN Iax', 
               'SN Ia-91bg', 
               'SN Ibc', 
               'SN II', 
               'M dwarf', 
               'EBE', 
               'Kilonova', 
               'RR Lyrae', 
               'u-Lensing', 
               'TDE', 
               'Mira variable', 
               'AGN', 
               'SLSN']

newindex = [classes.index(x) for x in classes_new]
oldindex = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

ConfMatrix_Test_new = ConfMatrix_Test.copy()
ConfMatrix_Test_new[oldindex] = ConfMatrix_Test_new[newindex]
ConfMatrix_Test_new[:, oldindex] = ConfMatrix_Test_new[:, newindex]

# Final plots for poster
title = 'Confusion matrix: Test'
plot_confusion_matrix(ConfMatrix_Test_new, classes_new, normalize=True, title=title)


# Combine SN Ia, Iax, Ia-91bg, Ibc
classes_new2 = ['SN I', 
                'SN II', 
                'M dwarf', 
                'EBE', 
                'Kilonova', 
                'RR Lyrae', 
                'u-Lensing', 
                'TDE', 
                'Mira variable', 
                'AGN', 
                'SLSN']

ConfMatrix_Test_new2 = np.zeros((ConfMatrix_Test_new.shape[0]-3, ConfMatrix_Test_new.shape[1]-3), dtype='int')
ConfMatrix_Test_new2[0, 0] = np.sum(ConfMatrix_Test_new[0:4, 0:4])
ConfMatrix_Test_new2[1:, 0] = np.sum(ConfMatrix_Test_new[4:, 0:4], axis=1)
ConfMatrix_Test_new2[0, 1:] = np.sum(ConfMatrix_Test_new[0:4, 4:], axis=0)
ConfMatrix_Test_new2[1:, 1:] = ConfMatrix_Test_new[4:, 4:]

# Final plots for poster
# All with SN I and II
title = 'Confusion matrix: Test'
plot_confusion_matrix(ConfMatrix_Test_new2, classes_new2, normalize=True, title=title)

# Just SN
title = 'Confusion matrix: SuperNova (SN) Only'
plot_confusion_matrix(ConfMatrix_Test_new[0:5, 0:5], classes_new[0:5], normalize=True, title=title)

