# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 10:21:30 2024

@author: Olivia
"""

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

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
data_name_train1 = 'Dataset/LSST/training_set_gp_all'
data_name_train2 = 'Dataset/LSST/test_set_batch1_gp_all'
b1 = 2
b2 = 3
data_name_test1 = 'Dataset/LSST/test_set_batch{}_gp_all'.format(b1)
data_name_test2 = 'Dataset/LSST/test_set_batch{}_gp_all'.format(b2)

X_train, y_train, X_val, y_val = load_lsst_data_npy2(
    data_name1=data_name_train1, data_name2=data_name_train2, data_scale=False, data_norm=False, val_ratio=0.2)
train_dataset = dataset_class(X_train, y_train)
val_dataset = dataset_class(X_val, y_val)

X_test, y_test, _, _ = load_lsst_data_npy2(
    data_name1=data_name_test1, data_name2=data_name_test2, data_scale=False, data_norm=False, val_ratio=0)
test_dataset = dataset_class(X_test, y_test)

input_shape = (X_train.shape[1], X_train.shape[2])



# Load Best Model
# Creat Transformer Model
best_model = AstronomyTransformer(input_shape=input_shape, embedding_size=32, heads_num=8, 
                fnn_num=256, num_classes=14, dropout=0.3)

best_model_name = 'Results/AstronomyTransformer/model_best.pth'
best_model.load_state_dict(torch.load(best_model_name))
best_model.to(device)
best_model.eval()



# Run Test
batch_size = 64

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

torch.no_grad()

# Evalue Training Data
per_batch = {'targets': [], 'predictions': [], 'IDs': []}
for i, batch in enumerate(train_loader):
    X, targets, IDs = batch
    predictions = best_model(X.to(device))
    
    targets = targets.type(torch.LongTensor)
    targets = targets.to(device)
    loss = F.cross_entropy(predictions, targets)

    per_batch['targets'].append(targets.cpu().numpy())
    predictions = predictions.detach()
    per_batch['predictions'].append(predictions.cpu().numpy())
    per_batch['IDs'].append(IDs)


predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
probs = torch.nn.functional.softmax(predictions, dim=1)
predictions = torch.argmax(probs, dim=1).cpu().numpy()
probs = probs.cpu().numpy()
targets = np.concatenate(per_batch['targets'], axis=0).flatten()
ConfMatrix_Train = confusion_matrix(targets, predictions)
print('Training Data Done.')

# Evalue Val Data
per_batch = {'targets': [], 'predictions': [], 'IDs': []}
for i, batch in enumerate(val_loader):
    X, targets, IDs = batch
    predictions = best_model(X.to(device))
    
    targets = targets.type(torch.LongTensor)
    targets = targets.to(device)
    loss = F.cross_entropy(predictions, targets)

    per_batch['targets'].append(targets.cpu().numpy())
    predictions = predictions.detach()
    per_batch['predictions'].append(predictions.cpu().numpy())
    per_batch['IDs'].append(IDs)


predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
probs = torch.nn.functional.softmax(predictions, dim=1)
predictions = torch.argmax(probs, dim=1).cpu().numpy()
probs = probs.cpu().numpy()
targets = np.concatenate(per_batch['targets'], axis=0).flatten()
ConfMatrix_Val = confusion_matrix(targets, predictions)
print('Eval Data Done.')


# Evalue Testing Data
per_batch = {'targets': [], 'predictions': [], 'IDs': []}
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


predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
probs = torch.nn.functional.softmax(predictions, dim=1)
predictions = torch.argmax(probs, dim=1).cpu().numpy()
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
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    plt.figure(figsize=[16, 14], dpi=90)
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
target_map = { 6: '6: U-lensing', 
              15: '15: TDE', 
              16: '16: EBE', 
              42: '42: SN II', 
              52: '52: SN Iax', 
              53: '53: Mira variable', 
              62: '62: SN Ibc', 
              64: '64: Kilonova', 
              65: '65: M dwarf', 
              67: '67: SN Ia-91bg', 
              88: '88: AGN', 
              90: '90: SN Ia', 
              92: '92: RR Lyrae', 
              95: '95: SLSN'}
classes = list(target_map.values())


plot_confusion_matrix(ConfMatrix_Train, classes, normalize=False, title='Confusion matrix - Train')
plot_confusion_matrix(ConfMatrix_Train, classes, normalize=True, title='Confusion matrix - Train')

plot_confusion_matrix(ConfMatrix_Val, classes, normalize=False, title='Confusion matrix - Val')
plot_confusion_matrix(ConfMatrix_Val, classes, normalize=True, title='Confusion matrix - Val')

plot_confusion_matrix(ConfMatrix_Test, classes, normalize=False, title='Confusion matrix - Test')
plot_confusion_matrix(ConfMatrix_Test, classes, normalize=True, title='Confusion matrix - Test')


