# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 10:21:30 2024

@author: Olivia
"""

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from load_lsst_data import load_lsst_data_npy2

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




# Load Training Data
data_name1 = 'Dataset/LSST/training_set_gp_all'
data_name2 = 'Dataset/LSST/test_set_batch1_gp_all'
X_train, y_train, X_val, y_val = load_lsst_data_npy2(data_name1, data_name2, data_scale=False, data_norm=False, val_ratio=0.2)


# balance training data
rate_factor = 1.0 / 28
X_train_new = X_train.copy()
y_train_new = y_train.copy()
NN = int(len(y_train) * rate_factor)
for itype in range(int(max(y_train)) + 1):
    xtmp = X_train[y_train==itype, :, :]
    if xtmp.shape[0] < NN:
        xxs = [xtmp]
        for _ in range(NN//xtmp.shape[0]):
            xx = xtmp.copy()
            d = int(np.random.uniform(low=0, high=10) - 5)
            if d < 0:
                xx[:, :, 0:d] = xx[:, :, -d:]
            elif d > 0:
                xx[:, :, d:] = xx[:, :, 0:-d]
            noise1 = np.random.randn(xx.shape[0], xx.shape[1], xx.shape[2]-2) * (0.0001 * np.max(np.abs(xx)))
            xx[:, :, 0:100] = xx[:, :, 0:100] + noise1
            noise2 = np.random.uniform(low=0, high=1, size=(xx.shape[0], xx.shape[1], 2))
            xx[:, :, 100:] = xx[:, :, 100:] + noise2
            xxs.append(xx)
        
        xx = np.vstack(xxs)
        N2 = NN - xtmp.shape[0]
        X_train_new = np.vstack((X_train_new, xx[0:N2, :, :]))
        y_train_new = np.hstack((y_train_new, np.ones(N2)*itype))

for itype in range(int(max(y_train_new)) + 1):
    print("number of type {} in training = {}".format(itype, np.sum(y_train_new == itype)))


X_val_new = X_val.copy()
y_val_new = y_val.copy()
NN = int(len(y_val) * rate_factor)
for itype in range(int(max(y_val)) + 1):
    xtmp = X_val[y_val==itype, :, :]
    if xtmp.shape[0] < NN:
        xxs = [xtmp]
        for _ in range(NN//xtmp.shape[0]):
            xx = xtmp.copy()
            d = int(np.random.uniform(low=0, high=10) - 5)
            if d < 0:
                xx[:, :, 0:d] = xx[:, :, -d:]
            elif d > 0:
                xx[:, :, d:] = xx[:, :, 0:-d]
            noise1 = np.random.randn(xx.shape[0], xx.shape[1], xx.shape[2]-2) * (0.0001 * np.max(np.abs(xx)))
            xx[:, :, 0:100] = xx[:, :, 0:100] + noise1
            noise2 = np.random.uniform(low=0, high=1, size=(xx.shape[0], xx.shape[1], 2))
            xx[:, :, 100:] = xx[:, :, 100:] + noise2
            xxs.append(xx)
        
        xx = np.vstack(xxs)
        N2 = NN - xtmp.shape[0]
        X_val_new = np.vstack((X_val_new, xx[0:N2, :, :]))
        y_val_new = np.hstack((y_val_new, np.ones(N2)*itype))

for itype in range(int(max(y_val_new)) + 1):
    print("number of type {} in val = {}".format(itype, np.sum(y_val_new == itype)))


train_dataset = dataset_class(X_train_new, y_train_new)
val_dataset = dataset_class(X_val_new, y_val_new)

input_shape = (X_train_new.shape[1], X_train_new.shape[2])


# Creat Transformer Model
model = AstronomyTransformer(input_shape=input_shape, embedding_size=32, heads_num=8, 
                fnn_num=256, num_classes=14, dropout=0.3)

model.to(device)


# Training
epochs = 50
batch_size = 32
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)
best_epoch_loss = 100000

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

losses = []
epoch_loss_pre = 100000
epoch_loss_cnt = 0
for epoch in range(epochs):
    # Training
    epoch_loss = 0
    total_samples = 0
    model.train()
    for i, batch in enumerate(train_loader):
        X, targets, IDs = batch
        predictions = model(X.to(device))

        targets = targets.type(torch.LongTensor)
        targets = targets.to(device)
        loss = F.cross_entropy(predictions, targets)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
        optimizer.step()

        with torch.no_grad():
            total_samples += len(targets)
            epoch_loss += (len(targets) * loss)

    epoch_loss_train = epoch_loss / total_samples

    # Evalue
    epoch_loss = 0
    total_samples = 0
    model.eval()
    per_batch = {'targets': [], 'predictions': [], 'metrics': [], 'IDs': []}
    for i, batch in enumerate(val_loader):
        X, targets, IDs = batch
        predictions = model(X.to(device))
        
        targets = targets.type(torch.LongTensor)
        targets = targets.to(device)
        loss = F.cross_entropy(predictions, targets)

        per_batch['targets'].append(targets.cpu().numpy())
        predictions = predictions.detach()
        per_batch['predictions'].append(predictions.cpu().numpy())
        loss = loss.detach()
        per_batch['metrics'].append([loss.cpu().numpy()])
        per_batch['IDs'].append(IDs)

        metrics = {"loss": loss}
        total_samples += len(targets)
        epoch_loss += (len(targets) * loss)

    epoch_loss_val = epoch_loss / total_samples

    predictions = torch.from_numpy(np.concatenate(per_batch['predictions'], axis=0))
    probs = torch.nn.functional.softmax(predictions, dim=1)
    predictions = torch.argmax(probs, dim=1).cpu().numpy()
    probs = probs.cpu().numpy()
    targets = np.concatenate(per_batch['targets'], axis=0).flatten()
    
    str_print = 'epoch ={:3d}, loss_train ={:8.5f}, loss_val ={:8.5f}.'.format(epoch, epoch_loss_train, epoch_loss_val)
    losses.append([epoch_loss_train.cpu().detach().numpy(), epoch_loss_val.cpu().detach().numpy()])

    # save model
    if epoch_loss_val < best_epoch_loss:
        best_epoch_loss = epoch_loss_val
        model_fname = 'Results/AstronomyTransformer/model_epoch_{}.pth'.format(epoch)
        torch.save(model.state_dict(), model_fname)
        torch.save(model.state_dict(), 'Results/AstronomyTransformer/model_best.pth')
        str_print += ' Save model.'
    
    print(str_print)
    
    # early stop
    if(epoch_loss_val <= epoch_loss_pre):
        epoch_loss_cnt = 0
    else:
        epoch_loss_cnt += 1
        if(epoch_loss_cnt > 10):
            print('Early Stop.')
            break
    
    epoch_loss_pre = epoch_loss_val


# plot loss
losses = np.array(losses)
plt.figure()
plt.plot(losses[:, 0], 'b', label='Training Loss')
plt.plot(losses[:, 1], 'r', label='Val Loss')
plt.legend(loc=0)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Val Loss')