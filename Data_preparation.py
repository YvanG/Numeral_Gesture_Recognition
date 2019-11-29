#!/usr/bin/env python

__author__ = "Ivan Gruber"
__version__ = "1.0.0"
__maintainer__ = "Ivan Gruber"
__email__ = "ivan.gruber@seznam.cz"

"""
Data shuffling
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import cv2
import h5py

#data loading
h5_num = 1
data_path = '/storage/plzen1/home/grubiv/AMIR/SPECOM2018/annotations/'+str(h5_num)+'/'
dirs = os.listdir(data_path)
dirs.sort()

with h5py.File(data_path+dirs[0],'r') as fr:
    print('Byl nacten soubor: '+dirs[0])
    X = fr['imgs'][:]
    y = fr['label'][:]
    print(X.shape)
    print(y.shape)

for files in dirs[1:]:
    with h5py.File(data_path+files,'r') as fr:
        print('Byl nacten soubor: '+files)
        X2 = fr['imgs'][:]
        y2 = fr['label'][:]
    X = np.concatenate((X,X2), axis = 0)
    y = np.concatenate((y,y2), axis = 0)
    print(X.shape)
    print(y.shape)

#delete garbage
index = 0
num_of_del = 0
while True:
    pomy = y[index]
    if pomy == -2:
        X = np.delete(X,index, axis = 0)
        y = np.delete(y,index, axis = 0)
        num_of_del +=1
    elif pomy == -1:
        y[index] = 10
        index +=1
    else:
        index +=1
    if index >= X.shape[0]:
        break
print(X.shape)
print(y.shape)
print('Pocet smazanych: '+str(num_of_del))

#shuffle data
perm = np.random.permutation(X.shape[0])
XS = np.zeros_like(X)
yS = np.zeros_like(y)
for i in range (0,X.shape[0]):
    pom = perm[i]
    XS[i] = X[pom]
    yS[i] = y[pom]

dst_h5_file = '/storage/plzen1/home/grubiv/AMIR/SPECOM2018/annotations/train_h5/train_'+str(h5_num)+'.h5'
with h5py.File(dst_h5_file, 'w') as fw:
    fw.create_dataset('data', ( XS.shape[0], XS.shape[1],  XS.shape[2]), dtype=np.float32)
    fw.create_dataset('label', (yS.shape[0], yS.shape[1]), dtype=int)
    fw['data'][:] = XS
    fw['label'][:] = yS

data_path_test = '/storage/plzen1/home/grubiv/AMIR/SPECOM2018/annotations/'+str(h5_num)+'_test/'
dirs = os.listdir(data_path_test)
dirs.sort()
print(dirs)
with h5py.File(data_path_test+dirs[0],'r') as fr:
    print('Byl nacten soubor: '+dirs[0])
    X = fr['imgs'][:]
    y = fr['label'][:]
    print(X.shape)
    print(y.shape)
for files in dirs[1:]:
    with h5py.File(data_path_test+files,'r') as fr:
        print('Byl nacten soubor: '+files)
        X2 = fr['imgs'][:]
        y2 = fr['label'][:]
    X = np.concatenate((X,X2), axis = 0)
    y = np.concatenate((y,y2), axis = 0)
    print(X.shape)
    print(y.shape)
index = 0
num_of_del = 0
while True:
    pomy = y[index]
    if pomy == -2:
        X = np.delete(X,index, axis = 0)
        y = np.delete(y,index, axis = 0)
        num_of_del +=1
    elif pomy == -1:
        y[index] = 10
        index +=1
    else:
        index +=1
    if index >= X.shape[0]:
        break
print(X.shape)
print(y.shape)
print('Pocet smazanych: '+str(num_of_del))
perm = np.random.permutation(X.shape[0])
print(perm.shape)
XS = np.zeros_like(X)
print(XS.shape)
yS = np.zeros_like(y)
print(yS.shape)
for i in range (0,X.shape[0]):
    pom = perm[i]
    XS[i] = X[pom]
    yS[i] = y[pom]
dst_h5_file = '/storage/plzen1/home/grubiv/AMIR/SPECOM2018/annotations/test_h5/test_'+str(h5_num)+'.h5'
with h5py.File(dst_h5_file, 'w') as fw:
    fw.create_dataset('data', ( XS.shape[0], XS.shape[1],  XS.shape[2]), dtype=np.float32)
    fw.create_dataset('label', (yS.shape[0], yS.shape[1]), dtype=int)
    fw['data'][:] = XS
    fw['label'][:] = yS