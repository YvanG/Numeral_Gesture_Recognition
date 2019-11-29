#!/usr/bin/env python

__author__ = "Ivan Gruber"
__version__ = "1.0.0"
__maintainer__ = "Ivan Gruber"
__email__ = "ivan.gruber@seznam.cz"

"""
Data preprocessing
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import cv2
import h5py

#data loading
data_path = '/storage/plzen1/home/grubiv/AMIR/SPECOM2018/annotations/train_h5/'
dirs = os.listdir(data_path)
dirs.sort()
print(dirs)
poc = 0

for files in dirs:
    f_out = data_path+files
    f_new = data_path+'trainN_'+str(poc)+'.h5'

    with h5py.File(f_out,'r') as fr:
        print('Byl nacten soubor: '+files)
        X = fr['data'][:]
        y = fr['label'][:]
    minim = X.min()
    X += np.abs(minim)
    maxim = X.max()
    X = X/maxim
    with h5py.File(f_new, 'w') as fw:
        fw.create_dataset('data', (X.shape[0], X.shape[1],  X.shape[2]), dtype=np.float32)
        fw.create_dataset('label', (y.shape[0],), dtype=int)
        fw['data'][:] = X[:]
        fw['label'][:] = y[:]
        print('Soubor: '+files+' byl ulozen!')
    poc +=1

data_path = '/storage/plzen1/home/grubiv/AMIR/SPECOM2018/annotations/test_h5/'
dirs = os.listdir(data_path)
dirs.sort()
print(dirs)

#data normalization
poc = 0
for files in dirs:
    f_out = data_path+files
    f_new = data_path+'trainN_'+str(poc)+'.h5'

    with h5py.File(f_out,'r') as fr:
        print('Byl nacten soubor: '+files)
        X = fr['data'][:]
        y = fr['label'][:]
    minim = X.min()
    X += np.abs(minim)
    maxim = X.max()
    X = X/maxim
    with h5py.File(f_new, 'w') as fw:
        fw.create_dataset('data', (X.shape[0], X.shape[1],  X.shape[2]), dtype=np.float32)
        fw.create_dataset('label', (y.shape[0],), dtype=int)
        fw['data'][:] = X[:]
        fw['label'][:] = y[:]
        print('Soubor: '+files+' byl ulozen!')
    poc +=1