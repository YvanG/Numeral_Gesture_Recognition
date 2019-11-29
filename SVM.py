#!/usr/bin/env python

__author__ = "Ivan Gruber"
__version__ = "1.0.0"
__maintainer__ = "Ivan Gruber"
__email__ = "ivan.gruber@seznam.cz"

"""
SVM over HoG calculation
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import  scipy.stats as stats
import cv2
import h5py
from skimage.feature import hog
from skimage import data, exposure
from sklearn import svm

#HoG loading
data_path = '/storage/plzen1/home/grubiv/AMIR/SPECOM2018/annotations/HoGs/'
filename = 'HoG_trainN_3.h5'
f_out = data_path+filename
with h5py.File(f_out,'r') as fr:
    print('Byl nacten soubor: '+f_out)
    X = fr['data'][:]
    y = fr['label'][:]
print(X.shape)
print(y.shape)

#SVM training
svc = svm.SVC()
svc.fit(X, y)

filename = 'HoG_testN_3.h5'
f_out = data_path+filename

with h5py.File(f_out,'r') as fr:
    print('Byl nacten soubor: '+f_out)
    X = fr['data'][:]
    y = fr['label'][:]
print(X.shape)
print(y.shape)

y_pred = svc.predict(X)
print("Number of mislabeled points : %d" % (y != y_pred).sum())
