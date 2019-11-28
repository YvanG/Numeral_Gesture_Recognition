#!/usr/bin/env python

__author__ = "Ivan Gruber"
__version__ = "1.0.0"
__maintainer__ = "Ivan Gruber"
__email__ = "ivan.gruber@seznam.cz"

"""
Modified VGG16 architecture for numeral gesture recognition
Input: Depth image - 96x96 pixels, normalized (0,1)
Output: 1 class for each numeral (10 in total) and 1 class for background (garbage class)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import keras
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, Layer
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.models import load_model
import tensorflow as tf
from keras.callbacks import LearningRateScheduler
import math
import h5py

h5_file = '/storage/plzen1/home/grubiv/AMIR/SPECOM2018/annotations/train_h5/trainN_1.h5'
with h5py.File(h5_file,'r') as fr:
        X = fr['data'][:]
        y = fr['label'][:]
y_one_hot = np.zeros((y.shape[0], 11))
y_one_hot[np.arange(y.shape[0]), y] = 1
X2 = np.expand_dims(X, axis = 3)

h5_file = '/storage/plzen1/home/grubiv/AMIR/SPECOM2018/annotations/test_h5/testN_1.h5'
with h5py.File(h5_file,'r') as fr:
        X_dev = fr['data'][:]
        y_dev = fr['label'][:]
y_one_hot_dev = np.zeros((y_dev.shape[0], 11))
y_one_hot_dev[np.arange(y_dev.shape[0]), y_dev] = 1
X2_dev = np.expand_dims(X_dev, axis = 3)

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(96,96,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(11, activation='softmax'))
model.summary()

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics = ['accuracy'])
epochs = 15
batch_size = 64
model.fit(X2, y_one_hot, epochs = epochs, batch_size = batch_size, validation_data = (X2_dev, y_one_hot_dev))
model.save('/storage/plzen1/home/grubiv/AMIR/SPECOM2018/NN/AMIR_VGG1.h5')