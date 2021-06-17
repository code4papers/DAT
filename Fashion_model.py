#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
fashon_mnist模型训练程序
val_loss: 0.3025
val_acc: 0.8949
'''

from LeNet5 import *
from keras import Model,Input
import sys
import time
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import Dense, Dropout, Activation, Flatten,Input
from keras.models import Model,load_model
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import sys
from utils import mnist_reader


def model_fashion(model_type):
    path = '/Users/qiang.hu/Desktop/work/datasets/fashion_mnist/'
    x_train, y_train = mnist_reader.load_mnist(path, kind='train')
    x_test, y_test = mnist_reader.load_mnist(path, kind='t10k')
    # print(y_train[:100])
    x_train = x_train.astype('float32').reshape(-1, 28, 28, 1)
    x_test = x_test.astype('float32').reshape(-1, 28, 28, 1)
    x_train /= 255
    x_test /= 255
    print('Train:{},Test:{}'.format(len(x_train), len(x_test)))
    nb_classes = 10
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)
    print('data success')
    if model_type == 'lenet5':
        model = Lenet5()
    else:
        model = Lenet1()
    model.summary()
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath='models/Fashion_MNIST_Lenet1/fashion_lenet1_5w.h5', monitor='val_accuracy', mode='auto', save_best_only='True')
    model.fit(x_train[:50000], y_train[:50000], batch_size=64, epochs=15, validation_data=(x_test, y_test), callbacks=[checkpoint])
    model = load_model('models/Fashion_MNIST_Lenet1/fashion_lenet1_5w.h5')
    score = model.evaluate(x_test, y_test, verbose=0)
    print(score)


def train_oe(ood_path, model_type, batch_size, save_path):
    if model_type == 'lenet5':
        model = Lenet5()
    else:
        model = Lenet1()
    path = '/Users/qiang.hu/Desktop/work/datasets/fashion_mnist/'
    x_train, y_train = mnist_reader.load_mnist(path, kind='train')
    x_test, y_test = mnist_reader.load_mnist(path, kind='t10k')
    print(y_train[:100])
    x_train = x_train.astype('float32').reshape(-1, 28, 28, 1)
    x_test = x_test.astype('float32').reshape(-1, 28, 28, 1)
    x_train /= 255
    x_test /= 255
    print('Train:{},Test:{}'.format(len(x_train), len(x_test)))
    nb_classes = 10
    y_train = np_utils.to_categorical(y_train, nb_classes)
    y_test = np_utils.to_categorical(y_test, nb_classes)

    x_train = x_train[:50000]
    y_train = y_train[:50000]

    ood_x = np.load(ood_path)
    ood_x = ood_x.astype('float32') / 255
    ood_x = ood_x.reshape(-1, 28, 28, 1)
    ood_x = ood_x[:50000]
    train_generator = generate_data_generator_for_two_images(x_train, ood_x, y_train, batch_size)
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=my_ood_loss,
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(train_generator,
              steps_per_epoch=len(x_train)/batch_size,
              epochs=15,
              verbose=1)

    score = model.evaluate(x_test, y_test)
    print("test acc: ", score[1])
    model.save(save_path)


if __name__ == '__main__':
    # model_fashion('lenet1')
    model_type = 'lenet5'
    ood_path = "data/fashion_mnist/lenet5_ood_combine.npy"
    save_path = "models/Fashion_MNIST_Lenet5/lenet5_fashion-5w-oe.h5"
    train_oe(ood_path, model_type, 64, save_path)
