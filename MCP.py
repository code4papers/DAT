#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 10:40:17 2019

@author: qq
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import keras
from keras.datasets import mnist
from keras.datasets import cifar10
from keras import optimizers
import random
from keras.utils import to_categorical
import numpy as np
from keras.models import load_model
from matplotlib import pyplot as plt
import csv
from keras.utils import np_utils
from keras.models import Model, Input, load_model
import pandas as pd
import argparse
# import condition

CLIP_MAX = 0.5


# 输入：待测模型、排序度量lsa还是dsa,待测试的集合，选择的size
# 输出：准确率，在待测试的集合上
# 把预测的标签作为伪标签
def retrain(model, args, layer_names, selectsize=100, attack='fgsm', measure='lsa', datatype='mnist'):
    (x_train, __), (__, __) = mnist.load_data()
    # npzfile=np.load('mnist.npz')
    # y_train= npzfile['y_train']
    # x_train= npzfile['x_train']
    x_train = x_train.astype("float32").reshape(-1, 28, 28, 1)
    x_train = (x_train / 255.0) - (1.0 - CLIP_MAX)
    npzfile = np.load('./adv/data/mnist/mnist_' + attack + '_compound8.npz')
    y_test = npzfile['y_test']
    x_test = npzfile['x_test']

    x_target = x_test

    target_lst = []

    # baselines =['LSA','DSA','CES','MCP','SRS','AAL']

    if measure == 'SRS':
        x_select, y_select = select_rondom(selectsize, x_target, x_target, y_test)
    if measure == 'MCP':
        x_select, y_select = select_my_optimize(model, selectsize, x_target, y_test)

    if measure == 'AAL':
        path = "./mnist_finalResults/mnist_" + attack + "_compound8_result.csv"
        csv_data = pd.read_csv(path, header=None)
        target_lst = []
        for i in range(len(csv_data.values.T)):
            target_lst.append(csv_data.values.T[i])
    # if measure == 'CES':
    #     indexlst = condition.conditional_sample(model, x_target, selectsize)
    #     x_select, y_select = select_from_index(selectsize, x_target, indexlst, y_test)
    elif measure not in ['SRS', 'MCP']:
        x_select, y_select = select_from_large(selectsize, x_target, target_lst, y_test)
    y_select = np_utils.to_categorical(y_select, 10)
    y_test = np_utils.to_categorical(y_test, 10)

    score = model.evaluate(x_target, y_test, verbose=0)
    # print('Test Loss: %.4f' % score[0])
    print('Before retrain, Test accuracy: %.4f' % score[1])
    origin_acc = score[1]

    # sgd = optimizers.SGD(lr=0.01)
    # loss="categorical_crossentropy", optimizer="adadelta", metrics=["accuracy"]
    model.compile(loss='categorical_crossentropy', optimizer="adadelta", metrics=['accuracy'])

    retrain_acc = 0

    model.fit(x_select, y_select, batch_size=100, epochs=5, shuffle=True, verbose=1, validation_data=(x_target, y_test))
    score = model.evaluate(x_target, y_test, verbose=0)
    retrain_acc = score[1]

    return retrain_acc, origin_acc


def retrain_2(model, data_path, selectsize=100):
    (_, _), (_, y_test) = mnist.load_data()
    x_test = np.load(data_path)
    x_select, y_select = select_my_optimize(model, selectsize, x_test, y_test)
    y_test = to_categorical(y_test, 10)
    y_select = to_categorical(y_select, 10)
    ori_score = model.evaluate(x_test, y_test, verbose=0)
    print("ori acc:{}".format(ori_score[1]))
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    model.fit(x_select, y_select, batch_size=100, epochs=5, shuffle=True, verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)
    print("MPC acc:{}".format(score[1]))
    return score[1]


def retrain_3(model, data_path, selectsize=100):
    (_, y_train), (_, _) = mnist.load_data()
    x_train = np.load(data_path)
    x_train = x_train.astype("float32") / 255
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_select, y_select, selected_idx = select_my_optimize(model, selectsize, x_train, y_train)
    selected_idx = np.asarray(selected_idx)
    margin_list = margin_score(model, x_train)
    print(margin_list[selected_idx])
    # x_select = x_train[selected_idx]
    # y_select = y_train[selected_idx]
    # y_train = to_categorical(y_train, 10)
    # y_select = to_categorical(y_select, 10)
    #
    # ori_score = model.evaluate(x_train, y_train, verbose=0)
    # print("ori acc:{}".format(ori_score[1]))
    # model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    # model.fit(x_select, y_select, batch_size=256, epochs=10, shuffle=True, verbose=1)
    # score = model.evaluate(x_train, y_train, verbose=0)
    # print("MPC acc:{}".format(score[1]))
    return margin_list[selected_idx]


def find_second(act, ncl=10):
    max_ = 0
    second_max = 0
    sec_index = 0
    max_index = 0
    for i in range(ncl):
        if act[i] > max_:
            max_ = act[i]
            max_index = i

    for i in range(ncl):
        if i == max_index:
            continue
        if act[i] > second_max:  # 第2大加一个限制条件，那就是不能和max_一样
            second_max = act[i]
            sec_index = i
    ratio = 1.0 * second_max / max_
    # print 'max:',max_index
    return max_index, sec_index, ratio  # ratio是第二大输出达到最大输出的百分比

# for selection only
def select_only(model, selectsize, x_target):
    act_layers = model.predict(x_target)
    dicratio = [[] for i in range(100)]  # 只用90，闲置10个
    dicindex = [[] for i in range(100)]
    for i in range(len(act_layers)):
        act = act_layers[i]
        max_index, sec_index, ratio = find_second(act)  # max_index
        # 安装第一和第二大的标签来存储，比如第一是8，第二是4，那么就存在84里，比例和测试用例的序号
        dicratio[max_index * 10 + sec_index].append(ratio)
        dicindex[max_index * 10 + sec_index].append(i)

    selected_lst = select_from_firstsec_dic(selectsize, dicratio, dicindex)
    # selected_lst,lsa_lst = order_output(target_lsa,select_amount)
    selected_idx = []
    for i in range(selectsize):
        selected_idx.append(selected_lst[i])

    return selected_idx

# for wilds datasets
def select_wilds_only(model, selectsize, x_target, ncl):
    window_size = int(ncl * ncl)
    act_layers = model.predict(x_target)
    dicratio = [[] for i in range(window_size)]  # 只用90，闲置10个
    dicindex = [[] for i in range(window_size)]
    for i in range(len(act_layers)):
        act = act_layers[i]
        max_index, sec_index, ratio = find_second(act, ncl)  # max_index
        # 安装第一和第二大的标签来存储，比如第一是8，第二是4，那么就存在84里，比例和测试用例的序号
        dicratio[max_index * ncl + sec_index].append(ratio)
        dicindex[max_index * ncl + sec_index].append(i)

    selected_lst = select_from_firstsec_dic(selectsize, dicratio, dicindex)
    # selected_lst,lsa_lst = order_output(target_lsa,select_amount)
    selected_idx = []
    for i in range(selectsize):
        selected_idx.append(selected_lst[i])

    return selected_idx



# 返回的是我们的方法优化版的采样用例
def select_my_optimize(model, selectsize, x_target, y_test):
    x = np.zeros((selectsize, 28, 28, 1))
    y = np.zeros((selectsize,))

    act_layers = model.predict(x_target)
    dicratio = [[] for i in range(100)]  # 只用90，闲置10个
    dicindex = [[] for i in range(100)]
    for i in range(len(act_layers)):
        act = act_layers[i]
        max_index, sec_index, ratio = find_second(act)  # max_index
        # 安装第一和第二大的标签来存储，比如第一是8，第二是4，那么就存在84里，比例和测试用例的序号
        dicratio[max_index * 10 + sec_index].append(ratio)
        dicindex[max_index * 10 + sec_index].append(i)

    selected_lst = select_from_firstsec_dic(selectsize, dicratio, dicindex)
    dicratio = np.asarray(dicratio)
    # print(dicratio[selected_lst])
    # print(selected_lst)
    # selected_lst,lsa_lst = order_output(target_lsa,select_amount)
    selected_idx = []
    for i in range(selectsize):
        x[i] = x_target[selected_lst[i]]
        y[i] = y_test[selected_lst[i]]
        selected_idx.append(selected_lst[i])

    return x, y, selected_idx


# 输入第一第二大的字典，输出selected_lst。用例的index
def select_from_firstsec_dic(selectsize, dicratio, dicindex, ncl=10):
    selected_lst = []
    tmpsize = selectsize
    # tmpsize保存的是采样大小，全程都不会变化

    noempty = no_empty_number(dicratio)
    # print(selectsize)
    # print(noempty)
    # 待选择的数目大于非空的类别数(满载90类)，每一个都选一个
    window_size = int(ncl * ncl)
    while selectsize >= noempty:
        for i in range(window_size):
            if len(dicratio[i]) != 0:  # 非空就选一个最大的出来
                tmp = max(dicratio[i])
                j = dicratio[i].index(tmp)
                # if tmp>=0.1:
                selected_lst.append(dicindex[i][j])
                dicratio[i].remove(tmp)
                dicindex[i].remove(dicindex[i][j])
        selectsize = tmpsize - len(selected_lst)
        noempty = no_empty_number(dicratio)
        # print(selectsize)
    # selectsize<noempty
    # no_empty_number(dicratio)
    # print(selectsize)

    # 剩下少量样本没有采样，比如还存在30类别非空，但是只要采样10个，此时我们取30个最大值中的前10大
    while len(selected_lst) != tmpsize:
        max_tmp = [0 for i in range(selectsize)]  # 剩下多少就申请多少
        max_index_tmp = [0 for i in range(selectsize)]
        for i in range(window_size):
            if len(dicratio[i]) != 0:
                tmp_max = max(dicratio[i])
                if tmp_max > min(max_tmp):
                    index = max_tmp.index(min(max_tmp))
                    max_tmp[index] = tmp_max
                    # selected_lst.append()
                    # if tmp_max>=0.1:
                    max_index_tmp[index] = dicindex[i][dicratio[i].index(tmp_max)]  # 吧样本序列号存在此列表中
        if len(max_index_tmp) == 0 and len(selected_lst) != tmpsize:
            print('wrong!!!!!!')
            break
        selected_lst = selected_lst + max_index_tmp
        print(len(selected_lst))
    # print(selected_lst)
    assert len(selected_lst) == tmpsize
    return selected_lst


# 配对表情非空的数目。比如第一是3，第二是5，此时里面没有任何实例存在那么就是0
def no_empty_number(dicratio):
    no_empty = 0
    for i in range(len(dicratio)):
        if len(dicratio[i]) != 0:
            no_empty += 1
    return no_empty


def select_from_large(select_amount, x_target, target_lsa, y_test):
    x = np.zeros((select_amount, 28, 28, 1))
    y = np.zeros((select_amount,))

    selected_lst, lsa_lst = order_output(target_lsa, select_amount)
    # print(lsa_lst)
    # print(selected_lst)
    for i in range(select_amount):
        x[i] = x_target[selected_lst[i]]
        y[i] = y_test[selected_lst[i]]
    return x, y


def select_rondom(select_amount, x_target, target_lsa, y_test):
    x = np.zeros((select_amount, 28, 28, 1))
    y = np.zeros((select_amount,))

    selected_lst = np.random.choice(range(len(target_lsa)), replace=False, size=select_amount)
    # selected_lst,lsa_lst = order_output(target_lsa,select_amount)
    for i in range(select_amount):
        x[i] = x_target[selected_lst[i]]
        y[i] = y_test[selected_lst[i]]
    return x, y


# 根据indexlist来选择用例
def select_from_index(select_amount, x_target, indexlst, y_test):
    x = np.zeros((select_amount, 28, 28, 1))
    y = np.zeros((select_amount,))
    # print(indexlst)
    for i in range(select_amount):
        x[i] = x_target[indexlst[i]]
        y[i] = y_test[indexlst[i]]
    return x, y


# 找到前select_amount大的值的index输出
# 这个函数得修改一下

# 找出max_lsa在 target_lsa中的index，排除selected_lst中已经选的
def find_index(target_lsa, selected_lst, max_lsa):
    for i in range(len(target_lsa)):
        if max_lsa == target_lsa[i] and i not in selected_lst:
            return i
    return 0


# 重新修改
def order_output(target_lsa, select_amount):
    lsa_lst = []

    tmp_lsa_lst = target_lsa[:]
    selected_lst = []
    while len(selected_lst) < select_amount:
        max_lsa = max(tmp_lsa_lst)
        selected_lst.append(find_index(target_lsa, selected_lst, max_lsa))
        lsa_lst.append(max_lsa)
        tmp_lsa_lst.remove(max_lsa)
    return selected_lst, lsa_lst


def fetch_our_measure(model, x_target):
    bound_data_lst = []
    # x_test=x_test.astype('float32').reshape(-1,28,28,1)
    # x_test/=255
    act_layers = model.predict(x_target)

    ratio_lst = []
    for i in range(len(act_layers)):
        act = act_layers[i]
        _, __, ratio = find_second(act)
        ratio_lst.append(ratio)

    return ratio_lst


def createdataset(attack, ratio=8):
    if attack in ['rotation', 'translation', 'shear', 'brightness', 'contrast', 'scale']:
        x_target = np.load('./imagetrans/mnist_' + attack + '.npy')
    else:
        x_target = np.load('./adv/data/mnist/Adv_mnist_' + attack + '.npy')
    if attack in ['rotation', 'translation', 'shear', 'brightness', 'contrast', 'scale']:
        x_target = x_target.astype("float32").reshape(-1, 28, 28, 1)
        x_target = (x_target / 255.0) - (1.0 - CLIP_MAX)

    npzfile = np.load('./mnist.npz')
    y_test = npzfile['y_test']
    x_test = npzfile['x_test']

    x_test = x_test.astype("float32").reshape(-1, 28, 28, 1)
    x_test = (x_test / 255.0) - (1.0 - CLIP_MAX)

    origin_lst = np.random.choice(range(10000), replace=False, size=ratio * 1000)
    mutated_lst = np.random.choice(range(10000), replace=False, size=10000 - ratio * 1000)

    x_dest = np.append(x_test[origin_lst], x_target[mutated_lst], axis=0)
    y_dest = np.append(y_test[origin_lst], y_test[mutated_lst])
    np.savez('./adv/data/mnist/mnist_' + attack + '_compound8.npz', x_test=x_dest, y_test=y_dest)

    y_dest = np_utils.to_categorical(y_dest, 10)
    # score = model.evaluate(x_dest, y_dest,verbose=0)
    # print('Test Loss: %.4f' % score[0])
    # print('Before retrain, Test accuracy: %.4f'% score[1])
    return


def MCP_selection_wilds(model, target_data, select_size, ncl, margin_score):
    print("Prepare...")
    select_index = select_wilds_only(model, select_size, target_data, ncl)
    print(margin_score[select_index])
    print("Over...")
    return select_index


def margin_score(model, target_data):
    prediction = model.predict(target_data)
    prediction_sorted = np.sort(prediction)

    margin_list = prediction_sorted[:, -1] - prediction_sorted[:, -2]

    return margin_list


if __name__ == "__main__":

    acc_ls = []
    for i in range(1):
        model = load_model("models/MNIST-Lenet5/Lenet-5.h5")
        data_path = "data/mnist/translation.npy"
        acc = retrain_3(model, data_path, 100)
        acc_ls.append(acc)
        # MCP_selection_wilds(model, )


    # print("mean acc: {}".format(np.mean(acc_ls)))
    # (_, _), (x_test, y_test) = mnist.load_data()
    # x_test = x_test.astype('float32') / 255
    # x_test = x_test.reshape(-1, 28, 28, 1)
    # y_test = to_categorical(y_test, 10)
    # score = model.evaluate(x_test, y_test)
    # print(score[1])

