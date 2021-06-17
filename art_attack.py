import art
from keras.datasets import mnist, cifar10, cifar100
import numpy as np
import csv
from keras.utils import to_categorical
from keras.models import load_model
import argparse
import tensorflow as tf
from art.estimators.classification import KerasClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent, SaliencyMapMethod, CarliniLInfMethod
# from check_art import *
from utils import mnist_reader
from utils import SVHN_DatasetUtil

tf.compat.v1.disable_eager_execution()


def color_preprocessing(x_train, x_test):

    mean = [0.5071, 0.4867, 0.4408]
    std = [0.2675, 0.2565, 0.2761]
    for i in range(3):
        x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
        x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
    return x_train, x_test


def art_attack_cifar10(model_path, data_type, tr_te, attack_type, save_x_path, save_y_path, eps=8/255, eps_step=8/2550):
    model = load_model(model_path)
    classifier = KerasClassifier(model=model, clip_values=(-1, 1), use_logits=False)
    if attack_type == 'pgd':
        attack = ProjectedGradientDescent(estimator=classifier, norm=np.inf, eps=eps, eps_step=eps_step, max_iter=200)
    if attack_type == 'cw':
        attack = CarliniLInfMethod(classifier=classifier, max_iter=20, eps=eps, batch_size=128, learning_rate=0.1)
    if attack_type == 'fgsm':
        attack = FastGradientMethod(estimator=classifier, norm=np.inf, eps=eps, batch_size=128)
    if data_type == 'cifar10':

        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train_mean = np.mean(x_train[:40000], axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
        if tr_te == 'train':
            x_candidate = x_train
            y_candidate = y_train
        elif tr_te == 'test':
            x_candidate = x_test
            y_candidate = y_test

    elif data_type == 'svhn':
        (X_train, Y_train), (X_test, Y_test) = SVHN_DatasetUtil.load_data()  # 32*32
        if tr_te == 'train':
            x_candidate = X_train
            y_candidate = Y_train
            y_candidate = np.argmax(y_candidate, axis=1)
        elif tr_te == 'test':
            x_candidate = X_test
            y_candidate = Y_test

    y_candidate = y_candidate.reshape(1, -1)[0]
    split = int(len(x_candidate) / 100)
    adv_num = 0
    total_num = 0
    for _ in range(100):
        x_part = x_candidate[split * _: split * (_ + 1)]
        y_part = y_candidate[split * _: split * (_ + 1)]
        ori_predictions = model.predict(x_part.reshape(-1, 32, 32, 3))
        ori_label = np.argmax(ori_predictions, axis=1)
        # print(ori_label)
        # print(y_part)
        correct_classified = np.where(ori_label == y_part)[0]
        print("correct classified: ", len(correct_classified))
        x_part = x_part[correct_classified]
        y_part = y_part[correct_classified]
        x_train_adv = attack.generate(x=x_part)

        predictions = model.predict(x_train_adv)
        adv_num += np.sum(np.argmax(predictions, axis=1) != y_part)
        adv_index = np.where(np.argmax(predictions, axis=1) != y_part)[0]
        if _ == 0:
            adv_data = x_train_adv[adv_index]
            adv_label = y_part[adv_index]
        else:
            adv_data = np.concatenate((x_train_adv[adv_index], adv_data))
            adv_label = np.concatenate((y_part[adv_index], adv_label))
        del x_part
        del x_train_adv
        # print(adv_num)
        # print(total_num)

    if data_type == 'cifar10':
        adv_data += x_train_mean
        adv_data *= 255
        adv_data = adv_data.astype('int')
    elif data_type == 'svhn':
        adv_data *= 255
        adv_data = adv_data.astype('int')
    np.save(save_x_path, adv_data)
    np.save(save_y_path, adv_label)


def art_attack_mnist(model_path, data_type, tr_te, attack_type, save_x_path, save_y_path, eps=8/255, eps_step=8/2550):
    # model_path = "models/MNIST-Lenet5/Lenet-5.h5"
    # data_type = 'mnist'
    # attack_type = 'cw'
    # save_x_path = "data/mnist/cw_inf.npy"
    # save_y_path = "data/mnist/cw_inf_label.npy"

    # if data_type == 'mnist' or data_type == 'fashion_mnist':
    model = load_model(model_path)
    classifier = KerasClassifier(model=model, clip_values=(0, 1), use_logits=False)
    if attack_type == 'pgd':
        attack = ProjectedGradientDescent(estimator=classifier, norm=np.inf, eps=eps, eps_step=eps_step, max_iter=200)
    if attack_type == 'cw':
        attack = CarliniLInfMethod(classifier=classifier, max_iter=50, eps=eps, batch_size=5, learning_rate=0.1)
    if attack_type == 'fgsm':
        attack = FastGradientMethod(estimator=classifier, norm=np.inf, eps=eps, batch_size=128)

    if data_type == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32') / 255
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.astype('float32') / 255
        x_test = x_test.reshape(-1, 28, 28, 1)
        # y_train_eva = to_categorical(y_test, 10)
        if tr_te == 'train':
            x_candidate = x_train
            y_candidate = y_train
        elif tr_te == 'test':
            x_candidate = x_test
            y_candidate = y_test

    elif data_type == 'fashion_mnist':
        path = 'data/fashion_mnist_ori/'
        x_train, y_train = mnist_reader.load_mnist(path, kind='train')
        x_test, y_test = mnist_reader.load_mnist(path, kind='t10k')
        x_train = x_train.astype('float32').reshape(-1, 28, 28, 1)
        x_test = x_test.astype('float32').reshape(-1, 28, 28, 1)
        x_train /= 255
        x_test /= 255
        print('Train:{},Test:{}'.format(len(x_train), len(x_test)))
        if tr_te == 'train':
            x_candidate = x_train
            y_candidate = y_train
        elif tr_te == 'test':
            x_candidate = x_test
            y_candidate = y_test

    y_candidate = y_candidate.reshape(1, -1)[0]
    split = int(len(x_candidate) / 100)
    adv_num = 0
    total_num = 0
    for _ in range(100):
        x_part = x_candidate[split * _: split * (_ + 1)]
        y_part = y_candidate[split * _: split * (_ + 1)]
        ori_predictions = model.predict(x_part.reshape(-1, 28, 28, 1))
        ori_label = np.argmax(ori_predictions, axis=1)
        # print(ori_label)
        # print(y_part)
        correct_classified = np.where(ori_label == y_part)[0]
        print("correct classified: ", len(correct_classified))
        x_part = x_part[correct_classified]
        y_part = y_part[correct_classified]
        x_train_adv = attack.generate(x=x_part)

        predictions = model.predict(x_train_adv)
        adv_num += np.sum(np.argmax(predictions, axis=1) != y_part)
        adv_index = np.where(np.argmax(predictions, axis=1) != y_part)[0]

        if _ == 0:
            adv_data = x_train_adv[adv_index]
            adv_label = y_part[adv_index]
        else:
            adv_data = np.concatenate((x_train_adv[adv_index], adv_data))
            adv_label = np.concatenate((y_part[adv_index], adv_label))
        del x_part
        del x_train_adv
        # print(adv_num)
        # print(total_num)
    adv_data = adv_data * 255
    adv_data = adv_data.astype('int')
    np.save(save_x_path, adv_data)
    np.save(save_y_path, adv_label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", "-eps",
                        type=float,
                        default=8/255,
                        )
    parser.add_argument("--eps_step", "-eps_step",
                        type=float,
                        default=8/2550,
                        )
    parser.add_argument("--attack_type", "-attack_type",
                        type=str,
                        default='fgsm',
                        )
    parser.add_argument("--model_type", "-model_type",
                        type=str,
                        default='lenet1',
                        )
    parser.add_argument("--data_type", "-data_type",
                        type=str,
                        default='mnist',
                        )
    args = parser.parse_args()
    eps = args.eps
    eps_step = args.eps_step
    data_type = args.data_type
    model_type = args.model_type
    attack_type = args.attack_type
    if data_type == 'mnist':
        if model_type == 'lenet1':
            # training data attack
            model_path = "models/MNIST-Lenet1/Lenet-1-5w.h5"
            save_x_path = "data/mnist/lenet1_" + attack_type + "_x.npy"
            save_y_path = "data/mnist/lenet1_" + attack_type + "_y.npy"
            art_attack_mnist(model_path, data_type, 'train', attack_type, save_x_path, save_y_path, eps=0.3,
                             eps_step=0.03)
            # test data attack
            save_x_path = "data/mnist/lenet1_" + attack_type + "_x_test.npy"
            save_y_path = "data/mnist/lenet1_" + attack_type + "_y_test.npy"
            art_attack_mnist(model_path, data_type, 'test', attack_type, save_x_path, save_y_path, eps=0.3,
                             eps_step=0.03)

        elif model_type == 'lenet5':
            # training data attack
            model_path = "models/MNIST-Lenet5/Lenet-5-5w.h5"
            save_x_path = "data/mnist/lenet5_" + attack_type + "_x.npy"
            save_y_path = "data/mnist/lenet5_" + attack_type + "_y.npy"
            art_attack_mnist(model_path, data_type, 'train', attack_type, save_x_path, save_y_path, eps=0.3,
                             eps_step=0.03)
            # test data attack
            save_x_path = "data/mnist/lenet5_" + attack_type + "_x_test.npy"
            save_y_path = "data/mnist/lenet5_" + attack_type + "_y_test.npy"
            art_attack_mnist(model_path, data_type, 'test', attack_type, save_x_path, save_y_path, eps=0.3,
                             eps_step=0.03)
    elif data_type == 'fashion_mnist':
        if model_type == 'lenet1':
            # training data attack
            model_path = "models/Fashion_MNIST_Lenet1/fashion_lenet1_5w.h5"
            save_x_path = "data/fashion_mnist/lenet1_" + attack_type + "_x.npy"
            save_y_path = "data/fashion_mnist/lenet1_" + attack_type + "_y.npy"
            art_attack_mnist(model_path, data_type, 'train', attack_type, save_x_path, save_y_path, eps=0.3,
                             eps_step=0.03)
            # test data attack
            save_x_path = "data/fashion_mnist/lenet1_" + attack_type + "_x_test.npy"
            save_y_path = "data/fashion_mnist/lenet1_" + attack_type + "_y_test.npy"
            art_attack_mnist(model_path, data_type, 'test', attack_type, save_x_path, save_y_path, eps=0.3,
                             eps_step=0.03)
        elif model_type == 'lenet5':
            # training data attack
            model_path = "models/Fashion_MNIST_Lenet5/model_fashion_5w.h5"
            save_x_path = "data/fashion_mnist/lenet5_" + attack_type + "_x.npy"
            save_y_path = "data/fashion_mnist/lenet5_" + attack_type + "_y.npy"
            art_attack_mnist(model_path, data_type, 'train', attack_type, save_x_path, save_y_path, eps=0.3,
                             eps_step=0.03)
            # test data attack
            save_x_path = "data/fashion_mnist/lenet5_" + attack_type + "_x_test.npy"
            save_y_path = "data/fashion_mnist/lenet5_" + attack_type + "_y_test.npy"
            art_attack_mnist(model_path, data_type, 'test', attack_type, save_x_path, save_y_path, eps=0.3,
                             eps_step=0.03)
    elif data_type == 'svhn':
        if model_type == 'lenet5':
            # training data attack
            model_path = "models/SVHN-Lenet5/svhn-Lenet5-5w.h5"
            save_x_path = "data/svhn_new/lenet5_" + attack_type + "_x.npy"
            save_y_path = "data/svhn_new/lenet5_" + attack_type + "_y.npy"
            art_attack_cifar10(model_path, data_type, 'train', attack_type, save_x_path, save_y_path, eps=8/255,
                             eps_step=8/2550)
            # test data attack
            save_x_path = "data/svhn_new/lenet5_" + attack_type + "_x_test.npy"
            save_y_path = "data/svhn_new/lenet5_" + attack_type + "_y_test.npy"
            art_attack_cifar10(model_path, data_type, 'test', attack_type, save_x_path, save_y_path, eps=8/255,
                             eps_step=8/2550)
        elif model_type == 'resnet20':
            # training data attack
            model_path = "models/SVHN-ResNet20/svhn-resnet20-5w.h5"
            save_x_path = "data/svhn_new/resnet20_" + attack_type + "_x.npy"
            save_y_path = "data/svhn_new/resnet20_" + attack_type + "_y.npy"
            art_attack_cifar10(model_path, data_type, 'train', attack_type, save_x_path, save_y_path, eps=8 / 255,
                             eps_step=8 / 2550)
            # test data attack
            save_x_path = "data/svhn_new/resnet20_" + attack_type + "_x_test.npy"
            save_y_path = "data/svhn_new/resnet20_" + attack_type + "_y_test.npy"
            art_attack_cifar10(model_path, data_type, 'test', attack_type, save_x_path, save_y_path, eps=8 / 255,
                             eps_step=8 / 2550)
    else:
        if model_type == 'nin':
            # training data attack
            model_path = "models/Cifar10_NiN/NiN-4w.h5"
            save_x_path = "data/cifar10/nin_" + attack_type + "_x.npy"
            save_y_path = "data/cifar10/nin_" + attack_type + "_y.npy"
            art_attack_cifar10(model_path, data_type, 'train', attack_type, save_x_path, save_y_path, eps=8 / 255,
                             eps_step=8 / 2550)
            # test data attack
            save_x_path = "data/cifar10/nin_" + attack_type + "_x_test.npy"
            save_y_path = "data/cifar10/nin_" + attack_type + "_y_test.npy"
            art_attack_cifar10(model_path, data_type, 'test', attack_type, save_x_path, save_y_path, eps=8 / 255,
                             eps_step=8 / 2550)
        elif model_type == 'resnet20':
            # training data attack
            model_path = "models/Cifar10_ReNet20/ResNet20_4w.h5"
            save_x_path = "data/cifar10/resnet20_" + attack_type + "_x.npy"
            save_y_path = "data/cifar10/resnet20_" + attack_type + "_y.npy"
            art_attack_cifar10(model_path, data_type, 'train', attack_type, save_x_path, save_y_path, eps=8 / 255,
                             eps_step=8 / 2550)
            # test data attack
            save_x_path = "data/cifar10/resnet20_" + attack_type + "_x_test.npy"
            save_y_path = "data/cifar10/resnet20_" + attack_type + "_y_test.npy"
            art_attack_cifar10(model_path, data_type, 'test', attack_type, save_x_path, save_y_path, eps=8 / 255,
                             eps_step=8 / 2550)













