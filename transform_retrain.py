from scipy.stats import entropy
import numpy as np
from keras.datasets import mnist
from keras.utils import to_categorical
import keras
from MCP import *
import csv
import CES
import keras.backend as K
import gc
from sa import *
from data_combine import *
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from utils import mnist_reader
from random import normalvariate
from utils import SVHN_DatasetUtil
from utils import SVHN_DatasetUtil


def normal_choice(lst, my_dev, mean=None, stddev=None):
    if mean is None:
        # if mean is not specified, use center of list
        mean = (len(lst) - 1) / 2

    if stddev is None:
        # if stddev is not specified, let list be -3 .. +3 standard deviations
        stddev = len(lst) / my_dev

    while True:
        index = int(normalvariate(mean, stddev) + 0.5)
        if 0 <= index < len(lst):
            return lst[index]

def entropy_selection(model, target_data, select_size):
    # print("Prepare...")
    prediction = model.predict(target_data)
    entropy_list = entropy(prediction, base=2, axis=1)

    sorted_index = np.argsort(entropy_list)
    select_index = sorted_index[-select_size:]
    # print("Over...")
    return select_index


def entropy_score(model, target_data):
    # print("Prepare...")
    prediction = model.predict(target_data)
    entropy_list = entropy(prediction, base=2, axis=1)

    sorted_index = np.argsort(entropy_list)
    # select_index = sorted_index[-select_size:]
    # print("Over...")
    return entropy_list


def deepgini_selection(model, target_data, select_size):
    # print("Prepare...")
    prediction = model.predict(target_data)
    entropy_list = np.sum(prediction ** 2, axis=1)

    sorted_index = np.argsort(entropy_list)
    select_index = sorted_index[:select_size]
    # print("Over...")
    return select_index


def deepgini_score(model, target_data):
    prediction = model.predict(target_data)
    gini_list = np.sum(prediction ** 2, axis=1)
    return gini_list


def MCP_selection(model, target_data, select_size):
    print("Prepare...")
    select_index = select_only(model, select_size, target_data)
    print("Over...")
    return select_index


def MCP_selection_wilds(model, target_data, select_size, ncl):
    print("Prepare...")
    select_index = select_wilds_only(model, select_size, target_data, ncl)
    print("Over...")
    return select_index


def random_selection(model, target_data, select_size):
    print("Prepare...")
    all_index = np.arange(len(target_data))
    select_index = np.random.choice(all_index, select_size, replace=False)
    print("Over...")
    return select_index


def margin_selection(model, target_data, select_size):
    prediction = model.predict(target_data)
    prediction_sorted = np.sort(prediction)

    margin_list = prediction_sorted[:, -1] - prediction_sorted[:, -2]
    sorted_index = np.argsort(margin_list)
    select_index = sorted_index[: select_size]
    return select_index



def fin_tune_lenet5(model, data_type, data_style, data_path, index_path, ratio, select_size, metric, test_data_path, args):
    if data_type == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        model_save_path = "models/lenet5_try_" + args.model_type + data_style + '.h5'
        x_train = x_train
        y_train = y_train
        batch_size = 256
        epochs = 5
        split_len = len(x_train[:-10000])
        if args.model_type == 'lenet1':
            dsa_layer = ['dense']
        else:
            dsa_layer = ['fc2']
        x_train_mean = None
    elif data_type == 'fashion_mnist':
        path = 'data/fashion_mnist_ori/'
        x_train, y_train = mnist_reader.load_mnist(path, kind='train')
        x_test, y_test = mnist_reader.load_mnist(path, kind='t10k')
        x_train = x_train.astype('float32').reshape(-1, 28, 28, 1)
        x_test = x_test.astype('float32').reshape(-1, 28, 28, 1)
        x_train /= 255
        x_test /= 255
        print('Train:{},Test:{}'.format(len(x_train), len(x_test)))
        nb_classes = 10
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)
        model_save_path = "models/fashion_try_" + args.model_type +data_style + '.h5'
        batch_size = 64
        x_train = x_train
        y_train = y_train
        epochs = 5
        split_len = len(x_train[:-10000])
        if args.model_type == 'lenet1':
            dsa_layer = ['dense']
        else:
            dsa_layer = ['fc2']
        x_train_mean = None
    elif data_type == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train_mean = np.mean(x_train[:40000], axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        model_save_path = "models/cifar_" + args.model_type + data_style + '.h5'
        batch_size = 128
        epochs = 10
        split_len = len(x_train[:-10000])
        if args.model_type == 'nin':
            dsa_layer = ['global_average_pooling2d']
        else:
            dsa_layer = ['before_softmax']

    else:
        (X_train, Y_train), (X_test, Y_test) = SVHN_DatasetUtil.load_data()  # 32*32
        x_train = X_train
        y_train = Y_train
        x_test = X_test[:10000]
        y_test = Y_test[:10000]
        batch_size = 128
        model_save_path = "models/svhn_" + args.model_type + data_style + '.h5'
        epochs = 5
        split_len = len(x_train[:-10000])
        if args.model_type == 'dense_2':
            dsa_layer = ['dense_2']
        else:
            dsa_layer = ['before_softmax']
        x_train_mean = None
    if data_style == 'fgsm' or data_style == 'pgd':
        train_index_path = index_path + '.npy'
        candidate_data, candidate_label, ood_candidate_index, id_candidate_index = combine_train_and_new_adv(x_train[split_len:], y_train[split_len:], data_type, data_path, train_index_path, ratio, x_train_mean=x_train_mean)


        test_index_path = index_path + '_test.npy'
        new_test_data, new_test_y, ood_test_index, id_test_index = combine_train_and_new_adv(
            x_test[:10000], y_test[:10000], data_type, test_data_path, test_index_path, ratio, x_train_mean=x_train_mean)

    else:
        candidate_data, ood_candidate_index, id_candidate_index = combine_train_and_new(x_train[split_len:], 'train', data_type, data_path, ratio, x_train_mean=x_train_mean)
        candidate_label = y_train[split_len:]
        new_test_data, ood_test_index, id_test_index = combine_train_and_new(x_test[:10000], 'test', data_type, test_data_path, ratio, x_train_mean=x_train_mean)
        new_test_y = y_test[:10000]

    # Entropy
    if metric == 0:
        select_index = entropy_selection(model, candidate_data, select_size)
    # DeepGini
    elif metric == 1:
        select_index = deepgini_selection(model, candidate_data, select_size)
    # MCP
    elif metric == 2:
        select_index = MCP_selection(model, candidate_data, select_size)
    # Random
    elif metric == 3:
        select_index = random_selection(model, candidate_data, select_size)
    # CES
    elif metric == 4:
        CES_index = CES.conditional_sample(model, candidate_data, select_size)
        select_index = CES.select_from_index(select_size, CES_index)
    # DSA
    elif metric == 5:
        DSA_index = fetch_dsa(model, x_train, candidate_data, 'fgsm', dsa_layer, args)
        select_index = CES.select_from_large(select_size, DSA_index)

    # All data
    elif metric == 6:
        select_index = np.arange(len(candidate_data))

    # LSA
    elif metric == 7:
        LSA_index = fetch_lsa(model, x_train, candidate_data, 'fgsm', dsa_layer, args)
        select_index = CES.select_from_large(select_size, LSA_index)

    # np.save("data_for_analysis/deepgini_index_bad.npy", select_index)
    # concatenate

    x_train_final = np.concatenate((candidate_data[select_index], x_train[:split_len]))
    y_train_final = np.concatenate((candidate_label[select_index], y_train[:split_len]))

    new_evl_data = np.concatenate((x_test, new_test_data))
    new_evl_label = np.concatenate((y_test, new_test_y))


    checkpoint = ModelCheckpoint(model_save_path,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 period=1)

    his = model.fit(x_train_final,
                    y_train_final,
                    validation_data=(new_evl_data, new_evl_label),
                    batch_size=batch_size,
                    shuffle=True,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[checkpoint]
                    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="fashion_mnist_5w")
    parser.add_argument(
        "--is_classification",
        "-is_classification",
        help="Is classification task",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--save_path", "-save_path", help="Save path", type=str, default="for_dsa/"
    )
    parser.add_argument(
        "--num_classes",
        "-num_classes",
        help="The number of classes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--ex_ls", "-ex_ls", help="exp set", type=int, default=0
    )
    parser.add_argument(
        "--model_type", "-model_type", help="model_type", type=str, default='lenet5'
    )
    parser.add_argument(
        "--var_threshold", "-var_threshold", help="var_threshold", type=float, default=0.00001
    )
    args = parser.parse_args()
    print(args)
    ex_data = args.d
    model_type = args.model_type
    select_size_list = [100, 300, 500, 1000]
    if args.ex_ls == 0:
        data_styles = ['rotation', 'shear']
    elif args.ex_ls == 1:
        data_styles = ['translation', 'scale']
    elif args.ex_ls == 2:
        data_styles = ['brightness', 'contrast']
    else:
        data_styles = ['pgd', 'fgsm']

    ratios = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

    if ex_data == 'cifar10_4w':
        data_folder = "data/cifar10/"
        index_folder = "data/cifar10/resnet20/"
        for data_style in data_styles:
            index_path = data_folder + model_type + '_' + data_style + '_y'
            if data_style not in ['fgsm', 'pgd']:
                data_path = data_folder + data_style + '_new.npy'
                test_data_path = data_folder + data_style + '_test_new.npy'
            else:
                data_path = data_folder + model_type + '_' + data_style + '_x.npy'
                test_data_path = data_folder + model_type + '_' + data_style + '_x_test.npy'

            for select_size in select_size_list:
                for metric in range(7, 8):
                    for ratio in ratios:
                        if model_type == 'nin':
                            model = keras.models.load_model("models/Cifar10_NiN/NiN-4w.h5")
                        elif model_type == 'resnet20':
                            model = keras.models.load_model("models/Cifar10_ReNet20/ResNet20_4w.h5")
                        fin_tune_lenet5(model, 'cifar10', data_style, data_path, index_path, ratio, select_size, metric,
                                                                test_data_path, args)

    elif ex_data == 'mnist_5w':

        data_folder = "data/mnist/"

        for data_style in data_styles:
            index_path = data_folder + model_type + '_' + data_style + '_y'
            if data_style not in ['fgsm', 'pgd']:
                data_path = data_folder + data_style + '.npy'
                test_data_path = data_folder + data_style + '_test.npy'
            else:
                data_path = data_folder + model_type + '_' + data_style + '_x.npy'
                test_data_path = data_folder + model_type + '_' + data_style + '_x_test.npy'
            print("data path: ", data_path)
            print("index path: ", test_data_path)
            for select_size in select_size_list:
                for metric in range(7, 8):
                    for ratio in ratios:
                        if model_type == 'lenet1':
                            model = keras.models.load_model("models/MNIST-Lenet1/Lenet-1-5w.h5")
                        elif model_type == 'lenet5':
                            model = keras.models.load_model("models/MNIST-Lenet5/Lenet-5-5w.h5")
                        fin_tune_lenet5(model, 'mnist', data_style, data_path, index_path, ratio, select_size, metric,
                                                                test_data_path, args)

    elif ex_data == 'fashion_mnist_5w':
        data_folder = "data/fashion_mnist/"
        for data_style in data_styles:
            index_path = data_folder + model_type + '_' + data_style + '_y'
            if data_style not in ['fgsm', 'pgd']:
                data_path = data_folder + data_style + '.npy'
                test_data_path = data_folder + data_style + '_test.npy'
            else:
                data_path = data_folder + model_type + '_' + data_style + '_x.npy'
                test_data_path = data_folder + model_type + '_' + data_style + '_x_test.npy'
            print("data path: ", data_path)
            print("index path: ", test_data_path)
            for select_size in select_size_list:
                for metric in range(7, 8):
                    for ratio in ratios:
                        if model_type == 'lenet1':
                            model = keras.models.load_model("models/Fashion_MNIST_Lenet1/fashion_lenet1_5w.h5")
                        elif model_type == 'lenet5':
                            model = keras.models.load_model("models/Fashion_MNIST_Lenet5/model_fashion_5w.h5")
                        fin_tune_lenet5(model, 'fashion_mnist', data_style, data_path, index_path, ratio, select_size, metric,
                                                                test_data_path, args)
