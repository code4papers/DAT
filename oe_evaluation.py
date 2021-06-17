from utils.oe_utils import *
from utils.plot_functions import *
from keras.models import load_model
from keras.datasets import mnist, cifar10
import keras
from sklearn.linear_model import LogisticRegressionCV
import joblib
from utils import mnist_reader
from utils import SVHN_DatasetUtil
from wilds_retrain import *


def oe_lr_model_generation(oe_model_path, data_type, ood_data_path, save_path):
    model = load_model(oe_model_path, compile=False)
    model.compile(loss=my_ood_loss,
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    if data_type == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        y_test = keras.utils.to_categorical(y_test, 10)
        y_train = keras.utils.to_categorical(y_train, 10)
        x_train = x_train[:50000]
        ood_data = np.load(ood_data_path)
        ood_data = ood_data.astype('float32') / 255
        ood_data = ood_data.reshape(-1, 28, 28, 1)

    elif data_type == 'fashion_mnist':
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
        x_train = x_train[:50000]

        ood_data = np.load(ood_data_path)
        ood_data = ood_data.astype('float32') / 255
        ood_data = ood_data.reshape(-1, 28, 28, 1)

    if data_type == 'svhn':
        (x_train, y_train), (x_test, y_test) = SVHN_DatasetUtil.load_data()  # 32*32
        x_train = x_train[:-10000]
        y_train = y_train[:-10000]
        x_test = x_test[:10000]
        y_test = y_test[:10000]
        x_train = x_train[:63000]
        y_train = y_train[:63000]
        ood_data = np.load(ood_data_path)
        ood_data = ood_data.astype('float32') / 255
        ood_data = ood_data[:63000]

    elif data_type == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train_mean = np.mean(x_train[:40000], axis=0)
        x_train -= x_train_mean
        x_train = x_train[:40000]
        ood_data = np.load(ood_data_path)
        print(ood_data.shape)
        ood_data = ood_data.astype('float32') / 255
        ood_data -= x_train_mean

    elif data_type == 'wilds':
        ood_file_folder = "data/iwildcam_v2.0/OE_data/*.JPEG"
        id_file_folder = "data/iwildcam_v2.0/OE_ID_data/*.JPEG"
        ood_data, ood_label = image_read(ood_file_folder)
        id_data, id_label = image_read(id_file_folder)
        ood_data = ood_data
        x_train = id_data
        ood_label = to_categorical(ood_label, 182)
        id_label = to_categorical(id_label, 182)

    elif data_type == 'fmow':
        ood_data, ood_label = idwild_test_from_index("data/fmow/oe_ood_index.npy", 'ood_val')
        id_data, id_label = idwild_test_from_index("data/fmow/oe_id_index.npy", 'train')
        ood_data = ood_data
        x_train = id_data

    partition = len(ood_data)
    train_num = partition - 1000
    logit_layer = K.function(inputs=model.input, outputs=model.layers[-2].output)
    split_len = len(x_train) / 50
    for i in range(50):
        if i == 0:
            _, id_scores, _ = calculate_oe_score(logit_layer, x_train[int(i * split_len): int((i+1) * split_len)], use_xent=True)
            _, ood_scores, _ = calculate_oe_score(logit_layer, ood_data[int(i * split_len): int((i+1) * split_len)], use_xent=True)
        else:
            _, id_scores_s, _ = calculate_oe_score(logit_layer, x_train[int(i * split_len): int((i+1) * split_len)],
                                                 use_xent=True)
            _, ood_scores_s, _ = calculate_oe_score(logit_layer, ood_data[int(i * split_len): int((i+1) * split_len)],
                                                  use_xent=True)
            id_scores = np.concatenate((id_scores, id_scores_s))
            ood_scores = np.concatenate((ood_scores, ood_scores_s))
    id_label = "id_train"
    ood_label = "ood_valid"
    print(id_scores)
    print(ood_scores)

    scores, labels = merge_and_generate_labels(ood_scores, id_scores)
    results = get_metric_scores(labels, scores, tpr_level=0.95)
    print("\tAUROC:{auroc:6.2f}\tAUPR:{aupr:6.2f}\tTNR:{tnr:6.2f}".format(
        auroc=results['AUROC'] * 100.,
        aupr=results['AUPR'] * 100.,
        tnr=results['TNR'] * 100.,
    ))
    print("threshold: ", results['TNR_threshold'])
    X_train, Y_train, X_test, Y_test = block_split(scores, labels, train_num=train_num, partition=partition)

    lr = LogisticRegressionCV(n_jobs=-1, cv=3, max_iter=5000).fit(X_train, Y_train)
    y_pred = lr.predict_proba(scores)[:, 1]

    results = get_metric_scores(labels, y_pred, tpr_level=0.95)
    print("\tAUROC:{auroc:6.2f}\tAUPR:{aupr:6.2f}\tTNR:{tnr:6.2f}".format(
        auroc=results['AUROC'] * 100.,
        aupr=results['AUPR'] * 100.,
        tnr=results['TNR'] * 100.,
    ))
    joblib.dump(lr, save_path)


if __name__ == '__main__':
    # Cifar10
    # oe_model_path = "models/Cifar10_ReNet20/ResNet20_4w.h5"
    # ood_path = "data/cifar10/resnet20_ood_combine.npy"
    # data_type = 'cifar10'
    # save_path = "models/Cifar10_ReNet20/ResNet20_lr.model"
    # oe_lr_model_generation(oe_model_path, data_type, ood_path, save_path)

    # oe_model_path = "models/Cifar10_NiN/NiN-4w.h5"
    # ood_path = "data/cifar10/nin_ood_combine.npy"
    # data_type = 'cifar10'
    # save_path = "models/Cifar10_NiN/NiN_lr.model"
    # oe_lr_model_generation(oe_model_path, data_type, ood_path, save_path)

    # Fashion Mnist
    # oe_model_path = "models/Fashion_MNIST_Lenet5/lenet5_fashion-5w-oe.h5"
    # ood_path = "data/fashion_mnist/lenet5_ood_combine.npy"
    # data_type = 'fashion_mnist'
    # save_path = "models/Fashion_MNIST_Lenet5/fashion_mnist_lenet5_lr.model"
    # oe_lr_model_generation(oe_model_path, data_type, ood_path, save_path)

    # oe_model_path = "models/Fashion_MNIST_Lenet1/lenet1_fashion-5w-oe.h5"
    # ood_path = "data/fashion_mnist/lenet1_ood_combine.npy"
    # data_type = 'fashion_mnist'
    # save_path = "models/Fashion_MNIST_Lenet5/fashion_mnist_lenet1_lr.model"
    # oe_lr_model_generation(oe_model_path, data_type, ood_path, save_path)

    # SVHN
    # oe_model_path = "models/SVHN-Lenet5/svhn-Lenet5-5w-oe.h5"
    # ood_path = "data/svhn_new/lenet5_ood_combine.npy"
    # data_type = 'svhn'
    # save_path = "models/SVHN-Lenet5/svhn-Lenet5-lr.model"
    # oe_lr_model_generation(oe_model_path, data_type, ood_path, save_path)

    # MNIST

    # oe_model_path = "models/MNIST-Lenet5/mnist_Lenet5_mnist-5w-oe.h5"
    # ood_path = "data/mnist/lenet5_ood_combine.npy"
    # data_type = 'mnist'
    # save_path = "models/MNIST-Lenet5/mnist_Lenet5_lr.model"
    # oe_lr_model_generation(oe_model_path, data_type, ood_path, save_path)
    # oe_model_path = "models/MNIST-Lenet1/mnist_Lenet1_mnist-5w-oe.h5"
    # ood_path = "data/mnist/lenet1_ood_combine.npy"
    # data_type = 'mnist'
    # save_path = "models/MNIST-Lenet1/mnist_Lenet1_lr.model"
    # oe_lr_model_generation(oe_model_path, data_type, ood_path, save_path)

    #wilds
    # oe_model_path = "models/iwildcam/ResNet50_oe.h5"
    # ood_path = "data/fashion_mnist/lenet5_ood_combine.npy"
    # data_type = 'wilds'
    # save_path = "models/iwildcam/ResNet50_lr.model"
    # oe_lr_model_generation(oe_model_path, data_type, ood_path, save_path)

    # fmow
    oe_model_path = "models/FMOW/DenseNet121_oe.h5"
    ood_path = "data/fashion_mnist/lenet5_ood_combine.npy"
    data_type = 'fmow'
    save_path = "models/FMOW/DenseNet121_lr.model"
    oe_lr_model_generation(oe_model_path, data_type, ood_path, save_path)
