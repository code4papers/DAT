from keras.datasets import mnist, cifar10
from keras.utils import to_categorical
import numpy as np
from keras.models import load_model
from utils import mnist_reader
from utils import SVHN_DatasetUtil


def combine_test_and_new():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    x_generated = np.load("data/mnist/shear_test.npy")
    print(x_generated.shape)
    x_generated = x_generated.astype('float32') / 255
    x_generated = x_generated.reshape(-1, 28, 28, 1)
    x_generated_index = np.arange(len(x_generated))
    x_generated_selected_index = np.random.choice(x_generated_index, 2000, replace=False)
    x_generated_remain_index = np.delete(x_generated_index, x_generated_selected_index)
    x_candidate = np.concatenate((x_test[x_generated_remain_index], x_generated[x_generated_selected_index]))
    y_candidate = np.concatenate((y_test[x_generated_remain_index], y_test[x_generated_selected_index]))
    print(len(x_generated_selected_index))
    print(len(x_generated_remain_index))
    print(x_candidate.shape)
    np.save("data/mnist/shear_test_selected_index.npy", x_generated_selected_index)


def combine_id_and_ood(id_data, id_label, ood_data, ood_label, ratio, save_path):
    train_len = 60000
    id_num = int(train_len * ratio)
    ood_num = train_len - id_num
    all_ood_index = np.arange(len(ood_data))
    all_id_index = np.arange(len(id_data))
    ood_select_index = np.random.choice(all_ood_index, ood_num, replace=False)
    id_select_index = np.random.choice(all_id_index, id_num, replace=False)

    candidate_data = np.concatenate((ood_data[ood_select_index], id_data[id_select_index]))
    condidate_label = np.concatenate((ood_label[ood_select_index], id_label[id_select_index]))
    print(candidate_data.shape)
    np.save(save_path + str(ratio) + "_x.npy", candidate_data)
    np.save(save_path + str(ratio) + "_y.npy", condidate_label)


def combine_train_and_new(x_id, tr_te, data_type, data_path, ratio, x_train_mean=None):
    x_ood = np.load(data_path)
    if tr_te == 'train':
        x_ood = x_ood[-10000:]
    else:
        x_ood = x_ood[:10000]
    # print(x_ood[0])
    data_len = len(x_id)
    replace_num = int(ratio * data_len)
    all_index = np.arange(data_len)
    if data_type == 'mnist':
        x_ood = x_ood.astype('float32') / 255
        x_ood = x_ood.reshape(-1, 28, 28, 1)
    elif data_type == 'fashion_mnist':
        x_ood = x_ood.astype('float32') / 255
        x_ood = x_ood.reshape(-1, 28, 28, 1)
    elif data_type == 'cifar10':
        x_ood = x_ood.astype('float32') / 255
        x_ood -= x_train_mean
    else:
        x_ood = x_ood.astype('float32') / 255
    x_ood[:replace_num] = x_id[:replace_num]
    ood_index = all_index[replace_num:]
    id_index = all_index[:replace_num]
    return x_ood, ood_index, id_index


def combine_train_and_new_adv(x_id, y_id, data_type, data_path, index_path, ratio, x_train_mean=None):

    x_ood = np.load(data_path)
    y_ood = np.load(index_path)

    x_ood = x_ood[-10000:]
    y_ood = y_ood[-10000:]
    y_ood = to_categorical(y_ood, 10)
    data_len = len(y_ood)
    # print(data_len)
    if data_type == 'mnist':
        x_ood = x_ood.astype('float32') / 255
        x_ood = x_ood.reshape(-1, 28, 28, 1)
    elif data_type == 'fashion_mnist':
        x_ood = x_ood.astype('float32') / 255
        x_ood = x_ood.reshape(-1, 28, 28, 1)

    elif data_type == 'cifar10':
        x_ood = x_ood.astype('float32') / 255
        x_ood -= x_train_mean
    else:
        x_ood = x_ood.astype('float32') / 255

    replace_num = int(ratio * data_len)
    # print(replace_num)
    # print(x_id.shape)
    # print(y_id.shape)
    x_ood[:replace_num] = x_id[:replace_num]
    y_ood[:replace_num] = y_id[:replace_num]
    # print(x_ood.shape)
    all_index = np.arange(data_len)
    ood_index = all_index[replace_num:]
    id_index = all_index[:replace_num]
    # print(ood_index)
    # print(id_index)
    # combined, id, ood
    return x_ood, y_ood, ood_index, id_index


def combine_oe_ood_data(model, paths, data_type, save_path):
    if data_type == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        y_train = y_train[:50000]
    elif data_type == 'cifar10':
        (x_train, y_train), (_, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_train_mean = np.mean(x_train[:40000], axis=0)
        y_train = y_train[:40000]
    elif data_type == 'fashion_mnist':
        path = 'data/fashion_mnist_ori/'
        x_train, y_train = mnist_reader.load_mnist(path, kind='train')
        y_train = y_train[:50000]
    elif data_type == 'svhn':
        (X_train, Y_train), (X_test, Y_test) = SVHN_DatasetUtil.load_data()
        y_train = Y_train[:-10000]
        y_train = np.argmax(y_train, axis=1)
    need_num = len(y_train)
    len_paths = len(paths)
    number_per_folder = int(need_num / len_paths)
    print("number per folder: ", number_per_folder)
    for i in range(len_paths - 2):
        path = paths[i]
        print(path)
        x_candidate = np.load(path)

        if data_type == 'cifar10':
            x_candidate = x_candidate[:-10000]
            x_copy = x_candidate.copy()
            x_candidate = x_candidate.astype('float32') / 255

            x_candidate -= x_train_mean
        elif data_type == 'mnist' or data_type == 'fashion_mnist':
            x_candidate = x_candidate[:-10000]
            x_candidate = x_candidate.reshape(-1, 28, 28, 1)
            x_copy = x_candidate.copy()
            x_candidate = x_candidate.astype('float32') / 255


        else:
            x_candidate = x_candidate[:-10000]
            x_candidate = x_candidate.reshape(-1, 32, 32, 3)
            x_copy = x_candidate.copy()
            x_candidate = x_candidate.astype('float32') / 255


        prediction = model.predict(x_candidate)
        labels = np.argmax(prediction, axis=1)
        wrong_index = np.where(labels != y_train)[0]
        if number_per_folder < len(wrong_index):
            selected_index = np.random.choice(wrong_index, number_per_folder, replace=False)
        else:
            selected_index = wrong_index
        print("select num: ", len(selected_index))

        x_selected = x_copy[selected_index]
        if i == 0:
            x_final_data = x_selected
        else:
            x_final_data = np.concatenate((x_final_data, x_selected))

        del x_candidate
        del x_copy
        del x_selected

    attack_1_data = np.load(paths[-2])
    if data_type == 'mnist' or data_type == 'fashion_mnist':
        attack_1_data = attack_1_data.reshape(-1, 28, 28, 1)
    else:
        attack_1_data = attack_1_data.reshape(-1, 32, 32, 3)
    attack_1_index = np.arange(len(attack_1_data))
    selected_index = np.random.choice(attack_1_index, number_per_folder, replace=False)
    x_selected = attack_1_data[selected_index]
    print(x_selected.shape)
    print(x_final_data.shape)
    x_final_data = np.concatenate((x_final_data, x_selected))

    number_per_folder = need_num - len(x_final_data)
    attack_2_data = np.load(paths[-1])
    if data_type == 'mnist' or data_type == 'fashion_mnist':
        attack_2_data = attack_2_data.reshape(-1, 28, 28, 1)
    else:
        attack_2_data = attack_2_data.reshape(-1, 32, 32, 3)
    attack_2_index = np.arange(len(attack_2_data))
    selected_index = np.random.choice(attack_2_index, number_per_folder, replace=False)
    x_selected = attack_2_data[selected_index]
    x_final_data = np.concatenate((x_final_data, x_selected))
    print(x_final_data.shape)
    np.save(save_path, x_final_data)
    return x_final_data


if __name__ == '__main__':
    # generate ood data for lr model mnist
    model = load_model("models/MNIST-Lenet5/Lenet-5-5w.h5")
    paths = ["data/mnist/rotation.npy", "data/mnist/brightness.npy", "data/mnist/contrast.npy", "data/mnist/scale.npy",
             "data/mnist/shear.npy", "data/mnist/translation.npy", "data/mnist/lenet5_pgd_x.npy", "data/mnist/lenet5_fgsm_x.npy"]
    data_type = 'mnist'
    save_path = "data/mnist/lenet5_ood_combine.npy"
    combine_oe_ood_data(model, paths, data_type, save_path)

    model = load_model("models/MNIST-Lenet1/Lenet-1-5w.h5")
    paths = ["data/mnist/rotation.npy", "data/mnist/brightness.npy", "data/mnist/contrast.npy", "data/mnist/scale.npy",
             "data/mnist/shear.npy", "data/mnist/translation.npy", "data/mnist/lenet1_pgd_x.npy", "data/mnist/lenet1_fgsm_x.npy"
             ]
    data_type = 'mnist'
    save_path = "data/mnist/lenet1_ood_combine.npy"
    combine_oe_ood_data(model, paths, data_type, save_path)

    # generate ood data for lr model fashion mnist
    model = load_model("models/Fashion_MNIST_Lenet1/fashion_lenet1_5w.h5")
    paths = ["data/fashion_mnist/rotation.npy", "data/fashion_mnist/brightness.npy", "data/fashion_mnist/contrast.npy", "data/mnist/scale.npy",
             "data/fashion_mnist/shear.npy", "data/fashion_mnist/translation.npy", "data/fashion_mnist/lenet1_pgd_x.npy", "data/fashion_mnist/lenet1_fgsm_x.npy"
             ]
    data_type = 'fashion_mnist'
    save_path = "data/fashion_mnist/lenet1_ood_combine.npy"
    combine_oe_ood_data(model, paths, data_type, save_path)

    model = load_model("models/Fashion_MNIST_Lenet5/model_fashion_5w.h5")
    paths = ["data/fashion_mnist/rotation.npy", "data/fashion_mnist/brightness.npy", "data/fashion_mnist/contrast.npy",
             "data/mnist/scale.npy",
             "data/fashion_mnist/shear.npy", "data/fashion_mnist/translation.npy", "data/fashion_mnist/lenet5_pgd_x.npy",
             "data/fashion_mnist/lenet5_fgsm_x.npy",
             ]
    data_type = 'fashion_mnist'
    save_path = "data/fashion_mnist/lenet5_ood_combine.npy"
    combine_oe_ood_data(model, paths, data_type, save_path)

    # generate ood data for lr model cifar10
    model = load_model("models/Cifar10_NiN/NiN-4w.h5")
    paths = ["data/cifar10/rotation.npy", "data/cifar10/brightness.npy", "data/cifar10/contrast.npy", "data/cifar10/scale.npy",
             "data/cifar10/shear.npy", "data/cifar10/translation.npy", "data/cifar10/nin_pgd_x.npy", "data/cifar10/nin_fgsm_x.npy"]
    data_type = 'cifar10'
    save_path = "data/cifar10/lenet5_ood_combine.npy"
    combine_oe_ood_data(model, paths, data_type, save_path)

    model = load_model("models/Cifar10_ReNet20/ResNet20_4w.h5")
    paths = ["data/cifar10/rotation.npy", "data/cifar10/brightness.npy", "data/cifar10/contrast.npy",
             "data/cifar10/scale.npy",
             "data/cifar10/shear.npy", "data/cifar10/translation.npy", "data/cifar10/resnet20_pgd_x.npy", "data/cifar10/resnet20_fgsm_x.npy"
             ]
    data_type = 'cifar10'
    save_path = "data/cifar10/resnet20_ood_combine.npy"
    combine_oe_ood_data(model, paths, data_type, save_path)

    # generate ood data for lr model svhn

    model = load_model("models/SVHN-Lenet5/svhn-Lenet5-5w.h5")
    paths = ["data/svhn_new/rotation.npy", "data/svhn_new/brightness.npy", "data/svhn_new/contrast.npy",
             "data/svhn_new/scale.npy",
             "data/svhn_new/shear.npy", "data/svhn_new/translation.npy", "data/svhn_new/lenet5_pgd_x.npy", "data/svhn_new/lenet5_fgsm_x.npy",
             ]
    data_type = 'svhn'
    save_path = "data/svhn_new/lenet5_ood_combine.npy"
    combine_oe_ood_data(model, paths, data_type, save_path)

    model = load_model("models/SVHN-ResNet20/svhn-resnet20-5w.h5")
    paths = ["data/svhn_new/rotation.npy", "data/svhn_new/brightness.npy", "data/svhn_new/contrast.npy",
             "data/svhn_new/scale.npy",
             "data/svhn_new/shear.npy", "data/svhn_new/translation.npy", "data/svhn_new/resnet20_fgsm_x.npy", "data/svhn_new/resnet20_pgd_x.npy"
             ]
    data_type = 'svhn'
    save_path = "data/svhn/resnet20_ood_combine.npy"
    combine_oe_ood_data(model, paths, data_type, save_path)


