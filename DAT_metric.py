from transform_retrain import *
from utils.oe_utils import *
import joblib
from keras.optimizers import Adam, SGD
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint

# id uncertain, ood random + label balance
def DAT(model, oe_model, lr_model, data_type, data_style, data_path, index_path, ratio, select_size, metric,
                             test_data_path, model_type, args):
    if data_type == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        batch_size = 256
        model_save_path = "models/mnist_lenet5_try_" + data_style + model_type + '.h5'
        x_train_mean = None
        epochs = 5
        para_here = 0.01
        para_there = 0.01
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
        model_save_path = "models/fashion_lenet5_try_" + data_style + model_type + '.h5'
        batch_size = 64
        x_train_mean = None
        epochs = 5
        para_here = 0.1
        para_there = 0.1
    elif data_type == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32') / 255
        x_test = x_test.astype('float32') / 255
        x_train_mean = np.mean(x_train[:40000], axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        model_save_path = "models/cifar_try" + data_style + model_type + '.h5'
        batch_size = 128
        epochs = 10
        para_here = 0.3
        para_there = 0.3

    split_len = len(x_train[:-10000])

    if data_style == 'fgsm' or data_style == 'pgd':
        train_index_path = index_path + '.npy'
        print(index_path)
        print(train_index_path)
        candidate_data, candidate_label, ood_candidate_index, id_candidate_index = combine_train_and_new_adv(x_train[split_len:], y_train[split_len:], data_type, data_path, train_index_path, ratio, x_train_mean=x_train_mean)
        test_index_path = index_path + '_test.npy'
        new_test_data, new_test_y, ood_test_index, id_test_index = combine_train_and_new_adv(
            x_test[:10000], y_test[:10000], data_type, test_data_path, test_index_path, ratio,
            x_train_mean=x_train_mean)

    else:
        candidate_data, ood_candidate_index, id_candidate_index = combine_train_and_new(x_train[split_len:], 'train', data_type, data_path, ratio, x_train_mean=x_train_mean)
        candidate_label = y_train[split_len:]
        new_test_data, ood_test_index, id_test_index = combine_train_and_new(x_test[:10000], 'test', data_type, test_data_path, ratio, x_train_mean=x_train_mean)
        new_test_y = y_test[:10000]

    # print(candidate_data.shape)

    # step 1: detect OOD data
    logit_layer = K.function(inputs=oe_model.input, outputs=oe_model.layers[-2].output)
    _, _scores, _ = calculate_oe_score(logit_layer, candidate_data, use_xent=True)
    # distribution score
    y_pred = lr_model.predict_proba(_scores.reshape(-1, 1))
    y_pred_order = np.argsort(y_pred[:, 1])

    # id data selection
    id_num = len(np.where(y_pred[:, 1] < para_here)[0])
    ood_num = len(y_pred[:, 1]) - id_num
    id_part_index = y_pred_order[:id_num]
    ood_part_index = y_pred_order[id_num:]

    id_select_num = int(select_size * id_num / len(y_pred[:, 1]))
    ood_select_num = select_size - id_select_num

    # Decide how many ID and OOD data we need
    if id_select_num > ood_select_num:
        flag_num = id_select_num
        id_select_num = ood_select_num
        ood_select_num = flag_num
    if id_select_num > int(para_there * select_size):
        id_select_num = int(para_there * select_size)
        ood_select_num = select_size - id_select_num

    # if id_select_num < 1:
    #     id_select_num = 1
    print("id num: ", id_select_num)
    print("ood num: ", ood_select_num)
    # print(id_part_index)
    # id data selection
    if len(id_part_index) > 0:
        if metric == 1:
            # deepgini
            id_select_part = deepgini_selection(model, candidate_data[id_part_index], id_select_num)
            # Margin
            # id_select_part = margin_selection(model, candidate_data[id_part_index], id_select_num)
            id_select_index = id_part_index[id_select_part]
        elif metric == 0:
            id_select_part = entropy_selection(model, candidate_data[id_part_index], id_select_num)
            id_select_index = id_part_index[id_select_part]
    else:
        id_select_index = []
    # ood data selection

    # top 1 label
    candidate_prediction_label = np.argmax(model.predict(candidate_data), axis=1)
    reference_prediction_label = np.argmax(model.predict(new_test_data), axis=1)
    # print(reference_prediction_label)
    reference_labels = []

    for i in range(0, 10):
        label_num = len(np.where(reference_prediction_label == i)[0])
        # print("label {}, num {}".format(i, label_num))
        reference_labels.append(label_num)
    reference_labels = np.asarray(reference_labels)
    s_ratio = len(candidate_data) / select_size
    reference_labels = reference_labels / s_ratio

    label_list = []
    index_list = []
    for _ in range(1000):
        ood_select_index = np.random.choice(ood_part_index, ood_select_num, replace=False)

        this_labels = candidate_prediction_label[ood_select_index]
        single_labels = []
        for i in range(0, 10):
            label_num = len(np.where(this_labels == i)[0])
            # print("label {}, num {}".format(i, label_num))

            single_labels.append(label_num)
        index_list.append(ood_select_index)
        label_list.append(single_labels)

    index_list = np.asarray(index_list)
    label_list = np.asarray(label_list)

    # compare to test label
    label_minus = np.abs(label_list - reference_labels)
    var_list = np.sum(label_minus, axis=1)
    var_list_order = np.argsort(var_list)

    ood_select_index = index_list[var_list_order[0]]
    select_index = np.concatenate((id_select_index, ood_select_index))
    select_index = np.asarray(select_index)

    select_index = select_index.astype('int')
    # print(select_index)
    select_index = select_index.reshape(-1, )
    selected_data = candidate_data[select_index]
    selected_label = candidate_label[select_index]

    x_train_final = np.concatenate((selected_data ,x_train[:split_len]))
    y_train_final = np.concatenate((selected_label, y_train[:split_len]))

    # calculate some return score
    new_val_data = np.concatenate((x_test, new_test_data))
    new_val_label = np.concatenate((y_test, new_test_y))

    # model_save_path = "models/Lenet-5-try.h5"
    checkpoint = ModelCheckpoint(model_save_path,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 period=1)

    his = model.fit(x_train_final,
                    y_train_final,
                    validation_data=(new_val_data, new_val_label),
                    batch_size=batch_size,
                    shuffle=True,
                    epochs=epochs,
                    verbose=1,
                    callbacks=[checkpoint])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--d", "-d", help="Dataset", type=str, default="mnist")
    parser.add_argument("--m", "-m", help="Model", type=str, default="fashion_lenet5_5w")
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
    args = parser.parse_args()
    data_type = args.d
    model_type = args.m

    if args.ex_ls == 0:
        data_styles = ['rotation', 'shear']
    elif args.ex_ls == 1:
        data_styles = ['translation', 'scale']
    elif args.ex_ls == 2:
        data_styles = ['brightness', 'contrast']
    # ratios = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    select_size_list = [100, 300, 500, 1000]
    ratios = [0, 0.1, 0.2, 0.3]
    # select_size_list = [1000]
    if model_type == 'lenet5_5w':
        model_lr = joblib.load("models/MNIST-Lenet5/mnist_Lenet5_lr.model")
        model_oe = load_model("models/MNIST-Lenet5/mnist_Lenet5_mnist-5w-oe.h5", compile=False)
        model_oe.compile(loss=my_ood_loss,
                         optimizer='adam',
                         metrics=['accuracy'])
        data_folder = "data/mnist/"
        index_folder = "data/cifar10/resnet20/"
        # original results in results/mnist/my_test
        metric = 1
        for data_style in data_styles:
            index_path = data_folder + model_type + '_' + data_style + '_y'
            if data_style not in ['fgsm', 'pgd']:
                data_path = data_folder + data_style + '.npy'
                test_data_path = data_folder + data_style + '_test.npy'
            else:
                index_path = data_folder + 'lenet5_' + data_style + '_y'
                data_path = data_folder + 'lenet5_' + data_style + '_x.npy'
                test_data_path = data_folder + 'lenet5_' + data_style + '_x_test.npy'

            for select_size in select_size_list:
                for ratio in ratios:
                    print("size: {}, data style: {}, ratio: {}".format(select_size, data_style, ratio))
                    model = keras.models.load_model("models/MNIST-Lenet5/Lenet-5-5w.h5")
                    DAT(model, model_oe, model_lr, 'mnist', data_style, data_path, index_path, ratio, select_size, metric,
                                                                test_data_path, model_type, args)

    elif model_type == 'lenet1_5w':
        model_lr = joblib.load("models/MNIST-Lenet1/mnist_Lenet1_lr.model")
        model_oe = load_model("models/MNIST-Lenet1/mnist_Lenet1_mnist-5w-oe.h5", compile=False)
        model_oe.compile(loss=my_ood_loss,
                         optimizer='adam',
                         metrics=['accuracy'])

        data_folder = "data/mnist/"
        metric = 1
        for data_style in data_styles:
            for select_size in select_size_list:
                index_path = data_folder + 'lenet1_' + data_style + '_y'
                if data_style not in ['fgsm', 'pgd']:
                    data_path = data_folder + data_style + '.npy'
                    test_data_path = data_folder + data_style + '_test.npy'
                else:
                    data_path = data_folder + 'lenet1_' + data_style + '_x.npy'
                    test_data_path = data_folder + 'lenet1_' + data_style + '_x_test.npy'
                # print(data_path)
                for ratio in ratios:
                    print("size: {}, data style: {}, ratio: {}".format(select_size, data_style, ratio))

                    model = keras.models.load_model("models/MNIST-Lenet1/Lenet-1-5w.h5")
                    DAT(model, model_oe, model_lr, 'mnist', data_style, data_path, index_path, ratio, select_size, metric,
                                                                test_data_path, model_type, args)

    elif model_type == 'fashion_lenet5_5w':
        model_lr = joblib.load("models/Fashion_MNIST_Lenet5/fashion_mnist_lenet5_lr.model")
        model_oe = load_model("models/Fashion_MNIST_Lenet5/lenet5_fashion-5w-oe.h5", compile=False)
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model_oe.compile(loss=my_ood_loss,
                         optimizer=sgd,
                         metrics=['accuracy'])
        data_folder = "data/fashion_mnist/"
        metric = 1
        for data_style in data_styles:
            for select_size in select_size_list:
                index_path = data_folder + model_type + '_' + data_style + '_y'
                if data_style not in ['fgsm', 'pgd']:
                    data_path = data_folder + data_style + '.npy'
                    test_data_path = data_folder + data_style + '_test.npy'
                else:
                    index_path = data_folder + 'lenet5_' + data_style + '_y'
                    data_path = data_folder + 'lenet5_' + data_style + '_x.npy'
                    test_data_path = data_folder + 'lenet5_' + data_style + '_x_test.npy'

                for ratio in ratios:
                    print("size: {}, data style: {}, ratio: {}".format(select_size, data_style, ratio))

                    model = keras.models.load_model("models/Fashion_MNIST_Lenet5/model_fashion_5w.h5")
                    DAT(model, model_oe, model_lr, 'fashion_mnist', data_style, data_path, index_path, ratio, select_size, metric,
                                                                test_data_path, model_type, args)

    elif model_type == 'fashion_lenet1_5w':
        model_lr = joblib.load("models/Fashion_MNIST_Lenet1/fashion_mnist_lenet1_lr.model")
        model_oe = load_model("models/Fashion_MNIST_Lenet1/lenet1_fashion-5w-oe.h5", compile=False)
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model_oe.compile(loss=my_ood_loss,
                         optimizer=sgd,
                         metrics=['accuracy'])
        data_folder = "data/fashion_mnist/"
        metric = 1
        for data_style in data_styles:
            for select_size in select_size_list:
                index_path = data_folder + model_type + '_' + data_style + '_y'
                if data_style not in ['fgsm', 'pgd']:
                    data_path = data_folder + data_style + '.npy'
                    test_data_path = data_folder + data_style + '_test.npy'
                else:
                    index_path = data_folder + 'lenet1_' + data_style + '_y'
                    data_path = data_folder + 'lenet1_' + data_style + '_x.npy'
                    test_data_path = data_folder + 'lenet1_' + data_style + '_x_test.npy'

                for ratio in ratios:
                    print("size: {}, data style: {}, ratio: {}".format(select_size, data_style, ratio))
                    model = keras.models.load_model("models/Fashion_MNIST_Lenet1/fashion_lenet1_5w.h5")
                    DAT(model, model_oe, model_lr, 'fashion_mnist', data_style, data_path, index_path, ratio, select_size, metric,
                                                                test_data_path, model_type, args)

    elif model_type == 'cifar10_resnet20_4w':
        model_lr = joblib.load("models/Cifar10_ReNet20/ResNet20_lr.model")
        model_oe = load_model("models/Cifar10_ReNet20/ResNet20_cifar10_4w-oe.h5", compile=False)
        model_oe.compile(loss=my_ood_loss, optimizer=Adam(lr=1e-3), metrics=['accuracy'])
        # data_styles = ['pgd', 'fgsm']
        data_folder = "data/cifar10/"
        metric = 1
        for data_style in data_styles:
            index_path = data_folder + model_type + '_' + data_style + '_y'
            for select_size in select_size_list:
                if data_style not in ['fgsm', 'pgd']:
                    data_path = data_folder + data_style + '_new.npy'
                    test_data_path = data_folder + data_style + '_test_new.npy'
                else:
                    index_path = data_folder + 'resnet20_' + data_style + '_y'
                    data_path = data_folder + 'resnet20_' + data_style + '_x.npy'
                    test_data_path = data_folder + 'resnet20_' + data_style + '_x_test.npy'

                for ratio in ratios:
                    print("size: {}, data style: {}, ratio: {}".format(select_size, data_style, ratio))
                    model = keras.models.load_model("models/Cifar10_ReNet20/ResNet20_4w.h5")
                    DAT(model, model_oe, model_lr, 'cifar10', data_style, data_path, index_path, ratio, select_size, metric,
                                                                test_data_path, model_type, args)

    elif model_type == 'cifar10_nin_4w':
        model_lr = joblib.load("models/Cifar10_NiN/NiN_lr.model")
        model_oe = load_model("models/Cifar10_NiN/NiN_cifar10_4w-oe.h5", compile=False)
        model_oe.compile(loss=my_ood_loss, optimizer=Adam(lr=1e-3), metrics=['accuracy'])
        data_folder = "data/cifar10/"
        # data_styles = ['pgd', 'fgsm']
        metric = 1
        for data_style in data_styles:
            index_path = data_folder + model_type + '_' + data_style + '_y'
            for select_size in select_size_list:
                if data_style not in ['fgsm', 'pgd']:
                    data_path = data_folder + data_style + '_new.npy'
                    test_data_path = data_folder + data_style + '_test_new.npy'
                else:
                    index_path = data_folder + 'nin_' + data_style + '_y'
                    data_path = data_folder + 'nin_' + data_style + '_x.npy'
                    test_data_path = data_folder + 'nin_' + data_style + '_x_test.npy'

                for ratio in ratios:
                    print("size: {}, data style: {}, ratio: {}".format(select_size, data_style, ratio))
                    model = keras.models.load_model("models/Cifar10_NiN/NiN-4w.h5")
                    DAT(model, model_oe, model_lr, 'cifar10', data_style,
                                                                 data_path, index_path, ratio, select_size, metric,
                                                                 test_data_path, model_type, args)

