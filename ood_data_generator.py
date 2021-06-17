import joblib
from utils.oe_utils import *
from keras.datasets import mnist
from keras.models import load_model
from image_process import *
from keras.optimizers import Adam


def id_ood_split(paths, oe_model, lr_model, save_path):
    (_, y_train), (_, _) = mnist.load_data()

    len_paths = len(paths)
    logit_layer = K.function(inputs=oe_model.input, outputs=oe_model.layers[-2].output)

    for i in range(len_paths):
        path = paths[i]
        x_candidate = np.load(path)
        x_candidate = x_candidate.astype('float32') / 255
        x_candidate = x_candidate.reshape(-1, 28, 28, 1)
        _, _scores, _ = calculate_oe_score(logit_layer, x_candidate.reshape(-1, 28, 28, 1), use_xent=True)
        y_pred = lr_model.predict_proba(_scores.reshape(-1, 1))
        ood_index = np.where(y_pred[:, 1] > 0.75)[0]
        id_index = np.where(y_pred[:, 1] <= 0.75)[0]

        if i == 0:
            ood_save = x_candidate[ood_index]
            id_save = x_candidate[id_index]
            ood_label = y_train[ood_index]
            id_label = y_train[id_index]
        else:
            ood_save = np.concatenate((ood_save, x_candidate[ood_index]))
            id_save = np.concatenate((id_save, x_candidate[id_index]))
            ood_label = np.concatenate((ood_label, y_train[ood_index]))
            id_label = np.concatenate((id_label, y_train[id_index]))
        del x_candidate

    # cw attack
    x_candidate = np.load("data/mnist/cw_inf.npy")
    y_candidate = np.load("data/mnist/cw_inf_label.npy")
    x_candidate = x_candidate.astype('float32') / 255
    x_candidate = x_candidate.reshape(-1, 28, 28, 1)
    _, _scores, _ = calculate_oe_score(logit_layer, x_candidate.reshape(-1, 28, 28, 1), use_xent=True)
    y_pred = lr_model.predict_proba(_scores.reshape(-1, 1))
    ood_index = np.where(y_pred[:, 1] > 0.75)[0]
    id_index = np.where(y_pred[:, 1] <= 0.75)[0]
    ood_save = np.concatenate((ood_save, x_candidate[ood_index]))
    id_save = np.concatenate((id_save, x_candidate[id_index]))
    ood_label = np.concatenate((ood_label, y_candidate[ood_index]))
    id_label = np.concatenate((id_label, y_candidate[id_index]))

    del x_candidate

    # pgd attack
    x_candidate = np.load("data/mnist/pgd_inf.npy")
    y_candidate = np.load("data/mnist/pgd_inf_label.npy")
    x_candidate = x_candidate.astype('float32') / 255
    x_candidate = x_candidate.reshape(-1, 28, 28, 1)
    _, _scores, _ = calculate_oe_score(logit_layer, x_candidate.reshape(-1, 28, 28, 1), use_xent=True)
    y_pred = lr_model.predict_proba(_scores.reshape(-1, 1))
    ood_index = np.where(y_pred[:, 1] > 0.75)[0]
    id_index = np.where(y_pred[:, 1] <= 0.75)[0]
    ood_save = np.concatenate((ood_save, x_candidate[ood_index]))
    id_save = np.concatenate((id_save, x_candidate[id_index]))
    ood_label = np.concatenate((ood_label, y_candidate[ood_index]))
    id_label = np.concatenate((id_label, y_candidate[id_index]))
    del x_candidate
    print("ood data len: ", len(ood_save))
    print("id data len: ", len(id_save))
    np.save(save_path + "total_id_data_v3.npy", id_save)
    np.save(save_path + "total_ood_data_v3.npy", ood_save)
    np.save(save_path + "total_id_label_v3.npy", id_label)
    np.save(save_path + "total_ood_label_v3.npy", ood_label)


def id_ood_split_cifar10(paths, oe_model, lr_model, save_path):
    (x_train, y_train), (_, _) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    len_paths = len(paths)
    logit_layer = K.function(inputs=oe_model.input, outputs=oe_model.layers[-2].output)

    for i in range(len_paths):
        path = paths[i]
        x_candidate = np.load(path)
        x_for_save = x_candidate.copy()
        x_candidate = x_candidate.astype('float32') / 255
        x_candidate -= x_train_mean
        _, _scores, _ = calculate_oe_score(logit_layer, x_candidate.reshape(-1, 32, 32, 3), use_xent=True)
        y_pred = lr_model.predict_proba(_scores.reshape(-1, 1))
        ood_index = np.where(y_pred[:, 1] > 0.8)[0]
        id_index = np.where(y_pred[:, 1] <= 0.8)[0]

        if i == 0:
            ood_save = x_for_save[ood_index]
            id_save = x_for_save[id_index]
            ood_label = y_train[ood_index]
            id_label = y_train[id_index]
        else:
            ood_save = np.concatenate((ood_save, x_for_save[ood_index]))
            id_save = np.concatenate((id_save, x_for_save[id_index]))
            ood_label = np.concatenate((ood_label, y_train[ood_index]))
            id_label = np.concatenate((id_label, y_train[id_index]))
        del x_candidate
        del x_for_save

    # cw attack
    x_candidate = np.load("data/cifar10/cw_inf.npy")
    y_candidate = np.load("data/cifar10/cw_inf_label.npy")
    print(y_train.shape)
    print(y_candidate.shape)
    y_candidate = y_candidate.reshape(-1, 1)
    _, _scores, _ = calculate_oe_score(logit_layer, x_candidate.reshape(-1, 32, 32, 3), use_xent=True)
    y_pred = lr_model.predict_proba(_scores.reshape(-1, 1))
    ood_index = np.where(y_pred[:, 1] > 0.8)[0]
    id_index = np.where(y_pred[:, 1] <= 0.8)[0]
    x_for_save = x_candidate + x_train_mean
    x_for_save *= 255
    x_for_save = x_for_save.astype('int')
    ood_save = np.concatenate((ood_save, x_for_save[ood_index]))
    id_save = np.concatenate((id_save, x_for_save[id_index]))
    ood_label = np.concatenate((ood_label, y_candidate[ood_index]))
    id_label = np.concatenate((id_label, y_candidate[id_index]))

    del x_candidate
    del x_for_save
    # pgd attack
    x_candidate = np.load("data/cifar10/pgd_inf.npy")
    y_candidate = np.load("data/cifar10/pgd_inf_label.npy")
    y_candidate = y_candidate.reshape(-1, 1)
    x_candidate = x_candidate.astype('float32') / 255
    x_for_save = x_candidate + x_train_mean
    x_for_save *= 255
    x_for_save = x_for_save.astype('int')
    _, _scores, _ = calculate_oe_score(logit_layer, x_candidate.reshape(-1, 32, 32, 3), use_xent=True)
    y_pred = lr_model.predict_proba(_scores.reshape(-1, 1))
    ood_index = np.where(y_pred[:, 1] > 0.8)[0]
    id_index = np.where(y_pred[:, 1] <= 0.8)[0]
    ood_save = np.concatenate((ood_save, x_for_save[ood_index]))
    id_save = np.concatenate((id_save, x_for_save[id_index]))
    ood_label = np.concatenate((ood_label, y_candidate[ood_index]))
    id_label = np.concatenate((id_label, y_candidate[id_index]))
    del x_candidate
    print("ood data len: ", len(ood_save))
    print("id data len: ", len(id_save))
    np.save(save_path + "total_id_data.npy", id_save)
    np.save(save_path + "total_ood_data.npy", ood_save)
    np.save(save_path + "total_id_label.npy", id_label)
    np.save(save_path + "total_ood_label.npy", ood_label)


if __name__ == "__main__":
    # mnist
    # model_lr = joblib.load("models/mnist_lr.model")
    # oe_model = load_model("models/oe_Lenet5_mnist_final.h5", compile=False)
    # oe_model.compile(loss=my_ood_loss,
    #                  optimizer='adam',
    #                  metrics=['accuracy'])
    # paths = ["data/mnist/rotation.npy", "data/mnist/brightness.npy", "data/mnist/contrast.npy", "data/mnist/scale.npy",
    #          "data/mnist/shear.npy", "data/mnist/translation.npy"]
    # id_ood_split(paths, oe_model, model_lr, "data/mnist/")
    # cifar10
    model_lr = joblib.load("models/Cifar10_ReNet20/ResNet20_lr.model")
    oe_model = load_model("models/Cifar10_ReNet20/ResNet20_oe.h5", compile=False)
    oe_model.compile(loss=my_ood_loss,
                     optimizer=Adam(lr=1e-3),
                     metrics=['accuracy'])

    paths = ["data/cifar10/rotation.npy", "data/cifar10/brightness.npy", "data/cifar10/contrast.npy", "data/cifar10/scale.npy",
             "data/cifar10/shear.npy", "data/cifar10/translation.npy"]

    id_ood_split_cifar10(paths, oe_model, model_lr, "data/cifar10/resnet20/")
