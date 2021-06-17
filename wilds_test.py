from wilds_retrain import *
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
import glob
from PIL import Image
from keras.utils import to_categorical
from pt4tf import *


def idwild_test(model):
    file_folder = "data/iwildcam_v2.0/test_data/*.JPEG"
    candidate_data, y_candidate = image_read(file_folder)
    print(candidate_data.shape)
    # candidate_data = np.swapaxes(np.swapaxes(candidate_data, 1, 3), 1, 2)
    # candidate_data = candidate_data.reshape(candidate_data.shape[0], candidate_data.shape[2], candidate_data.shape[3],
    #                                         candidate_data.shape[1])
    split_len = 1000
    right_num = 0
    total_len = len(candidate_data)
    for i in range(13):
        if (i + 1) * split_len > total_len:
            x_part = candidate_data[(split_len * i):]
            y_part = y_candidate[(split_len * i):]
        else:
            x_part = candidate_data[(split_len * i): (split_len * i) + split_len]
            y_part = y_candidate[(split_len * i): (split_len * i) + split_len]

        predictions = model.predict(x_part)
        predicted_label = np.argmax(predictions, axis=1)
        right_num += len(np.where(predicted_label == y_part)[0])
    # predictions = model.predict(candidate_data)
    # prediction_label = np.argmax(predictions, axis=1)
    # right_num = len(np.where(prediction_label == y_candidate)[0])

    test_acc = right_num / len(candidate_data)
    del candidate_data
    del y_candidate
    return test_acc


def idwild_test_ori(model):
    dataset = get_dataset(dataset='iwildcam', download=False)

    test_data = dataset.get_subset('test',
                                    transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                  transforms.ToTensor()]))
    test_loader = get_train_loader('standard', test_data, batch_size=100)
    dataloader = DataGenerator(test_loader, 182)
    right_num = 0
    print(test_data.__getitem__(0)[0])
    for x_test, y_test in dataloader:
        # print(x_test[0])

        predictions = model.predict(x_test)
        prediction_label = np.argmax(predictions, axis=1)
        real_label = np.argmax(y_test, axis=1)
        right_num += len(np.where(prediction_label == real_label)[0])
        # print(aaaaaaa)
        right_num += 1
        if right_num == 100:
            break

    test_acc = right_num / 10000
    return test_acc


def idwild_test_from_index(index_path, data_type):
    dataset = get_dataset(dataset='fmow', download=False)

    test_data = dataset.get_subset(data_type,
                                    transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                  transforms.ToTensor()]))
    selected_index = np.load(index_path)
    x_list = []
    y_list = []
    count_num = 0
    # print(selected_index)
    for i in selected_index:
        if count_num % 1000 == 0:
            print(count_num)
        count_num += 1
        single_img = test_data.__getitem__(i)[0]
        single_label = test_data.__getitem__(i)[1]
        # print(single_img.numpy())
        single_img_np = single_img.numpy()
        # print(single_img_np.shape)
        x_list.append(single_img_np)
        y_list.append(single_label.numpy())
        # print(aaaaaaa)
    x_list = np.asarray(x_list)
    y_list = np.asarray(y_list)
    x_list = np.swapaxes(np.swapaxes(x_list, 1, 3), 1, 2)
    # x_list = x_list.reshape(x_list.shape[0], x_list.shape[2], x_list.shape[3],
    #                                         x_list.shape[1])
    return x_list, y_list


def idwild_train_from_index(index_path, data_type):
    dataset = get_dataset(dataset='iwildcam', download=False)

    test_data = dataset.get_subset(data_type,
                                    transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                  transforms.ToTensor()]))
    selected_index = np.load(index_path)
    x_list = []
    y_list = []
    count_num = 0
    # print(selected_index)
    for i in selected_index:
        if count_num % 1000 == 0:
            print(count_num)
        count_num += 1
        single_img = test_data.__getitem__(i)[0]
        single_label = test_data.__getitem__(i)[1]
        # print(single_img.numpy())
        single_img_np = single_img.numpy()
        # print(single_img_np.shape)
        x_list.append(single_img_np)
        y_list.append(single_label.numpy())
        # print(aaaaaaa)
    x_list = np.asarray(x_list)
    y_list = np.asarray(y_list)
    x_list = np.swapaxes(np.swapaxes(x_list, 1, 3), 1, 2)
    # x_list = x_list.reshape(x_list.shape[0], x_list.shape[2], x_list.shape[3],
    #                                         x_list.shape[1])
    return x_list, y_list


def fmow_test(model):
    x_test, y_test = idwild_test_from_index("data/fmow/test_index.npy", 'test')
    right_num = 0
    split_len = 1000
    total_len = len(x_test)
    for i in range(13):
        if (i + 1) * split_len > total_len:
            x_part = x_test[(split_len * i):]
            y_part = y_test[(split_len * i):]
        else:
            x_part = x_test[(split_len * i): (split_len * i) + split_len]
            y_part = y_test[(split_len * i): (split_len * i) + split_len]

        predictions = model.predict(x_part)
        predicted_label = np.argmax(predictions, axis=1)
        right_num += len(np.where(predicted_label == y_part)[0])
    return right_num / total_len


if __name__ == '__main__':
    base_path = "models/iwildcam/"
    model_paths = ["ResNet50_1000_0_500_", "ResNet50_1000_1_500_", "ResNet50_1000_2_500_", "ResNet50_1000_3_500_", "ResNet50_1000_ours_500_"]
    # model_paths = ["ResNet50_1000.h5"]

    index_path = "data/iwildcam_v2.0/test_index.npy"
    # test_acc_all = []
    # for iter in range(0, 5):
    #     test_acc_l = []
    #     for model_path in model_paths:
    #         model = load_model(base_path + model_path + str(iter) + '.h5')
    #     # model = 1
    #         test_acc = idwild_test(model)
    #         test_acc_l.append(test_acc)
    #         K.clear_session()
    #         del model
    #         gc.collect()
    #     # test_acc = idwild_test_from_index(model, index_path)
    #     test_acc_all.append(test_acc_l)
    #
    # model_paths = ["ResNet50_1000_0_1000_", "ResNet50_1000_1_1000_", "ResNet50_1000_2_1000_", "ResNet50_1000_3_1000_",
    #                "ResNet50_1000_ours_1000_"]
    # test_acc_all_2 = []
    # for iter in range(0, 5):
    #     test_acc_l = []
    #     for model_path in model_paths:
    #         model = load_model(base_path + model_path + str(iter) + '.h5')
    #         # model = 1
    #         test_acc = idwild_test(model)
    #         test_acc_l.append(test_acc)
    #         K.clear_session()
    #         del model
    #         gc.collect()
    #     # test_acc = idwild_test_from_index(model, index_path)
    #     test_acc_all_2.append(test_acc_l)
    #
    #
    # for _ in test_acc_all:
    #     print(_)
    #
    # print("###################")
    #
    # for _ in test_acc_all_2:
    #     print(_)




    # base_path = "models/FMOW/"
    # model_paths = ["DenseNet121_0_1000_", "DenseNet121_1_1000_", "DenseNet121_2_1000_", "DenseNet121_3_1000_",
    #                "DenseNet121_ours_1000_"]
    #
    # test_acc_all = []
    # for iter in range(0, 5):
    #     test_acc_l = []
    #     for model_path in model_paths:
    #         model = load_model(base_path + model_path + str(iter) + '.h5')
    #     # model = 1
    #         test_acc = fmow_test(model)
    #         test_acc_l.append(test_acc)
    #         K.clear_session()
    #         del model
    #         gc.collect()
    #     # test_acc = idwild_test_from_index(model, index_path)
    #     test_acc_all.append(test_acc_l)
    # for _ in test_acc_all:
    #     print(_)

    # base_path = "models/iwildcam/"
    # model_paths = ["VGG16_half_0_1000_", "VGG16_half_1_1000_", "VGG16_half_2_1000_", "VGG16_half_3_1000_", "VGG16_half_ours_1000_"]
    # # model_paths = ["ResNet50_ours_1000.h5"]
    #
    # # index_path = "data/iwildcam_v2.0/test_index.npy"
    # test_acc_all = []
    # for iter in range(0, 3):
    #     test_acc_l = []
    #     for model_path in model_paths:
    #         model = load_model(base_path + model_path + str(iter) + '.h5')
    #     # model = 1
    #         test_acc = idwild_test(model)
    #         test_acc_l.append(test_acc)
    #         K.clear_session()
    #         del model
    #         gc.collect()
    #     # test_acc = idwild_test_from_index(model, index_path)
    #     test_acc_all.append(test_acc_l)
    # for _ in test_acc_all:
    #     print(_)
    #
    # # model = load_model("models/FMOW/DenseNet121_best.h5")
    # model = load_model("models/iwildcam/ResNet50_1000.h5")
    # test_acc = idwild_test(model)
    # print(test_acc)

    model = load_model("models/iwildcam/ResNet50_2000_best.h5")
    test_acc = idwild_test(model)
    print(test_acc)
