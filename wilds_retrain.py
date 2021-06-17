from transform_retrain import *
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import torchvision.transforms as transforms
import glob
from PIL import Image
from keras.utils import to_categorical
from pt4tf import *
from wilds_test import *


def image_read(file_folder):
    file_list = glob.glob(file_folder)
    transformer = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor()])
    x_list = []
    y_list = []
    count_num = 0
    for file_path in file_list:
        count_num += 1
        if count_num % 1000 == 0:
            print(count_num)
        img = Image.open(file_path)
        # img_np = np.asarray(img)
        # x_list.append(transforms.ToTensor()(img).numpy())
        x_list.append(transformer(img).numpy())
        y_list.append(int(file_path.split('/')[-1].split('_')[-1].split('.')[0]))
        # img_np = transformer(img_np)
        # print(img_np.shape)

    x_list = np.asarray(x_list)
    y_list = np.asarray(y_list)
    x_list = np.swapaxes(np.swapaxes(x_list, 1, 3), 1, 2)
    # x_list = transformer(x_list)
    # y_list = to_categorical(y_list, 182)
    # print(x_list.shape)
    # print(y_list.shape)
    return x_list, y_list


def idwild_retrain(model, metric, select_size, save_path):
    file_folder = "data/iwildcam_v2.0/candidate_data/*.JPEG"
    candidate_data, y_candidate = image_read(file_folder)
    y_candidate = to_categorical(y_candidate, 182)
    # candidate_data = candidate_data.reshape(candidate_data.shape[0], candidate_data.shape[2], candidate_data.shape[3], candidate_data.shape[1])
    print(candidate_data.shape)
    # Entropy
    if metric == 0:
        select_index = entropy_selection(model, candidate_data, select_size)
    # DeepGini
    elif metric == 1:
        select_index = deepgini_selection(model, candidate_data, select_size)
    # MCP
    elif metric == 2:
        select_index = MCP_selection_wilds(model, candidate_data, select_size, 182)
    # Random
    elif metric == 3:
        select_index = random_selection(model, candidate_data, select_size)
    # CES
    elif metric == 4:
        CES_index = CES.conditional_sample(model, candidate_data, select_size)
        select_index = CES.select_from_index(select_size, CES_index)
    # DSA
    # elif metric == 5:
    #     DSA_index = fetch_dsa(model, x_train, candidate_data, 'fgsm', dsa_layer, args)
    #     select_index = CES.select_from_large(select_size, DSA_index)

    dataset = get_dataset(dataset='iwildcam', download=False)
    train_data = dataset.get_subset('train',
                                    transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                  transforms.ToTensor()]))
    # print(dir(train_data))
    train_loader = get_train_loader('standard', train_data, batch_size=16)
    dataloader = DataGenerator(train_loader, 182)

    additional_training_data = candidate_data[select_index]
    additional_training_label = y_candidate[select_index]

    all_index = np.arange(129809)
    insert_index = np.random.choice(all_index, select_size)
    insert_index = insert_index / 16
    insert_index = insert_index.astype('int')
    # print(select_index)
    # print(insert_index)
    print(type(additional_training_data))
    for ite in range(2):
        epoch_start = 0
        for x_train, y_train in dataloader:
            print("data part: ", epoch_start)

            selected_add_index = np.where(insert_index == epoch_start)[0]
            if len(selected_add_index) > 0:

                x_train = np.concatenate((x_train, additional_training_data[selected_add_index]))
                y_train = np.concatenate((y_train, additional_training_label[selected_add_index]))

            model.fit(x_train, y_train, epochs=1)
            epoch_start += 1
    model.save(save_path)


def idwild_retrain_less(model, metric, select_size, save_path):
    file_folder = "data/iwildcam_v2.0/candidate_data/*.JPEG"
    candidate_data, y_candidate = image_read(file_folder)
    y_candidate = to_categorical(y_candidate, 182)
    # candidate_data = candidate_data.reshape(candidate_data.shape[0], candidate_data.shape[2], candidate_data.shape[3], candidate_data.shape[1])
    print(candidate_data.shape)
    # Entropy
    if metric == 0:
        select_index = entropy_selection(model, candidate_data, select_size)
    # DeepGini
    elif metric == 1:
        select_index = deepgini_selection(model, candidate_data, select_size)
    # MCP
    elif metric == 2:
        select_index = MCP_selection_wilds(model, candidate_data, select_size, 182)
    # Random
    elif metric == 3:
        select_index = random_selection(model, candidate_data, select_size)
    # CES
    elif metric == 4:
        CES_index = CES.conditional_sample(model, candidate_data, select_size)
        select_index = CES.select_from_index(select_size, CES_index)
    # DSA
    # elif metric == 5:
    #     DSA_index = fetch_dsa(model, x_train, candidate_data, 'fgsm', dsa_layer, args)
    #     select_index = CES.select_from_large(select_size, DSA_index)

    additional_training_data = candidate_data[select_index]
    additional_training_label = y_candidate[select_index]

    x_train, y_train = idwild_train_from_index("data/iwildcam_v2.0/iwildcam_1000.npy", 'train')
    y_train = to_categorical(y_train, 182)
    dataset = get_dataset(dataset='iwildcam', download=False)
    val_data = dataset.get_subset('val',
                                  transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                transforms.ToTensor()]))
    val_loader = get_train_loader('standard', val_data, batch_size=16)
    val_dataloader = DataGenerator(val_loader, 182)
    final_train = np.concatenate((additional_training_data, x_train))
    final_label = np.concatenate((additional_training_label, y_train))
    checkPoint = ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True,
                                 verbose=1)
    model.fit(final_train,
              final_label,
              batch_size=16,
              validation_data=val_dataloader,
              epochs=5,
              callbacks=[checkPoint])


if __name__ == '__main__':

    select_size = 300
    for iter in range(0, 5):
        for metric in [0, 1, 2, 3]:
            model = load_model("models/iwildcam/ResNet50_1000.h5")
            # metric = 4

            save_path = "models/iwildcam/ResNet50_1000_" + str(metric) + "_" + str(select_size) + "_" + str(iter) + ".h5"
            idwild_retrain_less(model, metric, select_size, save_path)
            K.clear_session()
            del model
            gc.collect()

    select_size = 500
    for iter in range(0, 5):
        for metric in [0, 1, 2, 3]:
            model = load_model("models/iwildcam/ResNet50_1000.h5")
            # metric = 4

            save_path = "models/iwildcam/ResNet50_1000_" + str(metric) + "_" + str(select_size) + "_" + str(
                iter) + ".h5"
            idwild_retrain_less(model, metric, select_size, save_path)
            K.clear_session()
            del model
            gc.collect()


    select_size = 1000
    for iter in range(0, 5):
        for metric in [0, 1, 2, 3]:
            model = load_model("models/iwildcam/ResNet50_1000.h5")
            # metric = 4

            save_path = "models/iwildcam/ResNet50_1000_" + str(metric) + "_" + str(select_size) + "_" + str(
                iter) + ".h5"
            idwild_retrain_less(model, metric, select_size, save_path)
            K.clear_session()
            del model
            gc.collect()

