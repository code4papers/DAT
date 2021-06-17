from wilds_retrain import *
from utils.oe_utils import *
import joblib
from transform_retrain import *
from wilds_test import *

def DAT_wilds(model, oe_model, lr_model, metric, save_path, select_size):
    file_folder = "data/iwildcam_v2.0/candidate_data/*.JPEG"
    candidate_data, y_candidate = image_read(file_folder)
    candidate_label = to_categorical(y_candidate, 182)

    file_folder = "data/iwildcam_v2.0/test_data/*.JPEG"
    new_test_data, y_test = image_read(file_folder)
    y_test = to_categorical(y_test, 182)
    # candidate_data = candidate_data.reshape(candidate_data.shape[0], candidate_data.shape[2], candidate_data.shape[3],
    #                                         candidate_data.shape[1])
    # new_test_data = new_test_data.reshape(new_test_data.shape[0], new_test_data.shape[2], new_test_data.shape[3],
    #                                       new_test_data.shape[1])
    batch_size = 16
    epochs = 2
    para_here = 0.5
    para_there = 0.5

    # step 1: detect OOD data
    logit_layer = K.function(inputs=oe_model.input, outputs=oe_model.layers[-2].output)
    split_len = len(candidate_data) / 50
    for i in range(50):
        if i == 0:
            _, _scores, _ = calculate_oe_score(logit_layer, candidate_data[int(i * split_len): int((i+1) * split_len)], use_xent=True)
        else:
            _, id_scores_s, _ = calculate_oe_score(logit_layer, candidate_data[int(i * split_len): int((i+1) * split_len)],
                                                 use_xent=True)

            _scores = np.concatenate((_scores, id_scores_s))

    # _, _scores, _ = calculate_oe_score(logit_layer, candidate_data, use_xent=True)

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
    split_len = 1000
    total_len = len(candidate_data)
    for i in range(10):
        if i == 0:
            x_part = candidate_data[(split_len * i): (split_len * i) + split_len]
            candidate_prediction_label = np.argmax(model.predict(x_part), axis=1)
            # print(x_part.shape)
        elif (i + 1) * split_len > total_len:
            x_part = candidate_data[(split_len * i):]
            # print(x_part.shape)
            candidate_prediction_label_s = np.argmax(model.predict(x_part), axis=1)
            candidate_prediction_label = np.concatenate((candidate_prediction_label, candidate_prediction_label_s))
        else:
            # print(x_part.shape)
            x_part = candidate_data[(split_len * i): (split_len * i) + split_len]
            candidate_prediction_label_s = np.argmax(model.predict(x_part), axis=1)
            candidate_prediction_label = np.concatenate((candidate_prediction_label, candidate_prediction_label_s))

    for i in range(13):
        if i == 0:
            x_part = new_test_data[(split_len * i): (split_len * i) + split_len]
            reference_prediction_label = np.argmax(model.predict(x_part), axis=1)
        elif (i + 1) * split_len > total_len:
            x_part = new_test_data[(split_len * i):]
            reference_prediction_label_s = np.argmax(model.predict(x_part), axis=1)
            reference_prediction_label = np.concatenate((reference_prediction_label, reference_prediction_label_s))
        else:
            x_part = new_test_data[(split_len * i): (split_len * i) + split_len]
            reference_prediction_label_s = np.argmax(model.predict(x_part), axis=1)
            reference_prediction_label = np.concatenate((reference_prediction_label, reference_prediction_label_s))

    # candidate_prediction_label = np.argmax(model.predict(candidate_data), axis=1)
    # reference_prediction_label = np.argmax(model.predict(new_test_data), axis=1)
    # print(reference_prediction_label)
    reference_labels = []

    for i in range(0, 182):
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
        for i in range(0, 182):
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

    dataset = get_dataset(dataset='iwildcam', download=False)
    train_data = dataset.get_subset('train',
                                    transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                  transforms.ToTensor()]))
    # print(dir(train_data))
    train_loader = get_train_loader('standard', train_data, batch_size=16)
    dataloader = DataGenerator(train_loader, 182)

    all_index = np.arange(129809)
    insert_index = np.random.choice(all_index, select_size)
    insert_index = insert_index / 16
    insert_index = insert_index.astype('int')

    for ite in range(2):
        epoch_start = 0
        for x_train, y_train in dataloader:
            print("data part: ", epoch_start)
            selected_add_index = np.where(insert_index == epoch_start)[0]
            if len(selected_add_index) > 0:
                # print(selected_add_index)
                # print(type(selected_add_index))
                # print(additional_training_data[selected_add_index].shape)
                x_train = np.concatenate((x_train, selected_data[selected_add_index]))
                y_train = np.concatenate((y_train, selected_label[selected_add_index]))
            # print(x_train.shape)
            # print(y_train.shape)
            model.fit(x_train, y_train, epochs=1)
            epoch_start += 1
    model.save(save_path)


def DAT_wilds_less(model, oe_model, lr_model, metric, save_path, select_size):
    file_folder = "data/iwildcam_v2.0/candidate_data/*.JPEG"
    candidate_data, y_candidate = image_read(file_folder)
    candidate_label = to_categorical(y_candidate, 182)

    file_folder = "data/iwildcam_v2.0/test_data/*.JPEG"
    new_test_data, y_test = image_read(file_folder)
    y_test = to_categorical(y_test, 182)
    # candidate_data = candidate_data.reshape(candidate_data.shape[0], candidate_data.shape[2], candidate_data.shape[3],
    #                                         candidate_data.shape[1])
    # new_test_data = new_test_data.reshape(new_test_data.shape[0], new_test_data.shape[2], new_test_data.shape[3],
    #                                       new_test_data.shape[1])
    batch_size = 16
    epochs = 2
    para_here = 0.3
    para_there = 0.3

    # step 1: detect OOD data
    logit_layer = K.function(inputs=oe_model.input, outputs=oe_model.layers[-2].output)
    split_len = len(candidate_data) / 50
    for i in range(50):
        if i == 0:
            _, _scores, _ = calculate_oe_score(logit_layer,
                                               candidate_data[int(i * split_len): int((i + 1) * split_len)],
                                               use_xent=True)
        else:
            _, id_scores_s, _ = calculate_oe_score(logit_layer,
                                                   candidate_data[int(i * split_len): int((i + 1) * split_len)],
                                                   use_xent=True)

            _scores = np.concatenate((_scores, id_scores_s))

    # _, _scores, _ = calculate_oe_score(logit_layer, candidate_data, use_xent=True)

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
    split_len = 1000
    total_len = len(candidate_data)
    for i in range(10):
        if i == 0:
            x_part = candidate_data[(split_len * i): (split_len * i) + split_len]
            candidate_prediction_label = np.argmax(model.predict(x_part), axis=1)
            # print(x_part.shape)
        elif (i + 1) * split_len > total_len:
            x_part = candidate_data[(split_len * i):]
            # print(x_part.shape)
            candidate_prediction_label_s = np.argmax(model.predict(x_part), axis=1)
            candidate_prediction_label = np.concatenate((candidate_prediction_label, candidate_prediction_label_s))
        else:
            # print(x_part.shape)
            x_part = candidate_data[(split_len * i): (split_len * i) + split_len]
            candidate_prediction_label_s = np.argmax(model.predict(x_part), axis=1)
            candidate_prediction_label = np.concatenate((candidate_prediction_label, candidate_prediction_label_s))

    for i in range(13):
        if i == 0:
            x_part = new_test_data[(split_len * i): (split_len * i) + split_len]
            reference_prediction_label = np.argmax(model.predict(x_part), axis=1)
        elif (i + 1) * split_len > total_len:
            x_part = new_test_data[(split_len * i):]
            reference_prediction_label_s = np.argmax(model.predict(x_part), axis=1)
            reference_prediction_label = np.concatenate((reference_prediction_label, reference_prediction_label_s))
        else:
            x_part = new_test_data[(split_len * i): (split_len * i) + split_len]
            reference_prediction_label_s = np.argmax(model.predict(x_part), axis=1)
            reference_prediction_label = np.concatenate((reference_prediction_label, reference_prediction_label_s))

    # candidate_prediction_label = np.argmax(model.predict(candidate_data), axis=1)
    # reference_prediction_label = np.argmax(model.predict(new_test_data), axis=1)
    # print(reference_prediction_label)
    reference_labels = []

    for i in range(0, 182):
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
        for i in range(0, 182):
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

    del candidate_data
    del new_test_data

    x_train, y_train = idwild_train_from_index("data/iwildcam_v2.0/iwildcam_1000.npy", 'train')
    y_train = to_categorical(y_train, 182)
    dataset = get_dataset(dataset='iwildcam', download=False)
    val_data = dataset.get_subset('val',
                                  transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                transforms.ToTensor()]))
    val_loader = get_train_loader('standard', val_data, batch_size=16)
    val_dataloader = DataGenerator(val_loader, 182)
    final_train = np.concatenate((selected_data, x_train))
    final_label = np.concatenate((selected_label, y_train))
    checkPoint = ModelCheckpoint(save_path, monitor="val_accuracy", save_best_only=True,
                                 verbose=1)
    model.fit(final_train,
              final_label,
              batch_size=16,
              validation_data=val_dataloader,
              # validation_steps=10,
              epochs=5,
              callbacks=[checkPoint])


if __name__ == '__main__':
    # iwilds
    oe_model = load_model("models/iwildcam/ResNet50_oe.h5", compile=False)
    oe_model.compile(loss=my_ood_loss, optimizer=optimizers.Adam(lr=3e-5), metrics=['accuracy'])
    model_lr = joblib.load("models/iwildcam/ResNet50_lr.model")

    select_size = 300
    metric = 1
    for iter in range(0, 5):
        model = load_model("models/iwildcam/ResNet50_1000.h5")
        save_path = "models/iwildcam/ResNet50_1000_ours_" + str(select_size) + "_" + str(iter) + ".h5"
        DAT_wilds_less(model, oe_model, model_lr, metric, save_path, select_size)
        K.clear_session()
        del model
        gc.collect()

    select_size = 500
    metric = 1
    for iter in range(0, 5):
        model = load_model("models/iwildcam/ResNet50_1000.h5")
        save_path = "models/iwildcam/ResNet50_1000_ours_" + str(select_size) + "_" + str(iter) + ".h5"
        DAT_wilds_less(model, oe_model, model_lr, metric, save_path, select_size)
        K.clear_session()
        del model
        gc.collect()

    select_size = 1000
    metric = 1
    for iter in range(0, 5):
        model = load_model("models/iwildcam/ResNet50_1000.h5")
        save_path = "models/iwildcam/ResNet50_1000_ours_" + str(select_size) + "_" + str(iter) + ".h5"
        DAT_wilds_less(model, oe_model, model_lr, metric, save_path, select_size)
        K.clear_session()
        del model
        gc.collect()


