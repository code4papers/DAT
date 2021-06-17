import tensorflow as tf
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
from keras import optimizers
from pt4tf import *
import torchvision.transforms as transforms
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Activation, Flatten, Reshape
from tensorflow.keras.models import Model
from wilds_retrain import *
from keras.preprocessing.image import ImageDataGenerator
from utils.oe_utils import *


def train_normal():
    # base_model = tf.keras.applications.ResNet50(
    #     include_top=False,
    #     # weights="imagenet",
    #     input_tensor=None,
    #     input_shape=None,
    #     pooling=None,
    #     classes=182,
    # )

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        # weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=182,
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Reshape((-1, 2))(x)
    x = Dense(182)(x)

    predictions = Activation('softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=3e-5),
                  metrics=['accuracy'])
    model.summary()

    print(model.output_shape)
    dataset = get_dataset(dataset='iwildcam', download=False)

    train_data = dataset.get_subset('train',
                                    transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                 transforms.ToTensor()]))
    val_data = dataset.get_subset('val',
                                  transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                transforms.ToTensor()]))

    # print(dir(train_data))
    train_loader = get_train_loader('standard', train_data, batch_size=16)
    val_loader = get_train_loader('standard', val_data, batch_size=16)
    dataloader = DataGenerator(train_loader, 182)
    val_dataloader = DataGenerator(val_loader, 182)

    steps_per_epoch = 129809 // 16

    # for d, l in dataloader:
    #     print(l[0])
    #     break
    checkPoint = ModelCheckpoint("models/iwildcam/ResNet50_half_best.h5", monitor="val_accuracy", save_best_only=True, verbose=1)
    model.fit_generator(dataloader,
                        steps_per_epoch=100,
                        validation_data=val_dataloader,
                        # validation_steps=10,
                        epochs=5,
                        callbacks=[checkPoint])
    model.save("models/iwildcam/ResNet50_half.h5")


def train_less_data(index_path):
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        # weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=182,
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Reshape((-1, 2))(x)
    x = Dense(182)(x)

    predictions = Activation('softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.Adam(lr=3e-5),
                  metrics=['accuracy'])
    model.summary()

    x_train, y_train = idwild_train_from_index(index_path, 'train')
    y_train = to_categorical(y_train, 182)
    print(x_train.shape)
    print(y_train.shape)
    dataset = get_dataset(dataset='iwildcam', download=False)
    val_data = dataset.get_subset('val',
                                  transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                transforms.ToTensor()]))
    val_loader = get_train_loader('standard', val_data, batch_size=16)
    val_dataloader = DataGenerator(val_loader, 182)

    checkPoint = ModelCheckpoint("models/iwildcam/ResNet50_1000_best.h5", monitor="val_accuracy", save_best_only=True,
                                 verbose=1)
    model.fit(x_train,
                y_train,
                validation_data=val_dataloader,
                # validation_steps=10,
                epochs=10,
                callbacks=[checkPoint])
    model.save("models/iwildcam/ResNet50_1000.h5")


def generate_data_generator_for_two_images(X1, X2, Y, batch_size):
    data_gen1 = ImageDataGenerator()
    data_gen2 = ImageDataGenerator()
    genX1 = data_gen1.flow(X1, Y, batch_size=batch_size)
    genX2 = data_gen2.flow(X2, batch_size=batch_size)
    while True:
            X1i = genX1.next()
            X2i = genX2 .next()
            yield [X1i[0], X2i], X1i[1]


def train_or_wilds(save_path):
    ood_file_folder = "data/iwildcam_v2.0/OE_data/*.JPEG"
    id_file_folder = "data/iwildcam_v2.0/OE_ID_data/*.JPEG"
    ood_data, ood_label = image_read(ood_file_folder)
    id_data, id_label = image_read(id_file_folder)
    # ood_data = ood_data.reshape(ood_data.shape[0], ood_data.shape[2], ood_data.shape[3],
    #                                         ood_data.shape[1])
    # id_data = id_data.reshape(id_data.shape[0], id_data.shape[2], id_data.shape[3],
    #                                         id_data.shape[1])
    ood_label = to_categorical(ood_label, 182)
    id_label = to_categorical(id_label, 182)
    train_generator = generate_data_generator_for_two_images(id_data, ood_data, id_label, 20)
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        # weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=182,
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # x = Reshape((-1, 2))(x)
    x = Dense(182)(x)

    predictions = Activation('softmax', name='predictions')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss=my_ood_loss,
                  optimizer=optimizers.Adam(lr=3e-5),
                  metrics=['accuracy'])
    model.fit(train_generator,
              steps_per_epoch=20000 / 20,
              epochs=20,
              verbose=1)
    model.save(save_path)


if __name__ == "__main__":
    # save_path = "models/iwildcam/ResNet50_oe.h5"
    # train_or_wilds(save_path)
    # train_normal()
    train_less_data("data/iwildcam_v2.0/iwildcam_1000.npy")
