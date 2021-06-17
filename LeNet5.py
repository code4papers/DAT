from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten, Dropout
from keras.models import Model
import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from utils.oe_utils import *
from utils import mnist_reader
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D


def Lenet1():
    img_rows, img_cols = 28, 28
    input_tensor = Input(shape=(img_rows, img_cols, 1))
    # block1
    x = Conv2D(4, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    # block2
    x = Conv2D(12, (5, 5), padding='valid', activation='relu', kernel_initializer='he_normal')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(10, kernel_initializer='he_normal')(x)
    x = Activation('softmax', name='predictions')(x)
    model = Model(input_tensor, x)

    return model


def Lenet5():
    # ori acc 0.9889
    nb_classes = 10
    # convolution kernel size
    kernel_size = (5, 5)
    img_rows, img_cols = 28, 28
    input_tensor = Input(shape=(img_rows, img_cols, 1))

    # block1
    x = Convolution2D(6, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    # block2
    x = Convolution2D(16, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dense(120, activation='relu', name='fc1')(x)
    x = Dense(84, activation='relu', name='fc2')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)
    return model


def Lenet5_dropout():
    # ori acc 0.9889
    nb_classes = 10
    # convolution kernel size
    kernel_size = (5, 5)
    img_rows, img_cols = 28, 28
    input_tensor = Input(shape=(img_rows, img_cols, 1))

    # block1
    x = Convolution2D(6, kernel_size, activation='relu', padding='same', name='block1_conv1')(input_tensor)
    x = MaxPooling2D(pool_size=(2, 2), name='block1_pool1')(x)

    # block2
    x = Convolution2D(16, kernel_size, activation='relu', padding='same', name='block2_conv1')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='block2_pool1')(x)

    x = Flatten(name='flatten')(x)
    x = Dropout(0.2)(x, training=True)
    x = Dense(120, activation='relu', name='fc1')(x)
    x = Dense(84, activation='relu', name='fc2')(x)
    x = Dense(nb_classes, name='before_softmax')(x)
    x = Activation('softmax', name='predictions')(x)

    model = Model(input_tensor, x)
    return model


def generate_data_generator_for_two_images(X1, X2, Y, batch_size):
    data_gen1 = ImageDataGenerator()
    data_gen2 = ImageDataGenerator()
    genX1 = data_gen1.flow(X1, Y, batch_size=batch_size)
    genX2 = data_gen2.flow(X2, batch_size=batch_size)
    while True:
            X1i = genX1.next()
            X2i = genX2 .next()
            yield [X1i[0], X2i], X1i[1]


def train_oe(ood_path, model_type, batch_size, save_path):
    if model_type == 'lenet5':
        model = Lenet5()
    elif model_type == 'lenet1':
        model = Lenet1()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_test = keras.utils.to_categorical(y_test, 10)
    y_train = keras.utils.to_categorical(y_train, 10)
    x_train = x_train[:50000]
    y_train = y_train[:50000]
    ood_x = np.load(ood_path)
    ood_x = ood_x.astype('float32') / 255
    ood_x = ood_x.reshape(-1, 28, 28, 1)
    train_generator = generate_data_generator_for_two_images(x_train, ood_x, y_train, 256)

    model.compile(loss=my_ood_loss,
                  optimizer='adam',
                  metrics=['accuracy'])

    model.fit(train_generator,
              steps_per_epoch=len(x_train) / batch_size,
              epochs=20,
              verbose=1)

    score = model.evaluate(x_test, y_test)
    print("test acc: ", score[1])
    model.save(save_path)


def train_normal(model_type):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_test = keras.utils.to_categorical(y_test, 10)
    y_train = keras.utils.to_categorical(y_train, 10)

    if model_type == 'lenet5':
        model = Lenet5()
    else:
        model = Lenet1()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    his = model.fit(x_train[:50000], y_train[:50000], batch_size=256, shuffle=True, epochs=10,
                    validation_data=(x_test, y_test), verbose=1)

    print(his.history['val_accuracy'])
    model.save("models/MNIST-Lenet1/Lenet-1-5w.h5")


if __name__ == '__main__':
    ood_path = "data/mnist/lenet1_ood_combine.npy"
    save_path = "models/mnist_Lenet1_mnist-5w-oe.h5"
    train_oe(ood_path, 'lenet1', 256, save_path)
    # train_normal('lenet1')
