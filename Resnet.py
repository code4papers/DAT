from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator

from keras.regularizers import l2
from keras.layers import Conv2D, Dense, Input, add, Activation, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras import optimizers, regularizers
from keras.models import Sequential, Model, load_model
from keras.initializers import RandomNormal

from keras.layers import Convolution2D, MaxPooling2D, Input, Dense, Activation, Flatten,MaxPool2D, Dropout

from keras.models import Model
from keras.datasets import cifar10
import numpy as np
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from resnet50_wilds import *
from utils.oe_utils import *
import argparse


def resnet20(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)
    weight_decay = 1e-6
    stack_n = 3
    def residual_block(intput, out_channel, increase=False):
        if increase:
            stride = (2, 2)
        else:
            stride = (1, 1)

        pre_bn = BatchNormalization()(intput)
        pre_relu = Activation('relu')(pre_bn)

        conv_1 = Conv2D(out_channel, kernel_size=(3, 3), strides=stride, padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(pre_relu)
        bn_1 = BatchNormalization()(conv_1)
        relu1 = Activation('relu')(bn_1)
        conv_2 = Conv2D(out_channel, kernel_size=(3, 3), strides=(1, 1), padding='same',
                        kernel_initializer="he_normal",
                        kernel_regularizer=regularizers.l2(weight_decay))(relu1)
        conv_2 = Dropout(0.5)(conv_2)
        if increase:
            projection = Conv2D(out_channel,
                                kernel_size=(1, 1),
                                strides=(2, 2),
                                padding='same',
                                kernel_initializer="he_normal",
                                kernel_regularizer=regularizers.l2(weight_decay))(intput)
            block = add([conv_2, projection])
        else:
            block = add([intput, conv_2])
        return block

        # build model
        # total layers = stack_n * 3 * 2 + 2
        # stack_n = 5 by default, total layers = 32, which is resnet32
        # input: 32x32x3 output: 32x32x16

    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',
               kernel_initializer="he_normal",
               kernel_regularizer=regularizers.l2(weight_decay))(input_tensor)
    # input: 32x32x16 output: 32x32x16
    for _ in range(stack_n):
        x = residual_block(x, 16, False)

    # input: 32x32x16 output: 16x16x32
    x = residual_block(x, 32, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 32, False)

    # input: 16x16x32 output: 8x8x64
    x = residual_block(x, 64, True)
    for _ in range(1, stack_n):
        x = residual_block(x, 64, False)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)

    # input: 64 output: 10
    x = Dense(num_classes, name='before_softmax',
              kernel_initializer="he_normal",
              kernel_regularizer=regularizers.l2(weight_decay))(x)
    x = Activation('softmax')(x)
    model = Model(input_tensor, x, name='res20-none')
    return model


def generate_data_generator_for_two_images(X1, X2, Y, batch_size):
    data_gen1 = ImageDataGenerator()
    data_gen2 = ImageDataGenerator()
    genX1 = data_gen1.flow(X1, Y, batch_size=batch_size)
    genX2 = data_gen2.flow(X2, batch_size=batch_size)
    while True:
            X1i = genX1.next()
            X2i = genX2 .next()
            yield [X1i[0], X2i ], X1i[1]


def train_resnet():
    input_shape = (32, 32, 3)
    model = resnet20(input_shape, 10)
    model.summary()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train[:40000]
    y_train = y_train[:40000]
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    model_filename = "models/Cifar10_ReNet20/ResNet20_4w.h5"

    checkpoint = ModelCheckpoint(model_filename,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 period=1)
    # cbks = [checkpoint]
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train,
                        batch_size=128,
                        epochs=200,
                        validation_data=(x_test, y_test),
                        verbose=1,
                        callbacks=[checkpoint]
                        )


def train_wilds():
    input_shape = (448, 448, 3)
    model = resnet20(input_shape, 182)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    dataset = get_dataset(dataset='iwildcam', download=False)
    train_data = dataset.get_subset('train',
                                    transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                  transforms.ToTensor()]))
    val_data = dataset.get_subset('val',
                                  transform=transforms.Compose([transforms.Resize((448, 448)),
                                                                transforms.ToTensor()]))

    # print(dir(train_data))
    train_loader = get_train_loader('standard', train_data, batch_size=50)
    val_loader = get_train_loader('standard', val_data, batch_size=50)
    dataloader = DataGenerator(train_loader, 182)
    val_dataloader = DataGenerator(val_loader, 182)

    steps_per_epoch = 129809 // 50

    # for d, l in dataloader:
    #     print(l[0])
    #     break
    checkPoint = ModelCheckpoint("models/iwildcam/ResNet20_best.h5", monitor="val_accuracy", save_best_only=True,
                                 verbose=1)

    model.fit_generator(dataloader,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=val_dataloader,
                        # validation_steps=10,
                        epochs=10,
                        callbacks=[checkPoint])
    model.save("models/iwildcam/ResNet20.h5")


def train_resnet_oe(ood_path, batch_size, save_path):
    input_shape = (32, 32, 3)
    model = resnet20(input_shape, 10)
    model.summary()
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    x_train = x_train[:40000]
    y_train = y_train[:40000]
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    model_filename = save_path
    ood_x = np.load(ood_path)
    ood_x = ood_x.astype('float32') / 255
    ood_x -= x_train_mean
    train_generator = generate_data_generator_for_two_images(x_train, ood_x, y_train, 256)

    checkpoint = ModelCheckpoint(model_filename,
                                 monitor='val_accuracy',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',
                                 period=1)
    model.compile(loss=my_ood_loss,
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    model.fit(train_generator,
              steps_per_epoch=len(x_train) / batch_size,
              epochs=200,
              verbose=1,
              validation_data=(x_test, y_test),
              callbacks=[checkpoint]
              )

    # score = model.evaluate(x_test, y_test)
    # print("test acc: ", score[1])
    # model.save(save_path)


if __name__ == '__main__':
    # ood_path = "data/cifar10/resnet20_ood_combine.npy"
    # batch_size = 128
    # save_path = "models/Cifar10_ReNet20/ResNet20_cifar10_4w-oe.h5"
    # train_resnet_oe(ood_path, batch_size, save_path)
    # train_resnet()
    train_wilds()
