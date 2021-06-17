from keras.datasets import mnist, cifar10
from image_process import *
from utils import mnist_reader
from utils import SVHN_DatasetUtil

# mnist
# (x_test, y_train), (x_test, y_test) = mnist.load_data()
# cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# fasionmnist
# path = '/Users/qiang.hu/Desktop/work/datasets/fashion_mnist/'
# x_train, y_train = mnist_reader.load_mnist(path, kind='train')
# x_test, y_test = mnist_reader.load_mnist(path, kind='t10k')
# x_train = x_train.reshape(-1, 28, 28)
# x_test = x_test.reshape(-1, 28, 28)
# y_train = to_categorical(y_train, 10)

# SVHN
# (X_train, Y_train), (X_test, Y_test) = SVHN_DatasetUtil.load_data_before_processing()
# x_train = X_train.reshape(-1, 32, 32, 3)
# x_test = X_test.reshape(-1, 32, 32, 3)

# x_save = x_train.copy()
x_save = x_test.copy()
# x_test = x_train


# CIfar10
for i in range(len(x_save)):
    newimg = image_rotation(x_test[i], 40)
    # newimg = image_translation(newimg, 3)
    # newimg = image_shear(x_test[i], 0.6)
    # newimg = image_scale(newimg, 0.7)
    # newimg = image_brightness(x_test[i], 100)
    newimg = image_contrast(newimg, 1.2)
    x_save[i] = newimg
print(x_test.shape)
print(x_save.shape)

# x_save = x_test.copy()
# MNIST
# for i in range(len(x_save)):
#     newimg = image_translation(x_test[i], 2)
#     # newimg = image_rotation(newimg, 30)
#     # newimg = image_translation(x_test[i], 3)
#     # newimg = image_shear(x_test[i], 0.4)
#     # newimg = image_scale(x_test[i], 0.8)
#     newimg = image_brightness(newimg, 100)
#     # newimg = image_contrast(newimg, 1.5)
#     x_save[i] = newimg

# fashion mnist, svhn
# for i in range(len(x_save)):
#     newimg = image_rotation(x_test[i], 30)
#     # newimg = image_translation(newimg, 3)
#     # newimg = image_shear(x_test[i], 0.4)
#     # newimg = image_scale(newimg, 0.8)
#     # newimg = image_brightness(newimg, 50)
#     newimg = image_contrast(newimg, 1.5)
#     x_save[i] = newimg

# model = load_model("models/model_fashion.h5")
# x_save = x_save.reshape(-1, 28, 28)
# x_save = x_save.astype('float32') / 255
# score = model.evaluate(x_save, y_train)
# print(score[1])
np.save("data/cifar10/contrast_test_new.npy", x_save)

