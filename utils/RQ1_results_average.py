import numpy as np
import csv
from numpy import genfromtxt
import glob

# paths = glob.glob("../results/mnist/Lenet1/a_mnist_lenet1*_0.csv")

# paths = glob.glob("../results/mnist/Lenet5/a_mnist_lenet5*_0.csv")
#
# paths = glob.glob("../results/fashion_mnist/Lenet1/a_f_mnist_lenet1*_0.csv")
#
# paths = glob.glob("../results/fashion_mnist/Lenet5/a_f_mnist_lenet5*_0.csv")
#
# paths = glob.glob("../results/resnet/NiN/a_c10_nin*_0.csv")
#
# paths = glob.glob("../results/resnet/resnet20/a_c10_resnet20*_0.csv")


# paths = glob.glob("../results_wrong_ft/mnist/a_mnist_lenet1*_0.csv")

# paths = glob.glob("../results_wrong_ft/mnist/a_mnist_lenet5*_0.csv")
#
# paths = glob.glob("../results_wrong_ft/fashion_mnist/a_f_mnist_lenet1*_0.csv")
#
# paths = glob.glob("../results_wrong_ft/fashion_mnist/a_f_mnist_lenet5*_0.csv")
#
# paths = glob.glob("../results_wrong_ft/resnet/a_c10_nin*_0.csv")
#
paths = glob.glob("../results_wrong_ft/resnet/a_c10_resnet20*_0.csv")



counting_results = [[0, 0] for i in range(11)]
# paths = ["../results/cifar10_nin_ori_acc.csv"]
check_index = 1
split_num = 11
metric_num = 1

for results_path in paths:
    print(results_path)
    my_data = genfromtxt(results_path, delimiter=',')
    for _ in range(2):
        single_data = my_data[:, _]
        # print(single_data)
        for i in range(split_num):
            temporary = 0
            for j in range(metric_num):
                idx = split_num * j + i
                # temporary_list.append(single_data[idx])
                temporary += single_data[idx]
                # print(single_data[idx])
                # print(temporary)
            # print(temporary_list)
            # max_idx = np.argsort(temporary_list)[-1]
            # # print(max_idx)
            # counting_results[i][max_idx] += 1
            if _ == 2:
                temporary = temporary / 8
            else:
                temporary = temporary / 8
            counting_results[i][(_ - 2)] += temporary
            # counting_results[i][_] += temporary

# counting_results = counting_results / len(paths)
counting_results = np.asarray(counting_results)
# counting_results = counting_results * 5
# counting_results[:, 0] = counting_results[:, 0] / len(paths)
for r in counting_results:
    print("{} {}".format(r[0], r[1]))
# print(counting_results)

