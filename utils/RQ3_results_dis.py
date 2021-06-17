import numpy as np
import csv
from numpy import genfromtxt
import glob

paths = glob.glob("../results/resnet/NiN/a_*")
# paths = ["../results/mnist/Lenet1/a_mnist_lenet1_brightness_100.csv"]
id_counting_results = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(6)]
ood_counting_results = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(6)]

check_index = 1
split_num = 11
metric_num = 6

for results_path in paths:
    print(results_path)
    my_data = genfromtxt(results_path, delimiter=',')
    id_data = my_data[:, 6]
    ood_data = my_data[:, 7]
    # print(id_data)
    # print(ood_data)
    # print(aaa)
    for i in range(metric_num):
        for j in range(split_num):
            idx = split_num * i + j
            # print(idx)
            # temporary_list.append(single_data[idx])
            # print(idx)
            id_this_num = id_data[idx]
            ood_this_num = ood_data[idx]
            id_counting_results[i][j] += id_this_num
            ood_counting_results[i][j] += ood_this_num


# counting_results = counting_results / len(paths)
id_counting_results = np.asarray(id_counting_results)
ood_counting_results = np.asarray(ood_counting_results)
# counting_results[:, 0] = counting_results[:, 0] / len(paths)
# for r in id_counting_results:
#     print(r)
# print("####################")
# for r in ood_counting_results:
#     print(r)
# print(counting_results)

final_result = id_counting_results / (id_counting_results + ood_counting_results)
print(final_result)
csv_file = open("../results/for_statistics.csv", "a")
try:
    writer = csv.writer(csv_file)
    for i in range(11):
        writer.writerow(final_result[:, i])
finally:
    csv_file.close()
