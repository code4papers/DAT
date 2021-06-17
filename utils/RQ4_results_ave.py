import numpy as np
import csv
from numpy import genfromtxt
import glob
from os import path

# paths = glob.glob("../results/resnet/NiN/a_*")
paths = glob.glob("../results/resnet/NiN/a_*_new.csv")
paths_2 = ["a_c10_nin_fgsm_100.csv",
           "a_c10_nin_fgsm_300.csv",
           "a_c10_nin_fgsm_500.csv",
           "a_c10_nin_fgsm_1000.csv",
           "a_c10_nin_pgd_100.csv",
           "a_c10_nin_pgd_300.csv",
           "a_c10_nin_pgd_500.csv",
           "a_c10_nin_pgd_1000.csv"]

paths_2 = ["../results/resnet/NiN/" + p for p in paths_2]

# paths_2 = ["a_c10_resnet20_fgsm_100.csv",
#            "a_c10_resnet20_fgsm_300.csv",
#            "a_c10_resnet20_fgsm_500.csv",
#            "a_c10_resnet20_fgsm_1000.csv",
#            "a_c10_resnet20_pgd_100.csv",
#            "a_c10_resnet20_pgd_300.csv",
#            "a_c10_resnet20_pgd_500.csv",
#            "a_c10_resnet20_pgd_1000.csv"]
#
# paths_2 = ["../results/resnet/resnet20/" + p for p in paths_2]

paths = paths + paths_2

counting_results = [[0, 0, 0, 0, 0, 0, 0] for i in range(11)]
check_index = 1
split_num = 11
metric_num = 6

for results_path in paths:
    # print(results_path[:-4])
    my_data = genfromtxt(results_path, delimiter=',')
    single_data = my_data[:, 1]
    # original
    our_path_base = "../results/resnet/NiN_ours/"
    # ablation
    # our_path_base_2 = "../results_ablation/resnet/"
    # mnist
    # ours_path = our_path_base + results_path.split("/")[-1][:-4] + "_ours.csv"
    # ours_path_2 = our_path_base_2 + results_path.split("/")[-1][:-4] + "_ours_ablation.csv"
    # print(ours_path_2)

    # f-mnist
    # ours_path = our_path_base + "a_fashion" + results_path.split("/")[-1][3: -4] + "_ours_margin.csv"
    # ours_path_2 = our_path_base_2 + "a_fashion" + results_path.split("/")[-1][3: -4] + "_ours_ablation.csv"

    # cifar10
    if 'fgsm' in results_path or 'pgd' in results_path:
        print(results_path.split("/")[-1])
        print(results_path.split("/")[-1][6: -4])
        ours_path = our_path_base + "a_" + results_path.split("/")[-1][6: -4] + "_ours_gini_new.csv"
    else:
        ours_path = our_path_base + "a_" + results_path.split("/")[-1][6: -7] + "ours_gini_new.csv"

    # ours_path = our_path_base + "a_" + results_path.split("/")[-1][6: -4] + "_ours.csv"
    # ours_path_2 = our_path_base_2 + "a_" + results_path.split("/")[-1][6: -4] + "_ours_ablation.csv"

    # if path.exists(ours_path_2):
    #     print("hahah")
    #     ours_path = ours_path_2
    # print(ours_path)
    our_data = genfromtxt(ours_path, delimiter=',')
    single_our_data = our_data[:, 1]
    # print(ours_path)
    # print(single_our_data)
    for i in range(split_num):
        if i < 4:
            temporary_list = []
            for j in range(metric_num):
                idx = split_num * j + i
                temporary_list.append(single_data[idx])
            # print(temporary_list)
            temporary_list.append(single_our_data[i])
            sorted_index = np.argsort(temporary_list)

            # mean
            mean_val = np.mean(temporary_list)
            greater_than_mean = np.where(temporary_list > mean_val)[0]
            # print(greater_than_mean)
            max_idx = sorted_index[-1]
            second_max_idx = sorted_index[-2]
            third_max_idx = sorted_index[-3]
            # print(max_idx)
            # Max index + 1 ###################
            counting_results[i][max_idx] += 1

            # Second max index + 1 ###################
            # counting_results[i][second_max_idx] += 1

            # Third max index + 1  ################
            # counting_results[i][third_max_idx] += 1

            # Average + 1 #####################
            # for idx in greater_than_mean:
            #     counting_results[i][idx] += 1



for r in counting_results:
    print(r)
# print(counting_results)

