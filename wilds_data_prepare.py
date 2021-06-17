from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader
import numpy as np
import torch
from PIL import Image

import torchvision.transforms as transforms

dataset = get_dataset(dataset='iwildcam', download=False)
pil_data = dataset.get_input(1)
pixels = np.asarray(pil_data)
# print(pixels)

train_data = dataset.get_subset('test',
                                transform=transforms.Compose([
                                                             # transforms.Resize((448, 448)),
                                                             transforms.ToTensor()
                                ]
                                )
                                )
all_index = np.arange(42791)
oe_index = np.random.choice(all_index, 20000, replace=False)
remain_index = np.delete(all_index, oe_index)
all_index = np.arange(len(remain_index))
# print(len(remain_index))
candidate_index_idx = np.random.choice(all_index, 10000, replace=False)
candidate_index = remain_index[candidate_index_idx]
test_index = np.delete(remain_index, candidate_index_idx)
print(len(oe_index))
print(len(candidate_index))
print(len(test_index))

np.save("data/iwildcam_v2.0/oe_od_index.npy", oe_index)
np.save("data/iwildcam_v2.0/candidate_index.npy", candidate_index)
np.save("data/iwildcam_v2.0/test_index.npy", test_index)


save_folder = "data/iwildcam_v2.0/OE_data/"
for i in oe_index:
    single_img = train_data.__getitem__(i)[0]
    single_label = train_data.__getitem__(i)[1]
    demo_array = np.moveaxis(single_img.numpy() * 255, 0, -1)
    img_try = Image.fromarray(demo_array.astype(np.uint8))
    final_path = save_folder + str(i) + '_' + str(single_label.numpy()) + '.JPEG'
    img_try.save(final_path)
print("finish oe")
#
# save_folder = "data/iwildcam_v2.0/candidate_data/"
# for i in candidate_index:
#     single_img = train_data.__getitem__(i)[0]
#     single_label = train_data.__getitem__(i)[1]
#     demo_array = np.moveaxis(single_img.numpy() * 255, 0, -1)
#     img_try = Image.fromarray(demo_array.astype(np.uint8))
#     final_path = save_folder + str(i) + '_' + str(single_label.numpy()) + '.JPEG'
#     img_try.save(final_path)
# print("finish candidate")
#
# save_folder = "data/iwildcam_v2.0/test_data/"
# for i in test_index:
#     single_img = train_data.__getitem__(i)[0]
#     single_label = train_data.__getitem__(i)[1]
#     demo_array = np.moveaxis(single_img.numpy() * 255, 0, -1)
#     img_try = Image.fromarray(demo_array.astype(np.uint8))
#     final_path = save_folder + str(i) + '_' + str(single_label.numpy()) + '.JPEG'
#     img_try.save(final_path)
# print("finish test")



# all_index = np.arange(120000)
# oe_index = np.random.choice(all_index, 20000, replace=False)
#
#
# train_data = dataset.get_subset('train',
#                                 transform=transforms.Compose([
#                                                              # transforms.Resize((448, 448)),
#                                                              transforms.ToTensor()
#                                 ]
#                                 )
#                                 )
#
#
# save_folder = "data/iwildcam_v2.0/OE_ID_data/"
# for i in oe_index:
#     single_img = train_data.__getitem__(i)[0]
#     single_label = train_data.__getitem__(i)[1]
#     demo_array = np.moveaxis(single_img.numpy() * 255, 0, -1)
#     img_try = Image.fromarray(demo_array.astype(np.uint8))
#     final_path = save_folder + str(i) + '_' + str(single_label.numpy()) + '.JPEG'
#     img_try.save(final_path)
# print("finish test")
