import torch
import torch.utils.data as torch_data
import random
import scipy.io as sio
import numpy as np
from sklearn import preprocessing
from global_define import *
import matplotlib.pyplot as plt


def init_data():
    # read data set
    data_set_mat_filename = "data/Hyperspectral Image/salina/Salinas_corrected.mat"
    data_label_mat_filename = "data/Hyperspectral Image/salina/Salinas_gt.mat"

    # tract data
    data_set = sio.loadmat(data_set_mat_filename)['salinas_corrected']
    shape = data_set.shape
    data_set = data_set.reshape(-1, DataSize[2])
    data_set = preprocessing.minmax_scale(data_set, axis=1).astype(np.float32)
    data_set = data_set.reshape(shape)

    data_label = sio.loadmat(data_label_mat_filename)['salinas_gt']
    data_set_x = data_set.shape[0]
    data_set_y = data_set.shape[1]
    d_x = DataSize[0]
    d_y = DataSize[1]
    data_set_x = data_set_x - d_x + 1
    data_set_y = data_set_y - d_y + 1
    data = np.zeros((data_set_x, data_set_y, DataSize[0],
                     DataSize[1], DataSize[2])).astype(np.float32)
    label = np.zeros((data_set_x, data_set_y)).astype(np.int8)
    # generate 7*7 data set
    for i in range(data_set_x - d_x + 1):
        for j in range(data_set_y - d_y + 1):
            data[i, j] = data_set[i:i+d_x, j:j+d_y]
            label[i, j] = data_label[int(i+(d_x-1)/2), int(j+(d_y-1)/2)]

    # remove all the zeros
    data = data.reshape((data_set_x*data_set_y, DataSize[0], DataSize[1], DataSize[2]))
    label = label.reshape(data_set_x*data_set_y)

    data = data[label != 0]
    label = label[label != 0]

    train_set_data = np.zeros(((LabelSize - 1)*TrainNumPerClass,
                               DataSize[0], DataSize[1], DataSize[2])).astype(np.float32)
    train_set_label = np.zeros(((LabelSize - 1)*TrainNumPerClass)).astype(np.uint8)
    test_set_data = np.zeros((data.shape[0] - train_set_data.shape[0],
                              DataSize[0], DataSize[1], DataSize[2])).astype(np.float32)
    test_set_label = np.zeros(data.shape[0] - train_set_data.shape[0]).astype(np.uint8)

    tmp = 0
    for i in range(1, LabelSize):
        data_i = data[label == i]
        data_num = data_i.shape[0]
        extract_nums = list(range(data_num))
        extract_subs = random.sample(extract_nums, TrainNumPerClass)
        train_set_data[(i-1)*TrainNumPerClass:i*TrainNumPerClass] = data_i[extract_subs, :]
        train_set_label[(i-1)*TrainNumPerClass:i*TrainNumPerClass] = i

        data_i[extract_subs] = np.inf
        data_i = data_i[data_i[:, 0, 0, 0] != np.inf]
        test_shape = data_i.shape
        test_set_data[tmp:tmp+test_shape[0]] = data_i
        test_set_label[tmp:tmp+test_shape[0]] = i
        tmp = tmp + test_shape[0]

    train_set_data = torch.from_numpy(train_set_data).type(torch.FloatTensor)
    train_set_label = torch.from_numpy(train_set_label).type(torch.LongTensor)
    train_set = torch_data.TensorDataset(
        data_tensor=train_set_data,
        target_tensor=train_set_label
    )

    # select 4000 test samples
    data_num = test_set_data.shape[0]
    extract_nums = list(range(data_num))
    extract_subs = random.sample(extract_nums, 10000)
    test_set_data = test_set_data[extract_subs, :]
    test_set_label = test_set_label[extract_subs]

    torch.save(train_set, 'data/train_data_set.lib')
    np.save("data/test_set_data.npy", test_set_data)
    np.save("data/test_set_label.npy", test_set_label)


if __name__ == "__main__":
    init_data()
# sudo sysctl -w vm.drop_caches=3

