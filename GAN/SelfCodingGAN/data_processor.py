import torch
import torch.utils.data as torch_data
import random
import scipy.io as sio
import numpy as np
from sklearn import preprocessing
import global_define
import matplotlib.pyplot as plt


def init_data():
    # read data set
    data_set_mat_filename = "data/Hyperspectral Image/salina/Salinas_corrected.mat"
    data_label_mat_filename = "data/Hyperspectral Image/salina/Salinas_gt.mat"

    # tract data
    data_set = sio.loadmat(data_set_mat_filename)['salinas_corrected']
    data_label = sio.loadmat(data_label_mat_filename)['salinas_gt']

    # remove all the zeros
    data_set = data_set.reshape((-1, data_set.shape[2]))
    data_label = data_label.reshape(-1)

    data_set = data_set[data_label != 0]
    data_label = data_label[data_label != 0]

    average = np.zeros([global_define.LabelSize, global_define.DataSize])

    for i in range(1, global_define.LabelSize):
        data = data_set[data_label == i]

        data_num = data.shape[0]
        data = data.astype(np.float32)
        extract_nums = list(range(data_num))
        extract_subs = random.sample(extract_nums, global_define.TrainNumPerClass)
        data_extracted = data[extract_subs, :]
        data[extract_subs] = np.inf
        data = data[data[:, 0] != np.inf]

        train_set = preprocessing.minmax_scale(data_extracted, axis=1)
        np.save("data/train_set_%d.npy" % i, train_set)

        for p in range(global_define.DataSize):
            average[i, p] = train_set[:, p].mean()

        train_set = torch.from_numpy(train_set).type(torch.FloatTensor)
        label = torch.zeros([data_extracted.shape[0], 1]).type(torch.LongTensor) + i

        train_set = torch_data.TensorDataset(
            data_tensor=train_set,
            target_tensor=label
        )
        torch.save(train_set, 'data/train_data_set_%d.lib' % i)
        test_set = preprocessing.minmax_scale(data, axis=1)

        np.save("data/test_set_%d.npy" % i, test_set)

    np.save("data/average.npy", average)



if __name__ == "__main__":
    init_data()


