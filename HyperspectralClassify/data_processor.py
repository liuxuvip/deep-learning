import torch
import torch.utils.data as torch_data
import numpy as np
import scipy.io as sio


# prameters
LabelSize = 16
TrainNumPerClass = 200
DataSize = 204


def init_data():
    # read data set
    data_set_mat_filename = u'Hyperspectral Image/salina/Salinas_corrected.mat'
    data_label_mat_filename = u'Hyperspectral Image/salina/Salinas_gt.mat'

    # tract data
    data_set_mat = sio.loadmat(data_set_mat_filename)['salinas_corrected']
    data_label_mat = sio.loadmat(data_label_mat_filename)['salinas_gt']

    # remove all the zeros
    data_set_no_zeros = []
    data_label_no_zeros = []
    for i in range(data_label_mat.shape[0]):
        for j in range(data_label_mat.shape[1]):
            if data_label_mat[i][j] != 0:
                data_set_no_zeros.append(data_set_mat[i][j])
                data_label_no_zeros.append(data_label_mat[i][j])

    # classify data
    data_classified = list(range(LabelSize))
    for i in range(len(data_classified)):
        data_classified[i] = []

    for i in range(len(data_set_no_zeros)):
        data_classified[data_label_no_zeros[i] - 1].append(data_set_no_zeros[i])

    # tract train data set
    train_set = list(range(LabelSize))
    for i in range(len(train_set)):
        train_set[i] = []

    for i in range(len(data_classified)):
        for j in range(TrainNumPerClass):
            train_set[i].append(data_classified[i].pop())

    # generate train label
    train_label = np.zeros(LabelSize * TrainNumPerClass)
    for i in range(LabelSize):
        train_label[i * TrainNumPerClass:(i * TrainNumPerClass + TrainNumPerClass)] = i + 1

    train_set = np.array(train_set)
    train_set = train_set.reshape([train_set.shape[0] * train_set.shape[1], train_set.shape[2]])

    # train_set normalization
    train_mean = train_set.mean()
    train_std_variance = np.sqrt(train_set.var())
    train_set = (train_set - train_mean) / train_std_variance

    train_set = torch.from_numpy(train_set).type(torch.FloatTensor)
    train_label = torch.from_numpy(train_label).type(torch.LongTensor)

    train_set = torch_data.TensorDataset(
        data_tensor=train_set,
        target_tensor=train_label
    )

    torch.save(train_set, 'train_set.lib')

    test_label = []
    # generate test data set
    for i in range(len(data_set_no_zeros)):
        for j in range(len(data_set_no_zeros[i])):
            test_label.append(i + 1)

    test_set = []
    for i in range(len(data_set_no_zeros)):
        for j in range(len(data_set_no_zeros[i])):
            test_set.append(data_set_no_zeros[i][j])

    test_label = np.array(test_label)
    test_set = np.array(test_set)

    test_mean = test_set.mean()
    test_std_variance = np.sqrt(test_set.var())
    test_set = (test_set - test_mean) / test_std_variance

    test_set = torch.from_numpy(test_set).type(torch.FloatTensor)
    test_label = torch.from_numpy(test_label).type(torch.LongTensor)

    test_set = torch_data.TensorDataset(
        data_tensor=test_set,
        target_tensor=test_label
    )

    torch.save(test_set, 'test_set.lib')



