import torch
import torch.utils.data as torch_data
import random
import scipy.io as sio
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt


class DataProcessor:
    def __init__(self):
        # prameters
        self.LabelSize = 16
        self.TrainNumPerClass = 200
        self.DataSize = 204

    def train_test_split(self, data, label):

        # classify data
        data_classified = list(range(self.LabelSize))
        for i in range(len(data_classified)):
            data_classified[i] = []
        # rearrange data by its label
        for i in range(len(data)):
            data_classified[label[i] - 1].append(data[i])

        # from each class, extract 200 random samples
        data_extracted = list(range(self.LabelSize))  # extracted data
        for i in range(len(data_classified)):
            class_i_len = len(data_classified[i])
            data_classified[i] = np.array(data_classified[i]).astype(np.float32)
            extract_nums = list(range(class_i_len))
            extract_subs = random.sample(extract_nums, self.TrainNumPerClass)
            data_extracted[i] = data_classified[i][extract_subs, :]
            data_classified[i][extract_subs] = np.inf
            data_classified[i] = data_classified[i][data_classified[i][:, 0] != np.inf]
        # flatten the list to array
        train_set = np.zeros((self.LabelSize*self.TrainNumPerClass, self.DataSize))
        train_label = np.zeros(self.LabelSize*self.TrainNumPerClass)
        for i in range(self.LabelSize):
            train_set[i * self.TrainNumPerClass:(i*self.TrainNumPerClass + self.TrainNumPerClass)] = data_extracted[i]
            train_label[i * self.TrainNumPerClass:(i*self.TrainNumPerClass + self.TrainNumPerClass)] = i + 1
        # rearrange data randomly
        shuffle_num = list(range(train_set.shape[0]))
        shuffle_num = random.sample(shuffle_num, len(shuffle_num))
        train_set = train_set[shuffle_num]
        train_label = train_label[shuffle_num]

        length = 0
        for i in range(len(data_classified)):
            length += len(data_classified[i])
        test_label = np.zeros(length)
        test_set = []
        start = 0
        for i in range(len(data_classified)):
            for j in range(len(data_classified[i])):
                test_set.append(data_classified[i][j])
            test_label[start:(start + len(data_classified[i]))] = i + 1
            start += len(data_classified[i])

        test_set = np.array(test_set)
        return train_set.astype(np.int32), train_label, test_set.astype(np.int32), test_label

    def init_data(self, train_size=200):
        self.TrainNumPerClass = train_size
        print('TrainNumPerClass is %d, initial data...' % self.TrainNumPerClass)
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

        train_set, train_label, test_set, test_label = \
            self.train_test_split(data_set, data_label)

        train_set = preprocessing.minmax_scale(train_set, axis=1)

        np.save("data/train_set.npy", train_set)
        np.save("data/train_set_label.npy", train_label)

        train_set = torch.from_numpy(train_set).type(torch.FloatTensor)
        train_label = torch.from_numpy(train_label).type(torch.LongTensor)
        train_set = torch_data.TensorDataset(
            data_tensor=train_set,
            target_tensor=train_label
        )
        torch.save(train_set, 'data/train_data_set.lib')
        test_set = preprocessing.minmax_scale(test_set, axis=1)

        np.save("data/test_set.npy", test_set)
        np.save("data/test_set_label.npy", test_label)


if __name__ == "__main__":
    dp = DataProcessor()
    dp.init_data()


