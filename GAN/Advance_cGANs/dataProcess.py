import train_test_split
import numpy as np
import torch
import torch.utils.data as torch_data
import os


def load_data(initial=False, with_test=False):
    if initial:
        y_train, train_index, x_train, y_test, test_index, x_test = \
            train_test_split.split_dataset_by_nums(train_nums=50)

        x_train = x_train[:, None, :, :, :, 0]

        y_train = y_train.flatten()
        train_index = train_index.flatten()

        x_test = x_test[:, None, :, :, :, 0]
        y_test = y_test.flatten()
        test_index = test_index.flatten()

        # create train data set in tensor
        x_train_tensor = torch.from_numpy(x_train).type(torch.FloatTensor)
        y_train_tensor = torch.from_numpy(y_train).type(torch.LongTensor)

        train_set = torch_data.TensorDataset(
            data_tensor=x_train_tensor,
            target_tensor=y_train_tensor
        )

        torch.save(train_set, './data/train_set.lib')

        np.save('./data/x_train.npy', x_train)
        np.save('./data/y_train.npy', y_train)
        np.save('./data/train_index.npy', train_index)

        if with_test:
            np.save('./data/x_test.npy', x_test)
            np.save('./data/y_test.npy', y_test)
            np.save('./data/test_index.npy', test_index)
            return x_train, y_train, train_index, x_test, y_test, test_index
        else:
            return x_train, y_train, train_index

    else:
        x_train = np.load('./data/x_train.npy')
        y_train = np.load('./data/y_train.npy')
        train_index = np.load('./data/train_index.npy')

        if with_test:
            x_test = np.load('./data/x_test.npy')
            y_test = np.load('./data/y_test.npy')
            test_index = np.load('./data/test_index.npy')
            return x_train, y_train, train_index, x_test, y_test, test_index
        else:
            return x_train, y_train, train_index


if __name__ == '__main__':
    load_data(initial=False, with_test=False)
    print('loda data done')
    # clear caches in memory
    os.system('sudo sysctl -w vm.drop_caches=3')
