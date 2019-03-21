import torch.optim as optim
import torch.utils.data as torch_data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import network
import numpy as np
import data_processor
import torch
import global_define
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data.dataloader import DataLoader


x_train = np.load("data/x_train.npy")
y_train = np.load("data/y_train.npy")
x_test = np.load("data/x_test.npy")
y_test = np.load("data/y_test.npy")

print(x_train.dtype, y_test.dtype)
x_train_new = np.zeros((x_train.shape[0], 1, 7, 7, 204), np.float32)
x_test_new = np.zeros((x_test.shape[0], 1, 7, 7, 204), np.float32)
for i in range(x_train.shape[0]):
    x_train_new[i, 0] = x_train[i, 0].transpose()
    x_test_new[i, 0] = x_test[i, 0].transpose()

print(x_train_new.shape, x_test_new.shape)
np.save("data/x_train_new.npy", x_train_new)
np.save("data/x_test_new.npy", x_test_new)

np.save("data/y_train.npy", y_train.astype(np.int32))
np.save("data/y_test.npy", y_test.astype(np.int32))
x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)
train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, batch_size=50, shuffle=True, num_workers=2, pin_memory=True)
torch.save(test_loader, "data/test_loader.lib")
torch.save(train_loader, "data/train_loader.lib")
print(test_loader.dataset.data_tensor.size())
