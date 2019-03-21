import torch
import torch.nn as nn
import torch.nn.functional as torch_f
import torch.optim as optim
import torch.utils.data as torch_data
from torchvision import transforms
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


def my_scatter(x, index):
    # y_label.scatter_(1, y_.view(mini_batch_size, 1), 1)
    if x.size()[0] != index.size()[0]:
        print('error size_1')
        return []
    else:
        for i in range(x.size()[0]):
            if index[i] >= x.size()[1]:
                print('error size_2')
                return []
            else:
                x[i][index[i]] = 1
    return x



x = torch.zeros(5, 6)
y = torch.LongTensor([1, 2, 5, 0, 3])

z = my_scatter(x, y)
print(x, y, z)
