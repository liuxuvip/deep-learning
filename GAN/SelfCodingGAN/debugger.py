import torch
from torch.autograd import Variable
import numpy as np
import network_1 as nw
import global_define
import matplotlib.pyplot as plt


average = np.load("data/average.npy")

train_set = np.load("data/train_set.npy")
train_set_label = np.load("data/train_set_label.npy").astype(np.int64)

test_set = np.load("data/test_set.npy")
test_set_label = np.load("data/test_set_label.npy")

print(train_set.shape, test_set.shape)

num_data = global_define.TrainNumPerClass * (global_define.LabelSize - 1)
y_ = torch.zeros((num_data, 1)).type(torch.LongTensor)
start = 0
for i in range(1, global_define.LabelSize):
    y_[start:(start + global_define.TrainNumPerClass)] = i
    start += global_define.TrainNumPerClass

y_label = torch.zeros(num_data, global_define.LabelSize)
y_label.scatter_(1, y_.view(num_data, 1), 1)
y_label = Variable(y_label)
fake_data = np.zeros((0, global_define.DataSize))


threshold = np.array([0, 10, 14, 17, 21, 12, 13, 10, 9, 13, 9, 15, 13, 12, 10, 10, 8])

for running_label in range(1, global_define.LabelSize):
    # get generative data
    G = nw.Generator()
    G.load_state_dict(torch.load('results/G_state_' + global_define.run_version + "_%d" % running_label + '.pkl'))
    i = 0
    while i < global_define.TrainNumPerClass:
        noise = torch.randn(5, global_define.G_NoiseSize)
        noise = Variable(noise)

        y = G(noise)[0].view(1, -1)
        y = y.data.numpy()
        # print(average[running_label].shape)
        # print(np.abs(y - average[running_label]).sum())
        if np.abs(y - average[running_label]).sum() < threshold[running_label]:
            fake_data = np.concatenate((fake_data, y), axis=0)
            i += 1

        # print(np.abs(y))

    if running_label == 16:
        plt.figure(1)
        plt.plot(fake_data[4, :])
        # plt.show()

fake_label = y_.numpy().reshape(num_data).astype(np.int64)


plt.figure(2)
plt.imshow(fake_data, cmap='gray')
plt.figure(3)
plt.imshow(train_set, cmap='gray')
plt.show()

