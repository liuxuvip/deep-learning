import torch
from torch.autograd import Variable
import numpy as np
import network_1 as nw
import SVM
import global_define
import matplotlib.pyplot as plt


# data_processor = dp.DataProcessor()
# data_processor.init_data()
# get training data and test data
average = np.load("data/average.npy")
train_set = np.load("data/train_set.npy")
train_set_label = np.load("data/train_set_label.npy").astype(np.int64)

test_set = np.load("data/test_set.npy")
test_set_label = np.load("data/test_set_label.npy")

print(train_set.shape, test_set.shape)

SVM0 = SVM.SVM(model_id=0)
SVM0.train(train_set, train_set_label)
res = SVM0.test(test_set, test_set_label)
print('using original data only result is', res)


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

# for running_label in range(1, global_define.LabelSize):
#     # get generative data
#     G = nw.Generator()
#     G.load_state_dict(torch.load('results/G_state_' + global_define.run_version + "_%d" % running_label + '.pkl'))
#
#     noise = torch.randn((global_define.TrainNumPerClass, global_define.G_NoiseSize))
#     noise = Variable(noise)
#
#     y = G(noise).data.numpy()
#     fake_data = np.concatenate((fake_data, y), axis=0)

threshold = np.array([0, 10, 14, 17, 21, 12, 13, 10, 9, 13, 9, 15, 13, 12, 10, 10, 8])

for running_label in range(1, global_define.LabelSize):
    # get generative data
    G = nw.Generator()
    G.load_state_dict(torch.load('results/G_state_' + global_define.run_version + "_%d" % running_label + '.pkl'))
    i = 0
    print("done ", running_label)
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

fake_label = y_.numpy().reshape(num_data).astype(np.int64)

# get training data and test data
SVM1 = SVM.SVM(model_id=1)
SVM1.train(fake_data, fake_label)
res = SVM1.test(test_set, test_set_label)
print('using fake data only result is', res)
# combine all the data

combined_data = np.concatenate((train_set, fake_data), axis=0)
combined_label = np.concatenate((train_set_label, fake_label), axis=0)

SVM2 = SVM.SVM(model_id=2)
SVM2.train(combined_data, combined_label)
res = SVM2.test(test_set, test_set_label)
print('using both data result is', res)

