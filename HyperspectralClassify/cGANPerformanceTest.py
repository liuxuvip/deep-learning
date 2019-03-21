import torch
from torch.autograd import Variable
import numpy as np
import NeuralNetwork_2_2 as NN
import SVM
import data_processor
import GlobalDefine
from sklearn import preprocessing


dp = data_processor.DataProcessor()
dp.init_data(train_size=50)

# get training data and test data
train_set = np.load("train_set.npy")
train_set_label = np.load("train_set_label.npy").astype(np.int64)

test_set = np.load("test_set.npy")
test_set_label = np.load("test_set_label.npy")

print(train_set.shape, test_set.shape)

SVM0 = SVM.SVM(model_id=0)
SVM0.train(train_set, train_set_label)
res = SVM0.test(test_set, test_set_label)
print('using original data only result is', res)


# get generative data
G = NN.Generator()
G.load_state_dict(torch.load('G_state' + GlobalDefine.run_version + '.pkl'))

num_data = dp.TrainNumPerClass*dp.LabelSize
noise = torch.rand((num_data, NN.GeneratorNoiseDim))

y_ = torch.zeros((num_data, 1)).type(torch.LongTensor)
start = 0
for i in range(dp.LabelSize):
    y_[start:(start + dp.TrainNumPerClass)] = i
    start += dp.TrainNumPerClass

y_label = torch.zeros(num_data, NN.LabelSize)
y_label.scatter_(1, y_.view(num_data, 1), 1)
noise, y_label = Variable(noise), Variable(y_label)

fake_data = G(noise, y_label)
fake_data = fake_data.data.numpy()
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
