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


def lr_decrease(l):
    x0 = 0.1
    x1 = 0.2
    x2 = 0.8
    lr0 = 2e-6
    lr1 = 2e-5
    lr2 = 2e-4
    if l > x2:
        return lr2
    elif l > x1:
        return float((lr2 - lr1)/(x2 - x1)*(l - x1) + lr1)
    elif l > x0:
        return 5*lr0
    else:
        return lr0


IfInitial = False
if IfInitial:
    data_processor.init_data()
    print('initial done!')

# get training data and test data
train_set = torch.load("data/train_data_set.lib")
test_set_data = np.load("data/test_set_data.npy")
test_set_label = np.load("data/test_set_label.npy")

test_set_number = test_set_data.shape[0]

test_set_data = torch.from_numpy(test_set_data).type(torch.FloatTensor)
test_set_data = Variable(test_set_data)

# network
net = network.DenseNet().cuda()
# continue to learn
# net.load_state_dict(torch.load("data/net_state.pkl"))

# net.weight_init(mean=0, std=0.02)

# optimizer
LearnRate = 1e-4
optimizer = optim.RMSprop(net.parameters(), lr=LearnRate)

# load train data
BatchSize = global_define.BatchSize

train_data = torch_data.DataLoader(
    train_set,
    batch_size=BatchSize,
    shuffle=True,
    num_workers=2,
)

# loss function
CELoss = nn.CrossEntropyLoss().cuda()

# record the trace of training
Loss_his = []
Accuracy_his = []

# training
Epoch = 50
start_time = time.clock()
# num_data = global_define.TrainNumPerClass * global_define.LabelSize
# x_axis = np.array(range(global_define.DataSize[2]))
for epoch in range(Epoch):
    # if epoch == 5:
    #     optimizer.param_groups[0]['lr'] /= 10

    for step, (data, label) in enumerate(train_data):
        mini_batch_size = data.size()[0]
        data = Variable(data).cuda()
        predict = net(data)

        label = label - 1
        label = Variable(label).cuda()
        optimizer.zero_grad()   # clear gradients for next train
        loss = CELoss(predict, label)

        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

        l = loss.cpu().data.numpy()
        # lr = lr_decrease(l)
        # optimizer.param_groups[0]['lr'] = lr
        if(step % 5 == 0):
            Loss_his.append(l)
    print("epoch: %3d | loss: %.4f" % (epoch, l))

    if epoch % 10 == 9:
        i = 1
        correct_sum = 0
        test_batch = 500
        while i * test_batch < test_set_number:
            test_output = net(test_set_data[(i - 1) * test_batch:i * test_batch].cuda())
            test_output = torch.max(test_output, 1)[1].cpu().data.numpy().reshape(-1)
            correct_sum += np.sum(test_output == (test_set_label[(i - 1) * test_batch:i * test_batch] - 1))
            i += 1

        accuracy = correct_sum / (test_set_number - test_set_number % test_batch)
        Accuracy_his.append(accuracy)
        print("accuracy: %.2f" % accuracy)
        net = net.cuda()

# training end
ent_time = time.clock()
print('total running time is %ds' % (ent_time - start_time))

# save the module
torch.save(net.state_dict(), "data/net_state.pkl")
print('training done')

plt.figure(2)
plt.plot(Loss_his)
plt.title("Loss_his")

plt.figure(3)
plt.plot(Accuracy_his)
plt.title("Accuracy his")
plt.show()
