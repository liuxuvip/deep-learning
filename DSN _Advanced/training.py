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
import torch.nn.functional as F
import sys
import time
import math


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


# IfInitial = False
# if IfInitial:
#     data_processor.init_data()
#     print('initial done!')

# get training data and test data
train_loader = torch.load("data/train_loader.lib")
test_set_data = np.load("data/test_set_data.npy")
test_set_label = np.load("data/test_set_label.npy")
test_loader = torch.load("data/test_loader.lib")
test_set_number = test_set_data.shape[0]
test_set_data = torch.from_numpy(test_set_data).type(torch.FloatTensor)
test_set_data = Variable(test_set_data)

# network
# net = network.ResNet(network.Bottleneck, [3, 4, 6, 3]).cuda()
net = network.DenseNet()#.cuda()
# continue to learn
# net.load_state_dict(torch.load("data/net_state.pkl"))

# net.weight_init(mean=0, std=0.02)

# optimizer
LearnRate = 1e-4
# optimizer = optim.RMSprop(net.parameters(), lr=LearnRate)

optimizer = optim.Adagrad(net.parameters(), lr=0.01, weight_decay=0.0005)

# load train data
# BatchSize = global_define.BatchSize
#
# train_data = torch_data.DataLoader(
#     train_set,
#     batch_size=BatchSize,
#     shuffle=True,
#     num_workers=2,
# )

# loss function
# CELoss = nn.CrossEntropyLoss().cuda()

# record the trace of training
Loss_his = []
Accuracy_his = []

# training
Epoch = 30
start_time = time.clock()
# num_data = global_define.TrainNumPerClass * global_define.LabelSize
# x_axis = np.array(range(global_define.DataSize[2]))
last_accuracy = 0
Net_Changed = False
for epoch in range(Epoch):
    # if epoch == 6:
    #     optimizer.param_groups[0]['lr'] /= 10

    for step, (data, label) in enumerate(train_loader):
        net.train()
        mini_batch_size = data.size()[0]
        data = Variable(data)#.cuda()
        predict = net(data)

        label = Variable(label)#.cuda()
        optimizer.zero_grad()   # clear gradients for next train
        # loss = CELoss(predict, label)
        # print(label.max(), label.min())
        # print(predict.max(), predict.min())
        loss = F.nll_loss(predict, label)

        loss.backward()         # backpropagation, compute gradients
        optimizer.step()        # apply gradients

        # l = loss.cpu().data.numpy()
        l = loss.data.numpy()
        # lr = lr_decrease(l)
        # optimizer.param_groups[0]['lr'] = lr
        if step % 2 == 0:
            if l > 10:
                l = 10
            Loss_his.append(l)
        if step % 10 == 0:
            # use train set to test accuracy
            net.eval()
            predict = net(data)
            pred = predict.data.max(1)[1]
            # accuracy = pred.eq(label.data).cpu().sum()/mini_batch_size*100
            accuracy = pred.eq(label.data).sum()/mini_batch_size*100
            print("epoch: %3d | loss: %.4f | accuracy train: %.2f %%" % (epoch, l, accuracy))
            if accuracy >= last_accuracy - 1e-3:
                torch.save(net.state_dict(), "module/net_state.pkl")
                print("record net")
                Net_Changed = True
                last_accuracy = accuracy
        # print("learn rate is %.6f" % optimizer.param_groups[0]['lr'])

    if epoch % 20 == 0 and Net_Changed:
        test_net = network.DenseNet()#.cuda()
        test_net.load_state_dict(torch.load("module/net_state.pkl"))
        test_net.eval()
        correct_sum = 0
        for step, (data, label) in enumerate(test_loader):
            mini_batch_size = data.size()[0]
            data = Variable(data)#.cuda()
            predict = test_net(data)

            label = Variable(label)#.cuda()

            pred = predict.data.max(1)[1]
            # correct_sum += pred.eq(label.data).cpu().sum()
            correct_sum += pred.eq(label.data).sum()
            if step % 5 == 0:
                sys.stdout.write("\rtest done %.2f %%" % (100*step/1591))
                sys.stdout.flush()
        print("test done 100 %")
        accuracy = correct_sum/test_loader.dataset.data_tensor.size()[0]
        Accuracy_his.append(accuracy)
        print("accuracy on test set: %.2f %%" % (100*accuracy))
        Net_Changed = False

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
