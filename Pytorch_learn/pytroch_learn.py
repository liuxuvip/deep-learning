# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# import torch
#
#
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input image channel, 6 output channels, 5x5 square convolution kernel
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)  # an affine operation: y = Wx + b
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # Max pooling over a (2, 2) window
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # If the size is a square you can only specify a single number
#         x = x.view(-1, self.num_flat_features(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
#     def num_flat_features(self, x):
#         size = x.size()[1:]  # all dimensions except the batch dimension
#         num_features = 1
#         for s in size:
#             num_features *= s
#         return num_features
#
#
# net = Net()
# print(net)
# params = list(net.parameters())
# print(len(params))
# print(params[0].size())  # conv1's .weight
#
# input = Variable(torch.randn(1, 1, 32, 32))
# out = net(input)
# print(out)
# '''out 的输出结果如下
# Variable containing:
# -0.0158 -0.0682 -0.1239 -0.0136 -0.0645  0.0107 -0.0230 -0.0085  0.1172 -0.0393
# [torch.FloatTensor of size 1x10]
# '''
#
# net.zero_grad()  # 对所有的参数的梯度缓冲区进行归零
# out.backward(torch.randn(1, 10))  # 使用随机的梯度进行反向传播
# output = net(input)
# target = Variable(torch.arange(1, 11))  # a dummy target, for example
# criterion = nn.MSELoss()
# loss = criterion(output, target)
# print(loss)
# '''loss的值如下
# Variable containing:
#  38.5849
# [torch.FloatTensor of size 1]
# '''
# # For illustration, let us follow a few steps backward
# print(loss.creator)  # MSELoss
# print(loss.creator.previous_functions[0][0])  # Linear
# print(loss.creator.previous_functions[0][0].previous_functions[0][0])  # ReLU
#
# '''
# <torch.nn._functions.thnn.auto.MSELoss object at 0x7fe8102dd7c8>
# <torch.nn._functions.linear.Linear object at 0x7fe8102dd708>
# <torch.nn._functions.thnn.auto.Threshold object at 0x7fe8102dd648>
# '''
#
# # 现在我们应当调用loss.backward(), 之后来看看 conv1's在进行反馈之后的偏置梯度如何
# net.zero_grad()  # 归零操作
# print('conv1.bias.grad before backward')
# print(net.conv1.bias.grad)
# loss.backward()
# print('conv1.bias.grad after backward')
# print(net.conv1.bias.grad)
#
# learning_rate = 0.01
# for f in net.parameters():
#     f.data.sub_(f.grad.data * learning_rate)
#
# import torch.optim as optim
#
# # create your optimizer
# optimizer = optim.SGD(net.parameters(), lr=0.01)
#
# # in your training loop:
# optimizer.zero_grad()  # zero the gradient buffers
# output = net(input)
# loss = criterion(output, target)
# loss.backward()
# optimizer.step()  # Does the update
# import torchvision
# import torchvision.transforms as transforms
#
# # torchvision数据集的输出是在[0, 1]范围内的PILImage图片。
# # 我们此处使用归一化的方法将其转化为Tensor，数据范围为[-1, 1]
#
# transform = transforms.Compose([transforms.ToTensor(),
#                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                                 ])
# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
#                                           shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                          shuffle=False, num_workers=2)
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# '''注：这一部分需要下载部分数据集 因此速度可能会有一些慢 同时你会看到这样的输出
#
# Downloading http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
# Extracting tar file
# Done!
# Files already downloaded and verified
# '''
# # functions to show an image
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
#     plt.colorbar()
#     plt.show()
#
#
# # show some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# # print labels
# print(' '.join('%5s' % classes[labels[j]] for j in range(4)))
#
# # print images
# imshow(torchvision.utils.make_grid(images))
