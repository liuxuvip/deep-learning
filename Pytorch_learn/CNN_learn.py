import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import time
import os


BatchSize = 50
TrainingEpoch = 2
WithGpu = True
LearnRate = 0.001
IsTrain = True


# get training data
train_dataset = torchvision.datasets.MNIST(
    root='data/mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
print(train_dataset.train_data.size())
print(train_dataset.train_labels.size())

test_dataset = torchvision.datasets.MNIST(
    root='data/mnist',
    train=False,
    transform=torchvision.transforms.ToTensor()
)

# print(test_dataset.test_data.size())
# print(test_dataset.test_labels.size())


# generate my convolution neural network
class CnnPicture(nn.Module):
    def __init__(self):
        super(CnnPicture, self).__init__()
        # input data size: [batch_size*1*28*28]
        self.convolution_layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,      # data input number of layer
                out_channels=16,    # out put data number of layer
                kernel_size=5,      # size of window
                stride=1,           # number of pix for one move
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2
            )
        )
        # input data size: [batch_size*16*14*14]
        self.convolution_layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,  # data input number of layer
                out_channels=32,  # out put data number of layer
                kernel_size=5,  # size of window
                stride=1,  # number of pix for one move
                padding=2
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2
            )
        )
        self.out_layer = nn.Linear(32*7*7, out_features=10)

    def forward(self, x):
        x1 = self.convolution_layer1(x)
        x2 = self.convolution_layer2(x1)
        x3 = x2.view(x.size(0), -1)
        x3 = self.out_layer(x3)
        return x3, x2, x1


if IsTrain:
    start = time.clock()
    # initial network
    picture_process_network = CnnPicture()
    loss_function = nn.CrossEntropyLoss()
    if WithGpu:
        picture_process_network = picture_process_network.cuda()
        loss_function = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(picture_process_network.parameters(), lr=LearnRate)

    # initial train data
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BatchSize, shuffle=True, num_workers=2)

    # initial test data
    test_x = Variable(torch.unsqueeze(test_dataset.test_data, dim=1), volatile=True).type(torch.FloatTensor)/255.
    # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
    test_y = test_dataset.test_labels

    # begin training
    for epoch in range(TrainingEpoch):
        for step, (x, y) in enumerate(train_loader):
            if WithGpu:
                train_x = Variable(x).cuda()
                train_y = Variable(y).cuda()
            else:
                train_x = Variable(x)
                train_y = Variable(y)

            out = picture_process_network(train_x)[0]
            loss = loss_function(out, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                # have a test

                if WithGpu:
                    test_x = test_x.cuda()
                    test_out = picture_process_network(test_x)[0]
                    prediction_y = torch.max(test_out, 1)[1].cpu().data.squeeze()
                    accuracy = torch.sum(torch.eq(prediction_y, test_y)) / float(test_y.size(0))
                    print('epoch: %d | step: %d | loss: %.4f | accuracy: %.4f'
                          % (epoch, step, loss.cpu().data[0], accuracy))
                else:
                    test_out = picture_process_network(test_x)[0]
                    prediction_y = torch.max(test_out, 1)[1].data.squeeze()
                    accuracy = torch.sum(torch.eq(prediction_y, test_y)) / float(test_y.size(0))
                    print('epoch: %d | step: %d | loss: %.4f | accuracy: %.4f'
                          % (epoch, step, loss.data[0], accuracy))

    end = time.clock()
    print('total running time is %.1f' % (end - start))
    torch.save(picture_process_network.state_dict(), 'picture_process_network.pkl')
else:
    last_net = CnnPicture()
    last_net.load_state_dict(torch.load('picture_process_network.pkl'))

    picture_in = test_dataset.test_data[0]
    In_data = torch.zeros(1, 1, 28, 28)
    In_data[0][0] = test_dataset.test_data[0]
    In_data = Variable(In_data).type(torch.FloatTensor) / 255.

    plt.imshow(picture_in.numpy(), cmap='gray')
    plt.title('Input number %d' % test_dataset.test_labels[0])

    x = last_net(In_data)
    for x_i in range(1, len(x)):
        for i in range(x[x_i].size()[1]):
            directory = 'output/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.imsave(directory + 'x%d_%d' % (x_i, i), x[x_i][0][i].data.numpy(), cmap='gray')
    plt.show()
