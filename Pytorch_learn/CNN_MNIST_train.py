import torch.cuda
from torch.autograd import Variable
import torchvision
import torch.nn as nn
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import time
import CNN_MNIST_network

version = "results/version-h1-"
retrain = False

BatchSize = 50
TestBatchSize_1 = 2000
TestBatchSize_2 = 500
LearnRate = 5e-4
TrainingEpoch = 8

# get training data
train_dataset = torchvision.datasets.MNIST(
    root='data/mnist',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

test_dataset = torchvision.datasets.MNIST(
    root='data/mnist',
    train=False,
    transform=torchvision.transforms.ToTensor()
)


train_loss_his = []
train_error_his = []
test_loss_his = []
test_error_his = []


if __name__ == "__main__":
    s = {}
    s["conv_1"] = CNN_MNIST_network.ConvSize(1, 5, 3, 1, 0)
    s["conv_2"] = CNN_MNIST_network.ConvSize(5, 10, 3, 1, 0)
    s["conv_3"] = CNN_MNIST_network.ConvSize(10, 20, 3, 1, 0)
    # s["conv_4"] = CNN_MNIST_network.ConvSize(20, 40, 3, 2, 0)
    # s["conv_5"] = CNN_MNIST_network.ConvSize(40, 80, 3, 2, 0)
    # s["fc_1"] = CNN_MNIST_network.FcSize(180, 100)
    s["fc_2"] = CNN_MNIST_network.FcSize(180, 10)

    start = time.clock()

    # initial network
    if retrain:
        cnn_network = torch.load(version+"cnn_network.pkl").cuda()
    else:
        cnn_network = CNN_MNIST_network.CNN_MNIST_1(size=s).cuda()

    print(cnn_network)

    loss_function = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.Adam(cnn_network.parameters(), lr=LearnRate)
    # optimizer = torch.optim.SGD(cnn_network.parameters(), lr=LearnRate)
    # optimizer = torch.optim.RMSprop(cnn_network.parameters(), lr=LearnRate)

    # initial train data
    train_loader_1 = Data.DataLoader(
        dataset=train_dataset,
        batch_size=BatchSize,
        shuffle=True,
        num_workers=2)
    # initial train data
    train_loader_2 = Data.DataLoader(
        dataset=train_dataset,
        batch_size=TestBatchSize_1,
        shuffle=True,
        num_workers=2)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=TestBatchSize_2,
        shuffle=True,
        num_workers=2)

    # # initial test data
    # test_x = Variable(torch.unsqueeze(test_dataset.test_data, dim=1),
    #                   volatile=True).type(torch.FloatTensor) / 255.
    # test_y = test_dataset.test_labels

    # begin training
    for epoch in range(TrainingEpoch):
        # if epoch == 5:
        #     LearnRate /= 5
        # elif epoch == 10:
        #     LearnRate /= 5
        print("learn rate: %.6f" % LearnRate)
        for step, (x, y) in enumerate(train_loader_1):
            cnn_network.train()

            train_x = Variable(x).cuda()
            train_y = Variable(y).cuda()
            # print(train_y)
            out = cnn_network(train_x)
            loss = loss_function(out, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                print('epoch: %d | step: %d | loss: %.4f'
                      % (epoch, step, loss.data))

            if step % 400 == 0:
                train_loss_his.append(loss.data)
                # have a test on train set
                print("have a test on train set...")
                cnn_network.eval()
                accuracy = 0
                for (train_test_x, train_test_y) in train_loader_2:
                    train_test_x = Variable(train_test_x).cuda()
                    train_test_y = train_test_y.cuda()
                    train_test_out = cnn_network(train_test_x).data
                    prediction_y = torch.max(train_test_out, 1)[1].squeeze()
                    accuracy += torch.sum(torch.eq(prediction_y, train_test_y))

                accuracy = accuracy.type(torch.FloatTensor)
                accuracy = accuracy / train_dataset.train_data.size()[0]
                print('epoch: %d | accuracy on train set: %.4f' % (epoch, accuracy))
                train_error_his.append(1 - accuracy)

                # have a test on test set
                print("have a test on test set...")

                cnn_network.eval()
                accuracy = 0
                test_loss = 0
                for step, (test_x, test_y) in enumerate(test_loader):
                    test_x = Variable(test_x).cuda()
                    test_y = Variable(test_y).cuda()

                    test_out = cnn_network(test_x)
                    test_loss += loss_function(test_out, test_y).data
                    test_out = test_out.data
                    test_y = test_y.data
                    prediction_y = torch.max(test_out, 1)[1].squeeze()
                    accuracy = accuracy + torch.sum(torch.eq(prediction_y, test_y))

                accuracy = accuracy.type(torch.FloatTensor)
                accuracy = accuracy/test_dataset.test_data.size()[0]
                test_loss = test_loss.cpu()/step

                print('epoch: %d | accuracy on test set: %.4f | loss: %.4f' % (epoch, accuracy, test_loss))
                test_error_his.append(1 - accuracy)
                test_loss_his.append(test_loss)
                print('\n')

    end = time.clock()
    print('total running time is %.1f' % (end - start))

    torch.save(cnn_network, version+"cnn_network.pkl")

    # print("train_loss_hist", train_loss_his,
    #       "train_error_his", train_error_his,
    #       "test_loss_his", test_loss_his,
    #       "test_error_his", test_error_his)

    train_loss_his = np.array(train_loss_his)
    train_error_his = np.array(train_error_his)
    test_loss_his = np.array(test_loss_his)
    test_error_his = np.array(test_error_his)

    np.save(version+"train_loss_his", train_loss_his)
    np.save(version+"train_error_his", train_error_his)
    np.save(version+"test_loss_his", test_loss_his)
    np.save(version+"test_error_his", test_error_his)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_his, label="train loss")
    plt.plot(test_loss_his, label="test loss")

    plt.legend()  # 显示图例
    plt.xlabel('iteration times')
    plt.ylabel('rate')

    plt.subplot(1, 2, 2)
    plt.plot(train_error_his, label="train error")
    plt.plot(test_error_his, label="test error")

    plt.legend()  # 显示图例
    plt.xlabel('iteration times')
    plt.ylabel('rate')

    plt.tight_layout()
    plt.savefig(version+"result")
    plt.show()
