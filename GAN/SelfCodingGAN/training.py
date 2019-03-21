import torch.optim as optim
import torch.utils.data as torch_data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import numpy as np
import global_define
import network_1 as nw
import data_processor
import torch
import torch.nn as nn
import math


def start_training(running_label):
    average_original = np.load("data/average.npy")[running_label, :]

    IfInitial = True
    if IfInitial:
        data_processor.init_data()
        print('initial done!')

    # get training data and test data
    train_set_np = np.load("data/train_set_%d.npy" % running_label)
    test_set_np = np.load("data/test_set_%d.npy" % running_label)

    # network
    G = nw.Generator().cuda()
    D = nw.Discriminator().cuda()

    G.weight_init(mean=0, std=0.007)
    D.weight_init(mean=0, std=0.007)

    # load train data
    BatchSize = 32

    train_set = torch.load('data/train_data_set_%d.lib' % running_label)
    train_data = torch_data.DataLoader(
        train_set,
        batch_size=BatchSize,
        shuffle=True,
        num_workers=2,
    )

    # optimizer
    G_LearnRate = 5e-4
    D_LearnRate = 5e-4
    G_optimizer = optim.Adam(G.parameters(), lr=G_LearnRate, betas=(0.5, 0.999))
    D_optimizer = optim.Adam(D.parameters(), lr=D_LearnRate, betas=(0.5, 0.999))

    # loss function
    BCE_loss = nn.BCELoss().cuda()

    # record the trace of training
    D_loss = []
    G_loss = []
    likelihood_his = []

    Epoch = 1000
    start_time = time.clock()
    fig = plt.figure(1)
    plt.ion()
    plt.show()
    x_axis = np.array(range(global_define.DataSize))
    for epoch in range(Epoch):
        for step, (real_data, label) in enumerate(train_data):
            mini_batch_size = real_data.size()[0]
            # generate fake data
            y_real = torch.ones(mini_batch_size)
            y_fake = torch.zeros(mini_batch_size)

            average = Variable(torch.from_numpy(average_original.repeat(mini_batch_size).
                                                reshape(mini_batch_size, -1)).type(torch.FloatTensor)).cuda()

            real_data, y_real, y_fake = \
                Variable(real_data.cuda()), \
                Variable(y_real.cuda()), Variable(y_fake.cuda())

            D_result = D(real_data)
            D_real_loss = BCE_loss(D_result, y_real)
            noise = torch.randn((mini_batch_size, global_define.G_NoiseSize))

            noise = Variable(noise.cuda())

            G_result = G(noise)
            D_result = D(G_result)
            D_fake_loss = BCE_loss(D_result, y_fake)
            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()

            D_optimizer.step()

            if (step < 50 and step % 2 == 0) or (step >= 50 and step % 5 == 0):
                G_optimizer.zero_grad()

                noise = torch.randn((mini_batch_size, global_define.G_NoiseSize))
                noise = Variable(noise.cuda())

                G_result = G(noise)
                D_result = D(G_result)
                G_train_loss = BCE_loss(D_result, y_real)
                G_train_loss.backward()
                # print(G.fc1.weight.abs().mean())
                G_optimizer.step()

        now_D_loss = D_train_loss.cpu().data.numpy()[0]
        now_G_loss = G_train_loss.cpu().data.numpy()[0]
        D_loss.append(now_D_loss)
        G_loss.append(now_G_loss)

        noise = torch.randn((global_define.TrainNumPerClass, global_define.G_NoiseSize))
        noise = Variable(noise).cuda()
        average = Variable(torch.from_numpy(average_original.repeat(global_define.TrainNumPerClass).
                                            reshape(global_define.TrainNumPerClass, -1)).type(torch.FloatTensor)).cuda()

        fake_data = G(noise)

        fake_data = fake_data.cpu().data.numpy()

        likelihood = (np.abs((train_set_np - fake_data))).sum(axis=1)

        max_dislike = likelihood.max()
        min_dislike = likelihood.min()
        likelihood = likelihood.mean()
        likelihood_his.append(likelihood)
        a = np.random.randint(low=0, high=global_define.TrainNumPerClass)

        print('epoch: %3d | D_loss: %.4f | G_loss %.4f | likelihood %.4f, max %.4f, min %.4f'
              % (epoch, now_D_loss, now_G_loss, likelihood, max_dislike, min_dislike))
        if likelihood < 40:
            break
        lr0 = 2e-5
        lr1 = 3e-4
        x_0 = 50
        x_1 = 110

        lr = pow(10, likelihood*math.log10(lr1/lr0)/(x_1-x_0) + (x_1*math.log10(lr0) - x_0*math.log10(lr1))/(x_1 - x_0))
        print("learn rate : ", lr)
        G_optimizer.param_groups[0]['lr'] = lr
        D_optimizer.param_groups[0]['lr'] = lr
        if epoch % 4 == 0:
            plt.cla()
            plt.title('label is %d epoch: %3d | D_loss: %.4f | G_loss %.4f | likelihood %.4f'
                      % (running_label, epoch, now_D_loss, now_G_loss, likelihood))
            plt.plot(x_axis, train_set_np[a], 'black', x_axis, fake_data[a], 'red')

            plt.axis([0, global_define.DataSize, 0, 1])
            plt.pause(0.0001)

    plt.ioff()

    # training end
    ent_time = time.clock()
    print('total running time is %ds' % (ent_time - start_time))

    # save the module
    torch.save(D.state_dict(), 'results/D_state_' + global_define.run_version + "_%d" % running_label + '.pkl')
    torch.save(G.state_dict(), 'results/G_state_' + global_define.run_version + "_%d" % running_label + '.pkl')
    print('training done')

    plt.figure(2)
    plt.cla()
    plt.plot(G_loss)
    plt.title("G_loss_" + global_define.run_version + "_%d" % running_label)
    plt.savefig("results/G_loss_" + global_define.run_version + "_%d" % running_label + '.jpg')

    plt.figure(3)
    plt.cla()
    plt.plot(D_loss)
    plt.title("D_loss_" + global_define.run_version + "_%d" % running_label)
    plt.savefig("results/D_loss_" + global_define.run_version + "_%d" % running_label + '.jpg')

    plt.figure(4)
    plt.cla()
    plt.plot(likelihood_his)
    plt.title('likelihood_his_' + global_define.run_version + "_%d" % running_label)
    plt.savefig('results/likelihood_his_' + global_define.run_version + "_%d" % running_label + '.jpg')


if __name__ == '__main__':
    for i in range(1, global_define.LabelSize):
        start_training(i)
