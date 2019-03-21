import network as nw
import numpy as np
import torch
from torch.autograd import Variable
import torch.utils.data as torch_data
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import dataProcess
import global_define
import time


def save_module():
    torch.save(D.state_dict(), 'D_state' + global_define.run_version + '.pkl')
    torch.save(G.state_dict(), 'G_state' + global_define.run_version + '.pkl')
    print('save model done')


def show_train_his():
    plt.figure(2)
    plt.plot(G_loss)
    plt.title('G_loss' + global_define.run_version)
    plt.savefig('G_loss' + global_define.run_version + '.jpg')

    plt.figure(3)
    plt.plot(D_loss)
    plt.title('D_loss' + global_define.run_version)
    plt.savefig('D_loss' + global_define.run_version + '.jpg')
    plt.show()

    plt.figure(4)
    plt.plot(likelihood_his)
    plt.title('likelihood_his' + global_define.run_version)
    plt.savefig('likelihood_his' + global_define.run_version + '.jpg')
    plt.show()


def on_press(event):
    save_module()
    show_train_his()
    exit(0)


# network
D = nw.CNN3D().cuda()
G = nw.DeCNN3D().cuda()

G.weight_init(0.0, 0.01)

train_set = torch.load('./data/train_set.lib')
train_data = torch_data.DataLoader(
    train_set,
    batch_size=global_define.BatchSize,
    shuffle=True,
    num_workers=2,
)

# optimizer
G_LearnRate = 1e-4
D_LearnRate = 1e-4
G_optimizer = optim.Adam(G.parameters(), lr=G_LearnRate, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=D_LearnRate, betas=(0.5, 0.999))

# loss function
BCE_loss = nn.BCELoss().cuda()

# record the trace of training
D_loss = []
G_loss = []
likelihood_his = []

# train total epoch
Epoch = 200


def start_training():
    # training
    start_time = time.clock()
    # fig = plt.figure(1)
    plt.ion()
    plt.show()
    num_data = global_define.TrainNumPerClass*global_define.LabelSize
    x_axis = np.array(range(global_define.DataSize))
    for epoch in range(Epoch):
        for step, (real_data, label) in enumerate(train_data):
            mini_batch_size = real_data.size()[0]
            # generate fake data
            y_real = torch.ones(mini_batch_size)
            y_fake = torch.zeros(mini_batch_size)
            label_one_hot = torch.zeros([mini_batch_size, global_define.LabelSize])
            label_one_hot.scatter_(1, label.unsqueeze(1), 1)

            label_one_hot, real_data, y_real, y_fake = \
                Variable(label_one_hot.cuda()), Variable(real_data.cuda()), \
                Variable(y_real.cuda()), Variable(y_fake.cuda())
            # get D result
            D_result = D(real_data, label_one_hot).squeeze()
            # calculate loss of real data of D
            D_real_loss = BCE_loss(D_result, y_real)

            noise = torch.rand((mini_batch_size, global_define.G_NoiseSize))
            # because "0-label" means nothing, so do not generate 0-label
            y_ = (torch.rand(mini_batch_size, 1)*100).type(torch.LongTensor) % (global_define.LabelSize - 1) + 1
            y_label = torch.zeros(mini_batch_size, global_define.LabelSize)
            y_label.scatter_(1, y_.view(mini_batch_size, 1), 1)

            noise, y_label = Variable(noise.cuda()), Variable(y_label.cuda())

            G_result = G(noise, y_label)
            D_result = D(G_result, y_label).squeeze()

            D_fake_loss = BCE_loss(D_result, y_fake)
            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            D_optimizer.step()

            # optimize generator
            noise = torch.rand((mini_batch_size, global_define.G_NoiseSize))
            y_ = (torch.rand(mini_batch_size, 1) * 100).type(torch.LongTensor) \
                 % (global_define.LabelSize - 1) + 1
            y_label = torch.zeros(mini_batch_size, global_define.LabelSize)
            y_label.scatter_(1, y_.view(mini_batch_size, 1), 1)

            noise, y_label = Variable(noise.cuda()), Variable(y_label.cuda())

            G_result = G(noise, y_label)
            D_result = D(G_result, y_label).squeeze()
            G_train_loss = BCE_loss(D_result, y_real)
            # if epoch % 2 == 0:
            G_train_loss.backward()
            G_optimizer.step()

        now_D_loss = D_train_loss.cpu().data.numpy()[0]
        now_G_loss = G_train_loss.cpu().data.numpy()[0]
        D_loss.append(now_D_loss)
        G_loss.append(now_G_loss)

        # # test the performance of generator
        noise = torch.rand((num_data, global_define.G_NoiseSize))

        y_ = torch.zeros((num_data, 1)).type(torch.LongTensor)
        start = 0
        for i in range(1, global_define.LabelSize):
            y_[start:(start + global_define.TrainNumPerClass)] = i
            start += global_define.TrainNumPerClass

        y_label = torch.zeros(num_data, global_define.LabelSize)
        y_label.scatter_(1, y_.view(num_data, 1), 1)
        noise, y_label = Variable(noise).cuda(), Variable(y_label).cuda()

        fake_data = G(noise, y_label)

        fake_data = fake_data.cpu().data.numpy()

        # likelihood = np.sqrt(np.abs(train_set_np - fake_data)).sum(axis=1)
        # max_dislike = likelihood.max()
        # min_dislike = likelihood.min()
        # likelihood = likelihood.mean()
        # likelihood_his.append(likelihood)
        a = np.random.randint(low=0, high=num_data)
        #
        # print('epoch: %3d | D_loss: %.4f | G_loss %.4f | likelihood %.4f, max %.4f, min %.4f'
        #       % (epoch, now_D_loss, now_G_loss, likelihood, max_dislike, min_dislike))

        # if epoch % 4 == 0:
        #     plt.cla()
        #     plt.title('label is %d epoch: %3d | D_loss: %.4f | G_loss %.4f | likelihood %.4f'
        #               % (train_set_label_np[a], epoch, now_D_loss, now_G_loss, likelihood))
        #     plt.plot(x_axis, train_set_np[a], 'black', x_axis, fake_data[a], 'red')
        #     # plt.scatter()
        #
        #     cid = fig.canvas.mpl_connect('key_press_event', on_press)
        #
        #     plt.axis([0, dp.DataSize, 0, 1])
        #     plt.pause(0.0001)
        if epoch % 1 == 0:
            plt.cla()
            plt.title('epoch: %3d | D_loss: %.4f | G_loss %.4f'
                      % (epoch, now_D_loss, now_G_loss))
            plt.plot(fake_data[a, 0, :, 2, 2])
            # plt.scatter()

            # cid = fig.canvas.mpl_connect('key_press_event', on_press)

            # plt.axis([0, dp.DataSize, 0, 1])
            plt.pause(0.0001)
        print('epoch: %3d | D_loss: %.4f | G_loss %.4f' % (epoch, now_D_loss, now_G_loss))

    plt.ioff()

    # training end
    ent_time = time.clock()
    print('total running time is %ds' % (ent_time - start_time))

    # save_module()
    # show_train_his()


if __name__ == '__main__':
    start_training()
