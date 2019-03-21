import torch.optim as optim
import torch.utils.data as torch_data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
from NeuralNetwork_2_2 import *
import numpy as np
import GlobalDefine


IfInitial = True
dp = data_processor.DataProcessor()
if IfInitial:
    dp.init_data()
    print('initial done!')

# get training data and test data
train_set_np = np.load("train_set.npy")
train_set_label_np = np.load("train_set_label.npy")

test_set_np = np.load("test_set.npy")
test_set_label_np = np.load("test_set_label.npy")

# network
G = Generator().cuda()
D = Discriminator().cuda()

G.weight_init(mean=0, std=0.01)
D.weight_init(mean=0, std=0.01)

# load train data
BatchSize = 32

train_set = torch.load('train_set.lib')
train_data = torch_data.DataLoader(
    train_set,
    batch_size=BatchSize,
    shuffle=True,
    num_workers=2,
)

# optimizer
G_LearnRate = 1e-4
D_LearnRate = 2e-4
G_optimizer = optim.Adam(G.parameters(), lr=G_LearnRate, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=D_LearnRate, betas=(0.5, 0.999))
#
# G_optimizer = optim.Adamax(G.parameters(), lr=G_LearnRate, betas=(0.5, 0.999))
# D_optimizer = optim.Adamax(D.parameters(), lr=D_LearnRate, betas=(0.5, 0.999))

# loss function
BCE_loss = nn.BCELoss().cuda()

# record the trace of training
D_loss = []
G_loss = []
likelihood_his = []


def on_press(event):
    # save the module
    torch.save(D.state_dict(), 'D_state' + GlobalDefine.run_version + '.pkl')
    torch.save(G.state_dict(), 'G_state' + GlobalDefine.run_version + '.pkl')
    print('training done')

    plt.figure(2)
    plt.plot(G_loss)
    plt.title('G_loss' + GlobalDefine.run_version)
    plt.savefig('G_loss' + GlobalDefine.run_version + '.jpg')

    plt.figure(3)
    plt.plot(D_loss)
    plt.title('D_loss' + GlobalDefine.run_version)
    plt.savefig('D_loss' + GlobalDefine.run_version + '.jpg')
    plt.show()

    plt.figure(4)
    plt.plot(likelihood_his)
    plt.title('likelihood_his' + GlobalDefine.run_version)
    plt.savefig('likelihood_his' + GlobalDefine.run_version + '.jpg')
    plt.show()
    plt.pause(10)
    exit(0)


def start_training():
    # training
    Epoch = 300
    start_time = time.clock()
    fig = plt.figure(1)
    plt.ion()
    plt.show()
    num_data = dp.TrainNumPerClass * dp.LabelSize
    x_axis = np.array(range(dp.DataSize))
    for epoch in range(Epoch):
        # # learning rate decay
        if epoch == 10:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("G lr: %f, D lr: %f" % (G_optimizer.param_groups[0]['lr'], D_optimizer.param_groups[0]['lr']))
        if epoch == 30:
            G_optimizer.param_groups[0]['lr'] /= 10
            D_optimizer.param_groups[0]['lr'] /= 10
            print("G lr: %f, D lr: %f" % (G_optimizer.param_groups[0]['lr'], D_optimizer.param_groups[0]['lr']))
        # elif epoch == 50:
        #     G_optimizer.param_groups[0]['lr'] /= 10
        #     D_optimizer.param_groups[0]['lr'] /= 2
        #     print("G lr: %f, D lr: %f" % (G_optimizer.param_groups[0]['lr'], D_optimizer.param_groups[0]['lr']))
        # elif epoch == 80:
        #     G_optimizer.param_groups[0]['lr'] /= 5
        #     # D_optimizer.param_groups[0]['lr'] /= 20
        #     print("G lr: %f, D lr: %f" % (G_optimizer.param_groups[0]['lr'], D_optimizer.param_groups[0]['lr']))
        elif epoch == 120:
            G_optimizer.param_groups[0]['lr'] /= 7
            D_optimizer.param_groups[0]['lr'] /= 7
            print("G lr: %f, D lr: %f" % (G_optimizer.param_groups[0]['lr'], D_optimizer.param_groups[0]['lr']))

        for step, (real_data, label) in enumerate(train_data):
            mini_batch_size = real_data.size()[0]
            # generate fake data
            y_real = torch.ones(mini_batch_size)
            y_fake = torch.zeros(mini_batch_size)
            label_oneHot = torch.zeros([mini_batch_size, LabelSize])
            label_oneHot.scatter_(1, label.unsqueeze(1), 1)

            label_oneHot, real_data, y_real, y_fake = \
                Variable(label_oneHot.cuda()), Variable(real_data.cuda()), \
                Variable(y_real.cuda()), Variable(y_fake.cuda())

            D_result = D(real_data, label_oneHot).squeeze()

            D_real_loss = BCE_loss(D_result, y_real)

            noise = torch.rand((mini_batch_size, GeneratorNoiseDim))
            y_ = (torch.rand(mini_batch_size, 1) * LabelSize % LabelSize).type(torch.LongTensor)
            y_label = torch.zeros(mini_batch_size, LabelSize)
            y_label.scatter_(1, y_.view(mini_batch_size, 1), 1)

            noise, y_label = Variable(noise.cuda()), Variable(y_label.cuda())

            G_result = G(noise, y_label)

            D_result = D(G_result, y_label).squeeze()
            D_fake_loss = BCE_loss(D_result, y_fake)
            D_train_loss = D_real_loss + D_fake_loss
            D_train_loss.backward()
            D_optimizer.step()

            if (epoch < 50) or (50 <= epoch < 120 and (step % 2 == 0)) or (120 <= epoch and (step % 3 == 0)):
                G_optimizer.zero_grad()

                noise = torch.rand((mini_batch_size, GeneratorNoiseDim))
                y_ = (torch.rand(mini_batch_size, 1) * LabelSize % LabelSize).type(torch.LongTensor)
                y_label = torch.zeros(mini_batch_size, LabelSize)
                y_label.scatter_(1, y_.view(mini_batch_size, 1), 1)

                noise, y_label = Variable(noise.cuda()), Variable(y_label.cuda())

                G_result = G(noise, y_label)
                D_result = D(G_result, y_label).squeeze()
                G_train_loss = BCE_loss(D_result, y_real)
                G_train_loss.backward()

                G_optimizer.step()

        now_D_loss = D_train_loss.cpu().data.numpy()[0]
        now_G_loss = G_train_loss.cpu().data.numpy()[0]
        D_loss.append(now_D_loss)
        G_loss.append(now_G_loss)

        noise = torch.rand((num_data, GeneratorNoiseDim))

        y_ = torch.zeros((num_data, 1)).type(torch.LongTensor)
        start = 0
        for i in range(dp.LabelSize):
            y_[start:(start + dp.TrainNumPerClass)] = i
            start += dp.TrainNumPerClass

        y_label = torch.zeros(num_data, LabelSize)
        y_label.scatter_(1, y_.view(num_data, 1), 1)
        noise, y_label = Variable(noise).cuda(), Variable(y_label).cuda()

        fake_data = G(noise, y_label)

        fake_data = fake_data.cpu().data.numpy()

        likelihood = np.sqrt(np.abs(train_set_np - fake_data)).sum(axis=1)
        max_dislike = likelihood.max()
        min_dislike = likelihood.min()
        likelihood = likelihood.mean()
        likelihood_his.append(likelihood)
        a = np.random.randint(low=0, high=num_data)

        print('epoch: %3d | D_loss: %.4f | G_loss %.4f | likelihood %.4f, max %.4f, min %.4f'
              % (epoch, now_D_loss, now_G_loss, likelihood, max_dislike, min_dislike))

        if epoch % 1 == 0:
            plt.cla()
            plt.title('label is %d epoch: %3d | D_loss: %.4f | G_loss %.4f | likelihood %.4f'
                      % (train_set_label_np[a], epoch, now_D_loss, now_G_loss, likelihood))
            plt.plot(x_axis, train_set_np[a], 'black', x_axis, fake_data[a], 'red')
            # plt.scatter()

            cid = fig.canvas.mpl_connect('key_press_event', on_press)

            plt.axis([0, dp.DataSize, 0, 1])
            plt.pause(0.0001)

    plt.ioff()

    # training end
    ent_time = time.clock()
    print('total running time is %ds' % (ent_time - start_time))

    # save the module
    torch.save(D.state_dict(), 'D_state' + GlobalDefine.run_version + '.pkl')
    torch.save(G.state_dict(), 'G_state' + GlobalDefine.run_version + '.pkl')
    print('training done')

    plt.figure(2)
    plt.plot(G_loss)
    plt.title('G_loss' + GlobalDefine.run_version)
    plt.savefig('G_loss' + GlobalDefine.run_version + '.jpg')

    plt.figure(3)
    plt.plot(D_loss)
    plt.title('D_loss' + GlobalDefine.run_version)
    plt.savefig('D_loss' + GlobalDefine.run_version + '.jpg')
    plt.show()

    plt.figure(4)
    plt.plot(likelihood_his)
    plt.title('likelihood_his' + GlobalDefine.run_version)
    plt.savefig('likelihood_his' + GlobalDefine.run_version + '.jpg')
    plt.show()


if __name__ == '__main__':
    start_training()
