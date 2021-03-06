import torch
import torch.nn as nn
import torch.nn.functional as torch_f
import torch.optim as optim
import torch.utils.data as torch_data
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import time

Data_set_mat_filename = u'Hyperspectral Image/Indian_pines/Indian_pines_corrected.mat'
Data_label_mat_filename = u'Hyperspectral Image/Indian_pines/Indian_pines_gt.mat'

Data_set_mat = sio.loadmat(Data_set_mat_filename)['indian_pines_corrected']
Data_label_mat = sio.loadmat(Data_label_mat_filename)['indian_pines_gt']


max_data_set = np.max(Data_set_mat)
Data_set_mat = Data_set_mat.reshape([Data_set_mat.shape[0]*Data_set_mat.shape[1], Data_set_mat.shape[2]])
Data_label_mat = Data_label_mat.reshape(Data_label_mat.shape[0]*Data_label_mat.shape[1])

Data_set = torch_data.TensorDataset(
    data_tensor=torch.FloatTensor(Data_set_mat.tolist()),
    target_tensor=torch.from_numpy(Data_label_mat).type(torch.LongTensor)
)

GeneratorNoiseDim = 100
GeneratorOutputSize = Data_set_mat.shape[1]   # the size of true data
GeneratorInputLayerOutputSize = 256
GeneratorCombineLayerSize = 512
GeneratorHiddenLayerSize = 512
LabelSize = int(np.max(Data_label_mat)) + 1


class Generator(nn.Module):
    # initializer
    def __init__(self):
        super(Generator, self).__init__()
        # input layer
        self.noise_input_layer = nn.Linear(GeneratorNoiseDim, GeneratorInputLayerOutputSize)
        self.noise_input_layer_norm = nn.BatchNorm1d(GeneratorInputLayerOutputSize)
        self.label_input_layer = nn.Linear(LabelSize, GeneratorInputLayerOutputSize)
        self.label_input_layer_norm = nn.BatchNorm1d(GeneratorInputLayerOutputSize)
        self.combine_layer = nn.Linear(GeneratorInputLayerOutputSize*2, GeneratorCombineLayerSize)
        self.combine_layer_norm = nn.BatchNorm1d(GeneratorCombineLayerSize)
        self.hidden_layer = nn.Linear(GeneratorCombineLayerSize, GeneratorHiddenLayerSize)
        self.hidden_layer_norm = nn.BatchNorm1d(GeneratorHiddenLayerSize)
        self.output_layer = nn.Linear(GeneratorHiddenLayerSize, GeneratorOutputSize)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, noise_input, label):
        x = torch_f.relu(self.noise_input_layer_norm(self.noise_input_layer(noise_input)))
        y = torch_f.relu(self.label_input_layer_norm(self.label_input_layer(label)))
        x = torch.cat([x, y], 1)
        x = torch_f.relu(self.combine_layer_norm(self.combine_layer(x)))
        x = torch_f.relu(self.hidden_layer_norm(self.hidden_layer(x)))
        x = torch_f.relu(self.output_layer(x))
        # x = torch_f.sigmoid(x)
        return x


DiscriminatorInputSize = GeneratorOutputSize
DiscriminatorInputLayerOutputSize = 512
DiscriminatorCombineLayerSize = 256
DiscriminatorHiddenLayerSize = 128


class Discriminator(nn.Module):
    # initializer
    def __init__(self):
        super(Discriminator, self).__init__()
        self.data_input_layer = nn.Linear(GeneratorOutputSize, DiscriminatorInputLayerOutputSize)
        self.label_input_layer = nn.Linear(LabelSize, DiscriminatorInputLayerOutputSize)
        self.combine_layer = nn.Linear(DiscriminatorInputLayerOutputSize*2, DiscriminatorCombineLayerSize)
        self.combine_layer_norm = nn.BatchNorm1d(DiscriminatorCombineLayerSize)
        self.hidden_layer = nn.Linear(DiscriminatorCombineLayerSize, DiscriminatorHiddenLayerSize)
        self.hidden_layer_norm = nn.BatchNorm1d(DiscriminatorHiddenLayerSize)
        self.output_layer = nn.Linear(DiscriminatorHiddenLayerSize, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, data_input, label):
        x = torch_f.leaky_relu(self.data_input_layer(data_input), 0.2)
        y = torch_f.leaky_relu(self.label_input_layer(label), 0.2)
        x = torch.cat([x, y], 1)
        x = torch_f.leaky_relu(self.combine_layer_norm(self.combine_layer(x)), 0.2)
        x = torch_f.leaky_relu(self.hidden_layer_norm(self.hidden_layer(x)), 0.2)
        x = torch_f.sigmoid(self.output_layer(x))
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


WithGpu = True
torch.manual_seed(0)
if WithGpu:
    torch.cuda.manual_seed(0)

# network
if WithGpu:
    G = Generator().cuda()
    D = Discriminator().cuda()
else:
    G = Generator()
    D = Discriminator()

G.weight_init(mean=0, std=1)
D.weight_init(mean=0, std=1)

# optimizer
G_LearnRate = 2e-3
D_LearnRate = 2e-4
G_optimizer = optim.Adam(G.parameters(), lr=G_LearnRate, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=D_LearnRate, betas=(0.5, 0.999))

# load train data
BatchSize = 50

data_loader = torch_data.DataLoader(
    Data_set,
    batch_size=BatchSize,
    shuffle=True,
    num_workers=2,
)

# loss function
if WithGpu:
    BCE_loss = nn.BCELoss().cuda()
else:
    BCE_loss = nn.BCELoss()

# record the trace of training
D_loss = []
G_loss = []

# training
Epoch = 100
start_time = time.clock()
for epoch in range(Epoch):
    # learning rate decay
    if epoch == 5:
        G_optimizer.param_groups[0]['lr'] /= 5
        D_optimizer.param_groups[0]['lr'] /= 5
        print("learning rate change!")
    elif epoch == 10:
        G_optimizer.param_groups[0]['lr'] /= 5
        D_optimizer.param_groups[0]['lr'] /= 5
        print("G lr: %f, D lr: %f" % (G_optimizer.param_groups[0]['lr'], D_optimizer.param_groups[0]['lr']))
    elif epoch == 15:
        G_optimizer.param_groups[0]['lr'] /= 5
        D_optimizer.param_groups[0]['lr'] /= 5
        print("G lr: %f, D lr: %f" % (G_optimizer.param_groups[0]['lr'], D_optimizer.param_groups[0]['lr']))
    elif epoch == 20:
        G_optimizer.param_groups[0]['lr'] /= 5
        D_optimizer.param_groups[0]['lr'] /= 5
        print("G lr: %f, D lr: %f" % (G_optimizer.param_groups[0]['lr'], D_optimizer.param_groups[0]['lr']))
    elif epoch == 30:
        G_optimizer.param_groups[0]['lr'] /= 5
        D_optimizer.param_groups[0]['lr'] /= 5
        print("G lr: %f, D lr: %f" % (G_optimizer.param_groups[0]['lr'], D_optimizer.param_groups[0]['lr']))
    elif epoch == 40:
        G_optimizer.param_groups[0]['lr'] /= 5
        D_optimizer.param_groups[0]['lr'] /= 5
        print("G lr: %f, D lr: %f" % (G_optimizer.param_groups[0]['lr'], D_optimizer.param_groups[0]['lr']))

    for step, (real_data, label) in enumerate(data_loader):
        mini_batch_size = real_data.size()[0]
        # generate fake data
        y_real = torch.ones(mini_batch_size)
        y_fake = torch.zeros(mini_batch_size)
        label_oneHot = torch.zeros([mini_batch_size, LabelSize])
        label_oneHot.scatter_(1, label.unsqueeze(1), 1)

        if WithGpu:
            label_oneHot, real_data, y_real, y_fake = \
                Variable(label_oneHot.cuda()), Variable(real_data.cuda()), \
                Variable(y_real.cuda()), Variable(y_fake.cuda())
        else:
            label_oneHot, real_data, y_real, y_fake = \
                Variable(label_oneHot), Variable(real_data), \
                Variable(y_real), Variable(y_fake)

        D_result = D(real_data, label_oneHot).squeeze()
        D_real_loss = BCE_loss(D_result, y_real)

        noise = torch.rand((mini_batch_size, GeneratorNoiseDim))
        y_ = (torch.rand(mini_batch_size, 1)*LabelSize % LabelSize).type(torch.LongTensor)
        y_label = torch.zeros(mini_batch_size, LabelSize)
        y_label.scatter_(1, y_.view(mini_batch_size, 1), 1)

        if WithGpu:
            noise, y_label = \
                Variable(noise.cuda()), Variable(y_label.cuda())
        else:
            noise, y_label = \
                Variable(noise), Variable(y_label)

        G_result = G(noise, y_label)

        D_result = D(G_result, y_label).squeeze()
        D_fake_loss = BCE_loss(D_result, y_fake)
        D_fake_score = D_result.data.mean()

        D_train_loss = D_real_loss + D_fake_loss
        D_train_loss.backward()
        D_optimizer.step()

        G_optimizer.zero_grad()

        noise = torch.rand((mini_batch_size, GeneratorNoiseDim))
        y_ = (torch.rand(mini_batch_size, 1) * LabelSize % LabelSize).type(torch.LongTensor)
        y_label = torch.zeros(mini_batch_size, LabelSize)
        y_label.scatter_(1, y_.view(mini_batch_size, 1), 1)

        if WithGpu:
            noise, y_label = \
                Variable(noise.cuda()), Variable(y_label.cuda())
        else:
            noise, y_label = \
                Variable(noise), Variable(y_label)

        G_result = G(noise, y_label)
        D_result = D(G_result, y_label).squeeze()
        G_train_loss = BCE_loss(D_result, y_real)

        if epoch < 10 and D_train_loss < 0.6:
            G_train_loss.backward()
            G_optimizer.step()
        elif D_train_loss < 0.5:
            G_train_loss.backward()
            G_optimizer.step()

    if epoch % 1 == 0:
        if WithGpu:
            now_D_loss = D_train_loss.cpu().data.numpy()[0]
            now_G_loss = G_train_loss.cpu().data.numpy()[0]
            D_loss.append(now_D_loss)
            G_loss.append(now_G_loss)
            print('epoch: %3d | D_loss: %.4f | G_loss %.4f'
                  % (epoch, now_D_loss, now_G_loss))
        else:
            now_D_loss = D_train_loss.data.numpy()[0]
            now_G_loss = G_train_loss.data.numpy()[0]
            D_loss.append(now_D_loss)
            G_loss.append(now_G_loss)
            print('epoch: %3d | D_loss: %.4f | G_loss %.4f'
                  % (epoch, now_D_loss, now_G_loss))

# training end
ent_time = time.clock()
print('total running time is %ds' % (ent_time - start_time))

# save the module
torch.save(D.state_dict(), 'D_state.pkl')
torch.save(G.state_dict(), 'G_state.pkl')
print('training done')

plt.figure(1)
plt.plot(G_loss)
plt.title("G_loss")

plt.figure(2)
plt.plot(D_loss)
plt.title("D_loss")
plt.show()




