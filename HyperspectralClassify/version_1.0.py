import torch
import torch.nn as nn
import torch.nn.functional as torch_f
import torch.optim as optim
import torch.utils.data as torch_data
from torchvision import transforms
import torchvision
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

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

GeneratorNoiseDim = 50
GeneratorOutputSize = Data_set_mat.shape[1]   # the size of true data
GeneratorInputLayerOutputSize = 100
GeneratorHiddenLayerSize = 100
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
        size_after_merge = 2*GeneratorInputLayerOutputSize
        # self.combine_layer = nn.Linear(size_after_merge, size_after_merge)
        # self.combine_layer_norm = nn.BatchNorm1d(size_after_merge)
        self.hidden_layer = nn.Linear(size_after_merge, GeneratorOutputSize)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, noise_input, label):
        x1 = torch_f.relu(self.noise_input_layer_norm(self.noise_input_layer(noise_input)))
        y = torch_f.relu(self.label_input_layer_norm(self.label_input_layer(label)))
        x2 = torch.cat([x1, y], 1)
        # x2_combine = torch_f.relu(self.combine_layer_norm(self.combine_layer(x2)))
        x3 = torch_f.relu(self.hidden_layer(x2))
        x4 = torch_f.tanh(x3)
        return x4


DiscriminatorInputSize = GeneratorOutputSize
DiscriminatorInputLayerOutputSize = 100
DiscriminatorHiddenLayerSize = 100


class Discriminator(nn.Module):
    # initializer
    def __init__(self):
        super(Discriminator, self).__init__()
        self.data_input_layer_norm = nn.BatchNorm1d(GeneratorOutputSize)
        self.data_input_layer = nn.Linear(GeneratorOutputSize, DiscriminatorInputLayerOutputSize)
        self.label_input_layer_norm = nn.BatchNorm1d(LabelSize)
        self.label_input_layer = nn.Linear(LabelSize, DiscriminatorInputLayerOutputSize)
        size_after_merge = 2*DiscriminatorInputLayerOutputSize
        # self.combine_layer = nn.Linear(size_after_merge, size_after_merge)
        # self.combine_layer_norm = nn.BatchNorm1d(size_after_merge)
        self.hidden_layer = nn.Linear(size_after_merge, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, data_input, label):
        x1 = torch_f.leaky_relu(self.data_input_layer(self.data_input_layer_norm(data_input)), 0.2)
        y = torch_f.leaky_relu(self.label_input_layer(self.label_input_layer_norm(label)), 0.2)
        x2 = torch.cat([x1, y], 1)
        # x2_combine = torch_f.leaky_relu(self.combine_layer_norm(self.combine_layer(x2)), 0.2)
        x3 = torch_f.leaky_relu(self.hidden_layer(x2), 0.2)
        x4 = torch_f.sigmoid(x3)
        return x4


def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


WithGpu = True

# network
if WithGpu:
    G = Generator().cuda()
    D = Discriminator().cuda()
else:
    G = Generator()
    D = Discriminator()
G.weight_init(mean=0, std=0.02)
D.weight_init(mean=0, std=0.02)

# optimizer
G_LearnRate = 5e-3
D_LearnRate = 5e-4
G_optimizer = optim.Adam(G.parameters(), lr=G_LearnRate, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=D_LearnRate, betas=(0.5, 0.999))

# load train data
BatchSize = 256

data_loader = torch_data.DataLoader(
    Data_set,
    batch_size=BatchSize,
    shuffle=True,
    num_workers=2,
)

# training
Epoch = 5000
for epoch in range(Epoch):
    # learning rate decay
    if (epoch + 1) == 3:
        G_optimizer.param_groups[0]['lr'] /= 5
        D_optimizer.param_groups[0]['lr'] /= 6
        print("learning rate change!")
    elif (epoch + 1) == 5:
        G_optimizer.param_groups[0]['lr'] /= 3
        D_optimizer.param_groups[0]['lr'] /= 4
        print("learning rate change!")
    elif (epoch + 1) == 20:
        G_optimizer.param_groups[0]['lr'] /= 2
        D_optimizer.param_groups[0]['lr'] /= 2
        print("learning rate change!")

    for step, (real_data, label) in enumerate(data_loader):
        mini_batch_size = real_data.size()[0]
        # generate fake data
        noise = torch.rand([mini_batch_size, GeneratorNoiseDim])
        label_oneHot = torch.zeros([mini_batch_size, LabelSize])
        label_oneHot.scatter_(1, label.unsqueeze(1), 1)

        if WithGpu:
            noise, label_oneHot, real_data = \
                Variable(noise.cuda()), Variable(label_oneHot.cuda()), Variable(real_data.cuda())
        else:
            noise, label_oneHot, real_data = Variable(noise), Variable(label_oneHot), Variable(real_data)

        fake_data = G(noise, label_oneHot)

        D_out_real = D(real_data, label_oneHot)
        D_out_fake = D(fake_data, label_oneHot)

        D_loss = -torch.mean(torch.log(D_out_real) + torch.log(1. - D_out_fake))
        D_optimizer.zero_grad()
        D_loss.backward(retain_variables=True)
        D_optimizer.step()

        # G_loss = torch.mean(torch.log(1. - D_out_fake))
        G_loss = -torch.mean(torch.log(D_out_fake))

        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        if step % 20 == 0:
            if WithGpu:
                print('epoch: %3d | step: %3d | D_loss: %.4f | G_loss %.4f real, fake: %.4f %.4f'
                      % (epoch, step, D_loss.cpu().data.numpy()[0], G_loss.cpu().data.numpy()[0],
                         D_out_real.cpu().data.numpy()[0], D_out_fake.cpu().data.numpy()[0]))
            else:
                print('epoch: %3d | step: %3d | D_loss: %.4f | G_loss %.4f'
                      % (epoch, step, D_loss.data.numpy()[0], G_loss.data.numpy()[0]))

