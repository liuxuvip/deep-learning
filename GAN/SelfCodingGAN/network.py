# self coding neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
import global_define
from torch.autograd import Variable
import numpy as np


average = np.load("data/average.npy")
ChannelBase_noise = int(global_define.G_NoiseSize/2)


class Generator(nn.Module):
    """
    in_channels, out_channels, kernel_size, stride=1, padding=0

    :math:`L_{out} = (L_{in} - 1) * stride - 2 * padding + kernel\_size + output\_padding`
    """
    # initializer
    def __init__(self):
        super(Generator, self).__init__()
        # input layer
        # self.noise_layer = nn.Linear(global_define.G_NoiseSize, 128)
        # self.noise_layer_bn = nn.BatchNorm1d(128)
        self.deconv1 = nn.ConvTranspose1d(
            in_channels=ChannelBase_noise*2,
            out_channels=ChannelBase_noise,
            kernel_size=15)
        self.deconv2 = nn.ConvTranspose1d(
            in_channels=ChannelBase_noise,
            out_channels=1,
            stride=5,
            padding=1,
            kernel_size=15)

        self.fc1 = nn.Linear(83 + 204, global_define.DataSize)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    def forward(self, noise, real):
        x = noise.view(-1, global_define.G_NoiseSize, 1)
        x = self.deconv1(x)
        x = F.leaky_relu(x)
        x = self.deconv2(x)
        x = F.leaky_relu(x)
        x = x.view(-1, 83)

        x = torch.cat((x, real), dim=1)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = x.view(-1, global_define.DataSize)
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Discriminator(nn.Module):
    # initializer
    def __init__(self):
        super(Discriminator, self).__init__()
        """
        in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
        """
        # self.data_layer = nn.Linear(global_define.DataSize, 128)
        self.conv1 = nn.Conv1d(
            in_channels=2,
            out_channels=ChannelBase_noise,
            kernel_size=15,
            stride=5)
        self.conv2 = nn.Conv1d(
            in_channels=ChannelBase_noise,
            out_channels=ChannelBase_noise*2,
            kernel_size=15,
            stride=1)

        self.size = ChannelBase_noise*2*24
        self.fc1 = nn.Linear(self.size, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    def forward(self, fake, real):
        # x = self.data_layer(input)
        # x = F.leaky_relu(x, 0.2)
        x = fake.view(-1, 1, global_define.DataSize)
        y = real.view(-1, 1, global_define.DataSize)
        x = torch.cat((x, y), dim=1)
        x = self.conv1(x)
        x = F.leaky_relu(x)
        x = self.conv2(x)
        x = F.leaky_relu(x)

        # print(x.size())
        x = x.view(-1, self.size)
        x = F.sigmoid(self.fc1(x))
        x = x.view(-1, 1)
        return x


# net = Generator()
# noise = torch.randn((3, global_define.G_NoiseSize))
#
# noise, average = Variable(noise), Variable(torch.from_numpy(average[1, :].repeat(3).reshape(3, -1))
#                                            .type(torch.FloatTensor))
# y = net(noise, average)
# print(y.size())


kernel_size = 32
GeneratorNoiseDim = global_define.G_NoiseSize
GeneratorOutputSize = global_define.DataSize   # the size of true data
LabelSize = global_define.LabelSize
#
#
# class Generator(nn.Module):
#     # initializer
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.kernel_size = kernel_size
#         self.size = 204 - self.kernel_size + 1
#         # input layer
#         self.fc1 = nn.Linear(GeneratorNoiseDim, 128)
#         self.fc1_bn = nn.BatchNorm1d(128)
#         self.fc2 = nn.Linear(128, self.size)
#         self.fc2_bn = nn.BatchNorm1d(self.size)
#         self.deconv1 = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=self.kernel_size)
#
#     # weight_init
#     def weight_init(self, mean, std):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)
#
#     # forward method
#     def forward(self, noise_input):
#         x = self.fc1(noise_input)
#         x = self.fc1_bn(x)
#         x = F.relu(x)
#
#         x = self.fc2(x)
#         x = self.fc2_bn(x)
#         x = F.relu(x)
#
#         x = x.view(-1, 1, self.size)
#
#         x = self.deconv1(x)
#         x = F.sigmoid(x)
#         x = x.view(-1, global_define.DataSize)
#         return x
#
#
# DiscriminatorInputSize = GeneratorOutputSize
#
#
# class Discriminator(nn.Module):
#     # initializer
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.kernel_size = kernel_size
#         self.size = 204 - self.kernel_size + 1
#         # layers
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=self.kernel_size)
#         self.conv1_bn = nn.BatchNorm1d(1)
#         self.fc3 = nn.Linear(self.size, 128)
#         self.fc3_bn = nn.BatchNorm1d(128)
#         self.fc4 = nn.Linear(128, 1)
#
#     # weight_init
#     def weight_init(self, mean, std):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)
#
#     # forward method
#     def forward(self, data_input):
#         x = data_input.view(-1, 1, global_define.DataSize)
#         x = self.conv1(x)
#         x = self.conv1_bn(x)
#         x = F.relu(x)
#
#         x = x.view(-1, self.size)
#         x = self.fc3(x)
#         # x = self.fc3_bn(x)
#         x = F.relu(x)
#
#         x = self.fc4(x)
#         x = F.sigmoid(x)
#         x = x.view(-1)
#         return x
#

# net = Generator()
# noise = torch.randn((3, global_define.G_NoiseSize))
#
# noise = Variable(noise)
#
# y = net(noise)
# print(y.size())






net = Discriminator()
noise = torch.randn((3, global_define.DataSize))

noise, average = Variable(noise), Variable(torch.from_numpy(average[1, :].repeat(3).reshape(3, -1))
                                           .type(torch.FloatTensor))

y = net(noise, average)
print(y.size())
