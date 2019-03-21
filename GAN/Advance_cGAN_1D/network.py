import torch
import torch.nn as nn
import torch.nn.functional as F
import global_define
from torch.autograd import Variable


class Generator(nn.Module):
    """
    in_channels, out_channels, kernel_size, stride=1, padding=0

    :math:`L_{out} = (L_{in} - 1) * stride - 2 * padding + kernel\_size + output\_padding`
    """
    # initializers
    def __init__(self):
        super(Generator, self).__init__()
        # input layer
        self.noise_layer = nn.Linear(global_define.G_NoiseSize, 128)
        self.noise_layer_bn = nn.BatchNorm1d(128)
        self.label_layer = nn.Linear(global_define.LabelSize, 128)
        self.label_layer_bn = nn.BatchNorm1d(128)
        self.deconv1 = nn.ConvTranspose1d(
            in_channels=2,
            out_channels=20,
            kernel_size=38)
        self.deconv1_bn = nn.BatchNorm1d(40)
        self.deconv2 = nn.ConvTranspose1d(
            in_channels=20,
            out_channels=1,
            kernel_size=40)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    def forward(self, noise, label):
        x = self.noise_layer(noise)
        # x = self.noise_layer_bn(x)
        x = F.relu(x)
        y = self.label_layer(label)
        # y = self.label_layer_bn(y)
        y = F.relu(y)
        x = x.view(-1, 1, 128)
        y = y.view(-1, 1, 128)
        x = torch.cat([x, y], 1)
        x = self.deconv1(x)
        # x = self.deconv1_bn(x)
        x = F.relu(x)
        x = self.deconv2(x)
        # x = F.relu(x)
        x = F.sigmoid(x)
        x = x.view(-1, global_define.DataSize)
        return x


class Discriminator(nn.Module):
    # initializer
    def __init__(self):
        super(Discriminator, self).__init__()
        """
        in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
        """
        self.data_layer = nn.Linear(global_define.DataSize, 128)
        self.label_layer = nn.Linear(global_define.LabelSize, 128)
        self.conv2 = nn.Conv1d(
            in_channels=2,
            out_channels=16,
            kernel_size=20,
            stride=2)
        self.conv2_bn = nn.BatchNorm1d(10)
        self.conv3 = nn.Conv1d(
            in_channels=16,
            out_channels=32,
            kernel_size=10,
            stride=2)
        self.conv3_bn = nn.BatchNorm1d(20)
        self.size = 32*23
        self.out_layer = nn.Linear(self.size, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    def forward(self, input, label):
        x = self.data_layer(input)
        x = F.leaky_relu(x, 0.2)
        y = self.label_layer(label)
        y = F.leaky_relu(y, 0.2)
        x = x.view(-1, 1, 128)
        y = y.view(-1, 1, 128)
        x = torch.cat([x, y], 1)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.conv3(x)
        x = F.leaky_relu(x, 0.2)
        x = x.view(-1, self.size)
        # print(x.size())
        x = F.sigmoid(self.out_layer(x))
        # print(x.size())
        x = x.view(-1, 1)
        return x


# class Discriminator(nn.Module):
#     # initializer
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.data_input_layer = nn.Linear(global_define.DataSize, 256)
#         self.label_input_layer = nn.Linear(global_define.LabelSize, 256)
#         self.combine_layer = nn.Linear(256 + 256, 256)
#         self.combine_layer_norm = nn.BatchNorm1d(256)
#         # self.hidden_layer = nn.Linear(256, 256)
#         # self.hidden_layer_norm = nn.BatchNorm1d(256)
#         self.output_layer = nn.Linear(256, 1)
#
#     # weight_init
#     def weight_init(self, mean, std):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)
#
#     # forward method
#     def forward(self, data_input, label):
#         x = F.relu(self.data_input_layer(data_input))
#         y = F.relu(self.label_input_layer(label))
#         x = torch.cat([x, y], 1)
#         x = F.relu(self.combine_layer_norm(self.combine_layer(x)))
#         # x = F.leaky_relu(self.hidden_layer_norm(self.hidden_layer(x)), 0.2)
#         x = F.sigmoid(self.output_layer(x))
#         return x

#

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# net = Generator()
# noise = torch.randn((3, global_define.G_NoiseSize))
# label = torch.zeros((3, global_define.LabelSize))
#
# noise, label = Variable(noise), Variable(label)
# y = net(noise, label)
# print(y.size())


net = Discriminator()
noise = torch.randn((3, global_define.DataSize))
label = torch.zeros((3, global_define.LabelSize))

noise, label = Variable(noise), Variable(label)

y = net(noise, label)
print(y.size())
