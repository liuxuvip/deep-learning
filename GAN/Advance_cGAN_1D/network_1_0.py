import torch
import torch.nn as nn
import torch.nn.functional as F
import global_define
from torch.autograd import Variable

#
class Generator(nn.Module):
    """
    in_channels, out_channels, kernel_size, stride=1, padding=0

    :math:`L_{out} = (L_{in} - 1) * stride - 2 * padding + kernel\_size + output\_padding`
    """
    # initializers
    def __init__(self, channel=64):
        super(Generator, self).__init__()
        self.deconv1_noise = nn.ConvTranspose1d(global_define.G_NoiseSize, channel*4, 25, 1, 0)
        self.deconv1_noise_bn = nn.BatchNorm1d(channel*4)
        self.deconv1_label = nn.ConvTranspose1d(global_define.LabelSize, channel*4, 25, 1, 0)
        self.deconv1_label_bn = nn.BatchNorm1d(channel*4)
        self.deconv2 = nn.ConvTranspose1d(channel*8, channel*4, 20, 1, 1)
        self.deconv2_bn = nn.BatchNorm1d(channel*4)
        self.deconv3 = nn.ConvTranspose1d(channel*4, channel*2, 15, 2, 0)
        self.deconv3_bn = nn.BatchNorm1d(channel*2)
        self.deconv4 = nn.ConvTranspose1d(channel*2, channel, 8, 2, 0)
        self.deconv4_bn = nn.BatchNorm1d(channel)
        self.deconv5 = nn.ConvTranspose1d(channel, 1, 5, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    # def forward(self, input):
    def forward(self, input, label):
        input = input.view(-1, global_define.G_NoiseSize, 1)
        label = label.view(-1, global_define.LabelSize, 1)
        x = F.leaky_relu(self.deconv1_noise(input), 0.2)
        y = F.leaky_relu(self.deconv1_label(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.deconv2(x), 0.2)
        x = F.leaky_relu(self.deconv3(x), 0.2)
        x = F.leaky_relu(self.deconv4(x), 0.2)
        x = F.leaky_relu(self.deconv5(x), 0.2)
        x = F.leaky_relu(x, 0.2)
        x = x.view(-1, global_define.DataSize)
        # x = F.sigmoid(x)
        return x
#

# class Generator(nn.Module):
#     # initializer
#     def __init__(self, kernel_size=15):
#         super(Generator, self).__init__()
#         # input layer
#         self.noise_input_layer = nn.Linear(global_define.G_NoiseSize, 256)
#         self.label_input_layer = nn.Linear(global_define.LabelSize, 128)
#         self.combine_layer = nn.Linear(256 + 128, global_define.DataSize)
#         self.conv = nn.Conv1d(1, 20, kernel_size)
#         self.deconv = nn.ConvTranspose1d(20, 1, kernel_size)
#
#     # weight_init
#     def weight_init(self, mean, std):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)
#
#     # forward method
#     def forward(self, noise_input, label):
#         x = F.relu(self.noise_input_layer(noise_input))
#         y = F.relu(self.label_input_layer(label))
#         x = torch.cat([x, y], dim=1)
#         x = F.relu(self.combine_layer(x))
#         x = x.view(-1, 1, global_define.DataSize)
#         x = F.relu(self.conv(x))
#         x = F.sigmoid(self.deconv(x))
#         x = x.view(-1, global_define.DataSize)
#         return x


class Discriminator(nn.Module):
    # initializer
    def __init__(self):
        super(Discriminator, self).__init__()
        self.data_input_layer = nn.Linear(global_define.DataSize, 256)
        self.label_input_layer = nn.Linear(global_define.LabelSize, 256)
        self.combine_layer = nn.Linear(256 + 256, 256)
        self.combine_layer_norm = nn.BatchNorm1d(256)
        # self.hidden_layer = nn.Linear(256, 256)
        # self.hidden_layer_norm = nn.BatchNorm1d(256)
        self.output_layer = nn.Linear(256, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, data_input, label):
        x = F.relu(self.data_input_layer(data_input))
        y = F.relu(self.label_input_layer(label))
        x = torch.cat([x, y], 1)
        x = F.relu(self.combine_layer_norm(self.combine_layer(x)))
        # x = F.leaky_relu(self.hidden_layer_norm(self.hidden_layer(x)), 0.2)
        x = F.sigmoid(self.output_layer(x))
        return x

#
# class Discriminator(nn.Module):
#     # initializer
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         """
#         in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1
#         """
#         channel = 64
#         self.conv1_input = nn.Conv1d(in_channels=1, out_channels=channel,
#                                      kernel_size=27, stride=5, padding=0, dilation=5)
#         self.conv1_label = nn.Conv1d(1, 16, 3, 1, 0)
#         self.conv2 = nn.Conv1d(channel + 16, channel*2, 8)
#         self.conv2_bn = nn.BatchNorm1d(channel*2)
#         self.conv3 = nn.Conv1d(channel*2, channel*4, 4)
#         self.conv3_bn = nn.BatchNorm1d(channel*4)
#         self.conv4 = nn.Conv1d(channel*4, channel*4, 3)
#         self.conv4_bn = nn.BatchNorm1d(channel*4)
#         self.conv5 = nn.Conv1d(channel*4, 1, 3)
#
#     # weight_init
#     def weight_init(self, mean, std):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)
#
#     # forward method
#     # def forward(self, input):
#     def forward(self, input, label):
#         input = input.view(-1, 1, global_define.DataSize)
#         label = label.view(-1, 1, global_define.LabelSize)
#         x = F.leaky_relu(self.conv1_input(input), 0.2)
#         y = F.leaky_relu(self.conv1_label(label), 0.2)
#         x = torch.cat([x, y], 1)
#         x = F.leaky_relu(self.conv2(x), 0.2)
#         x = F.leaky_relu(self.conv3(x), 0.2)
#         x = F.leaky_relu(self.conv4(x), 0.2)
#         x = F.sigmoid(self.conv5(x))
#         x = x.view(-1, 1)
#         return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# net = Generator(channel=64)
# noise = torch.randn((3, global_define.G_NoiseSize))
# label = torch.zeros((3, global_define.LabelSize))
#
# noise, label = Variable(noise), Variable(label)
# y = net(noise, label)
# print(y.size())

#
# net = Discriminator()
# noise = torch.randn((3, global_define.DataSize))
# label = torch.zeros((3, global_define.LabelSize))
#
# noise, label = Variable(noise), Variable(label)
#
# y = net(noise, label)
