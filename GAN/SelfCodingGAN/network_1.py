# self coding neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
import global_define


def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Generator(nn.Module):
    # initializer
    def __init__(self):
        super(Generator, self).__init__()
        # input layer
        self.fc1 = nn.Linear(global_define.G_NoiseSize, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, global_define.DataSize)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, noise_input):
        x = self.fc1(noise_input)
        x = self.fc1_bn(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)

        x = self.fc3(x)
        x = self.fc3_bn(x)
        x = F.relu(x)

        x = self.fc4(x)
        # x = F.relu(x)
        x = F.sigmoid(x)
        x = x.view(-1, global_define.DataSize)
        return x


class Discriminator(nn.Module):
    # initializer
    def __init__(self):
        super(Discriminator, self).__init__()
        # layers
        self.fc1 = nn.Linear(global_define.DataSize, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 256)
        self.fc2_bn = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, data_input):
        x = self.fc1(data_input)
        # x = self.fc1_bn(x)
        x = F.leaky_relu(x)

        x = self.fc2(x)
        # x = self.fc2_bn(x)
        x = F.leaky_relu(x)
        #
        # x = self.fc3(x)
        # x = self.fc3_bn(x)
        # x = F.leaky_relu(x)

        x = self.fc4(x)
        x = F.sigmoid(x)
        x = x.view(-1)
        return x
