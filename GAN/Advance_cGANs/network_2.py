import torch
import torch.nn as nn
import torch.nn.functional as F
import global_define


def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class CNN1D(nn.Module):
    def __init__(self,
                 input_size=global_define.DataSize,
                 input_label_size=global_define.LabelSize,
                 kernel_size=((7, 3, 3), (3, 3, 3))):
        super(CNN1D, self).__init__()
        # (N, C, D, H, W)
        self.conv1 = nn.Conv1d(1, 2, kernel_size[0])
        self.conv2 = nn.Conv1d(2, 8, kernel_size[1])
        self.fc1 = nn.Linear((input_size - kernel_size[0][0] + 1 - kernel_size[1][0] + 1)*8, 128)
        self.label_layer = nn.Linear(input_label_size, 128)
        self.combine_layer = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x, label):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        y = F.relu(self.label_layer(label))
        x = torch.cat([x, y], dim=1)
        x = F.relu(self.combine_layer(x))
        x = self.fc2(x)
        return F.sigmoid(x)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class DeCNN3D(nn.Module):
    def __init__(self,
                 input_label_size=global_define.LabelSize,
                 kernel_size=((7, 3, 3), (3, 3, 3))
                 ):
        super(DeCNN3D, self).__init__()
        # (N, C, D, H, W)
        self.label_layer = nn.Linear(input_label_size, 128)
        self.noise_layer = nn.Linear(global_define.G_NoiseSize, 128)
        self.combine_layer = nn.Linear(256, 512)
        self.combine_hidden = nn.Linear(512, 1568)
        self.deconv1_3d = nn.ConvTranspose3d(8, 2, kernel_size[1])
        self.deconv2_3d = nn.ConvTranspose3d(2, 1, kernel_size[0])

    def forward(self, noise, label):
        x = F.relu(self.label_layer(label))
        x = torch.cat([x, noise], dim=1)
        x = F.relu(self.combine_layer(x))
        x = F.relu(self.combine_hidden(x))
        assert(x.size()[1] == 1568)
        x = x.view((-1, 8, 196, 1, 1))
        x = F.relu(self.deconv1_3d(x))
        x = F.sigmoid(self.deconv2_3d(x))
        return x

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)
