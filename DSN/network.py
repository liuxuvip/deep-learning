import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.Conv3d_0 = nn.Conv3d(1, 24, (1, 1, 1))
        self.Conv3d_0_bn = nn.BatchNorm3d(24)
        self.Conv3d_1 = nn.Conv3d(24, 24, (1, 1, 7), (1, 1, 2))
        self.Conv3d_1_bn = nn.BatchNorm3d(24)
        self.Conv3d_2 = nn.Conv3d(24, 24, (1, 1, 7), (1, 1, 2))
        self.Conv3d_2_bn = nn.BatchNorm3d(24)
        self.Conv3d_3 = nn.Conv3d(24, 24, (1, 1, 7), (1, 1, 2))
        self.Conv3d_3_bn = nn.BatchNorm3d(24)
        self.Pool = nn.AvgPool3d((7, 7, 1))
        self.Fc = nn.Linear(24*675, 16)

    def forward(self, input_x):
        input_x = input_x.view(-1, 1, 7, 7, 204)
        # print(input_x.size())
        input_x = self.Conv3d_0(input_x)
        input_x = self.Conv3d_0_bn(input_x)
        input_x = F.relu(input_x)
        # print(input_x.size())

        x_1 = self.Conv3d_1(input_x)
        x_1 = self.Conv3d_1_bn(x_1)
        x_1 = F.relu(x_1)
        # print(x_1.size())

        x_2 = torch.cat((input_x, x_1), dim=4)
        # print(x_2.size())

        x_2 = self.Conv3d_2(x_2)
        x_2 = self.Conv3d_2_bn(x_2)
        x_2 = F.relu(x_2)
        # print(x_2.size())

        x_3 = torch.cat((input_x, x_1, x_2), dim=4)
        # print(x_3.size())

        x_3 = self.Conv3d_3(x_3)
        x_3 = self.Conv3d_3_bn(x_3)
        x_3 = F.relu(x_3)

        x_4 = torch.cat((input_x, x_1, x_2, x_3), dim=4)
        x_4 = self.Pool(x_4)

        # print(x_4.size())
        x_4 = x_4.view(-1, 24*675)
        out = self.Fc(x_4)
        return out


if __name__ == "__main__":
    net = DenseNet()
    x = torch.randn((3, 1, 5, 5, 204))
    x = Variable(x)
    y = net(x)
