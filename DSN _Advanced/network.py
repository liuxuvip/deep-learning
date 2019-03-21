import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
import math


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes=32, planes=8, stride=1, has_branch1=None):
        super(Bottleneck, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        m['bn1'] = nn.BatchNorm3d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv3d(planes, planes, kernel_size=(1, 1, 3), stride=1, padding=(0, 0, 1), bias=False)
        m['bn2'] = nn.BatchNorm3d(planes)
        m['relu2'] = nn.ReLU(inplace=True)
        m['conv3'] = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        m['bn3'] = nn.BatchNorm3d(planes * 4)
        self.group1 = nn.Sequential(m)

        self.relu= nn.Sequential(nn.ReLU(inplace=True))

        branch1 = OrderedDict()
        branch1['conv1'] = nn.Conv3d(inplanes, planes * 4, kernel_size=1, stride=stride, bias=False)
        branch1['bn1'] = nn.BatchNorm3d(planes * 4)
        self.branch1 = nn.Sequential(branch1)
        self.has_branch1 = has_branch1

    def forward(self, x):
        if self.has_branch1 is not None:
            residual = self.branch1(x)
        else:
            residual = x

        out = self.group1(x) + residual
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, n_blocks, num_classes=16):
        self.inplanes = 32
        super(ResNet, self).__init__()

        m = OrderedDict()
        m['conv1'] = nn.Conv3d(1, 32, kernel_size=(3, 3, 24), stride=(1, 1, 2), padding=(1, 1, 2), bias=False)
        m['bn1'] = nn.BatchNorm3d(32)
        m['relu1'] = nn.ReLU(inplace=True)
        self.group1 = nn.Sequential(m)

        self.layer1 = self._make_layer(block, blocks=n_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, blocks=n_blocks[1])
        self.layer3 = self._make_layer(block, blocks=n_blocks[2])
        self.layer4 = self._make_layer(block, blocks=n_blocks[3])
        # self.layer5 = self._make_layer(block, blocks=n_blocks[4])

        self.avgpool = nn.Sequential(nn.AvgPool3d(kernel_size=(7, 7, 12)))

        self.group2 = nn.Sequential(
            OrderedDict([
                ('fc', nn.Linear(32, num_classes))
            ])
        )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes=8, blocks=4, stride=(1, 1, 2)):
        layers = []
        for i in range(blocks):
            if i == 1:
                layers.append(block(self.inplanes, planes, stride = stride, has_branch1=True))
            else:
                layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, 1, 7, 7, 204)
        x = self.group1(x)
        print(x.data.size())
        x = self.layer1(x)
        print(x.data.size())
        x = self.layer2(x)
        print(x.data.size())
        x = self.layer3(x)
        # print(x.data.size())
        x = self.layer4(x)
        # print(x.data.size())
        # x = self.layer5(x)
        # print(x.data.size())
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(x.data.size())
        x = self.group2(x)

        # return F.log_softmax(x)
        return x


class DenseNet(nn.Module):
    def __init__(self):
        super(DenseNet, self).__init__()
        self.Conv3d_1 = nn.Conv3d(1, 24, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        self.Conv3d_1_bn = nn.BatchNorm3d(24)
        self.Conv3d_2 = nn.Conv3d(25, 25, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        self.Conv3d_2_bn = nn.BatchNorm3d(25)
        self.Conv3d_3 = nn.Conv3d(50, 50, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        self.Conv3d_3_bn = nn.BatchNorm3d(50)
        self.Conv3d_4 = nn.Conv3d(100, 100, (1, 1, 3), (1, 1, 1), (0, 0, 1))
        self.Conv3d_4_bn = nn.BatchNorm3d(100)
        self.Pool = nn.MaxPool3d((1, 1, 204))
        self.Fc = nn.Linear(4900*2, 17)

    def forward(self, input_x):
        input_x = input_x.view(-1, 1, 7, 7, 204)
        # print(input_x.size())

        x_1 = self.Conv3d_1(input_x)
        x_1 = self.Conv3d_1_bn(x_1)
        x_1 = F.relu(x_1)

        x_2 = torch.cat((input_x, x_1), dim=1)
        x_2 = self.Conv3d_2(x_2)
        x_2 = self.Conv3d_2_bn(x_2)
        # print(x_2.size())

        x_3 = torch.cat((input_x, x_1, x_2), dim=1)
        x_3 = self.Conv3d_3(x_3)
        x_3 = self.Conv3d_3_bn(x_3)
        # print(x_3.size())

        x_4 = torch.cat((input_x, x_1, x_2, x_3), dim=1)
        x_4 = self.Conv3d_4(x_4)
        x_4 = self.Conv3d_4_bn(x_4)

        out = torch.cat((input_x, x_1, x_2, x_3, x_4), dim=1)
        out = self.Pool(out)
        # print(out.size())
        out = out.view(-1, 4900*2)
        out = F.log_softmax(self.Fc(out))
        return out


if __name__ == "__main__":
    net = DenseNet()
    x = torch.randn((3, 1, 7, 7, 204))
    x = Variable(x)
    y = net(x)
    print(y.size())
    # net = ResNet(Bottleneck, [3, 4, 6, 3])
    # x = torch.randn((3, 1, 7, 7, 204))
    # x = Variable(x)
    # y = net(x)
    # print(y.size())
