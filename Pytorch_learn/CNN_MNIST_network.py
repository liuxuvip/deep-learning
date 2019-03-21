import torch.nn as nn
import torch.nn.functional as Functional


class CNN_MNIST_1(nn.Module):
    def __init__(self, size):
        super(CNN_MNIST_1, self).__init__()
        self.conv_1 = nn.Conv2d(
            in_channels=size["conv_1"].in_channels,
            out_channels=size["conv_1"].out_channels,
            kernel_size=size["conv_1"].kernel_size,
            stride=size["conv_1"].stride,
            padding=size["conv_1"].padding
        )
        self.conv_2 = nn.Conv2d(
            in_channels=size["conv_2"].in_channels,
            out_channels=size["conv_2"].out_channels,
            kernel_size=size["conv_2"].kernel_size,
            stride=size["conv_2"].stride,
            padding=size["conv_2"].padding
        )
        self.conv_3 = nn.Conv2d(
            in_channels=size["conv_3"].in_channels,
            out_channels=size["conv_3"].out_channels,
            kernel_size=size["conv_3"].kernel_size,
            stride=size["conv_3"].stride,
            padding=size["conv_3"].padding
        )
        # self.conv_4 = nn.Conv2d(
        #     in_channels=size["conv_4"].in_channels,
        #     out_channels=size["conv_4"].out_channels,
        #     kernel_size=size["conv_4"].kernel_size,
        #     stride=size["conv_4"].stride,
        #     padding=size["conv_4"].padding
        # )
        # self.conv_5 = nn.Conv2d(
        #     in_channels=size["conv_5"].in_channels,
        #     out_channels=size["conv_5"].out_channels,
        #     kernel_size=size["conv_5"].kernel_size,
        #     stride=size["conv_5"].stride,
        #     padding=size["conv_5"].padding
        # )

        # self.bn_1 = nn.BatchNorm2d(num_features=size["conv_1"].out_channels)
        # self.bn_2 = nn.BatchNorm2d(num_features=size["conv_2"].out_channels)
        # self.bn_3 = nn.BatchNorm2d(num_features=size["conv_3"].out_channels)
        # self.bn_4 = nn.BatchNorm2d(num_features=size["conv_4"].out_channels)
        # self.bn_5 = nn.BatchNorm2d(num_features=size["conv_5"].out_channels)

        # self.drop_out_1 = nn.Dropout2d()
        # self.drop_out_2 = nn.Dropout2d()
        # self.drop_out_3 = nn.Dropout2d()
        # self.drop_out_4 = nn.Dropout2d()
        # self.drop_out_5 = nn.Dropout2d()

        self.max_pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.fc_1 = nn.Linear(
        #     in_features=size["fc_1"].in_features,
        #     out_features=size["fc_1"].out_features
        # )
        self.fc_2 = nn.Linear(
            in_features=size["fc_2"].in_features,
            out_features=size["fc_2"].out_features
        )

    def forward(self, x):

        # print("0", x.size())
        x = self.conv_1(x)
        # x = Functional.relu(self.bn_1(x))
        # x = Functional.relu(self.drop_out_1(x))
        # x = Functional.leaky_relu(self.drop_out_1(x))
        x = Functional.relu(x)
        x = self.max_pool_1(x)
        # x = Functional.relu(self.bn_1(x))
        # x = self.drop_out_1(x)
        # print("1", x.size())

        x = self.conv_2(x)
        # x = Functional.relu(self.bn_2(x))
        # x = Functional.relu(self.drop_out_2(x))
        # x = Functional.leaky_relu(self.drop_out_2(x))
        x = Functional.relu(x)
        x = self.max_pool_2(x)
        # x = Functional.relu(self.bn_2(x))
        # x = self.drop_out_2(x)
        # print("2", x.size())

        x = self.conv_3(x)
        # x = Functional.relu(self.bn_3(x))
        # x = Functional.relu(self.drop_out_3(x))
        # x = Functional.leaky_relu(self.drop_out_3(x))
        x = Functional.relu(x)
        # x = self.max_pool_2(x)
        # x = Functional.relu(self.bn_3(x))
        # x = self.drop_out_3(x)
        # print("3", x.size())

        # x = self.conv_4(x)
        # x = Functional.relu(self.drop_out_4(x))
        # x = Functional.relu(self.bn_4(x))

        # x = self.conv_5(x)
        # x = Functional.relu(self.drop_out_5(x))
        # x = Functional.relu(self.bn_5(x))
        #
        x = x.view(x.size(0), -1)
        # print("4", x.size())

        # x = self.fc_1(x)
        # x = Functional.relu(x)
        # print("5", x.size())

        x = self.fc_2(x)
        # print("6", x.size())
        return x


class ConvSize:
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding


class FcSize:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features