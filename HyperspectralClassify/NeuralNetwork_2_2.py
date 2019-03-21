import torch
import torch.nn as nn
import torch.nn.functional as torch_f
import data_processor


dp = data_processor.DataProcessor()
GeneratorNoiseDim = 256
GeneratorOutputSize = dp.DataSize   # the size of true data
GeneratorInputLayerOutputSize = 256
GeneratorHiddenLayerSize = 256
LabelSize = dp.LabelSize
GeneratorCombineLayerSize = 256


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
        # self.hidden_layer = nn.Linear(GeneratorCombineLayerSize, GeneratorHiddenLayerSize)
        # self.hidden_layer_norm = nn.BatchNorm1d(GeneratorHiddenLayerSize)
        self.output_layer = nn.Linear(GeneratorCombineLayerSize, GeneratorOutputSize)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, noise_input, label):
        x = torch_f.relu(self.noise_input_layer_norm(self.noise_input_layer(noise_input)))
        y = torch_f.relu(self.label_input_layer(label))
        x = torch.cat([x, y], 1)
        x = torch_f.relu(self.combine_layer_norm(self.combine_layer(x)))
        # x = torch_f.relu(self.hidden_layer_norm(self.hidden_layer(x)))
        x = self.output_layer(x)
        x = torch_f.sigmoid(x)
        return x


DiscriminatorInputSize = GeneratorOutputSize
DiscriminatorInputLayerOutputSize = 256
DiscriminatorCombineLayerSize = 256
DiscriminatorHiddenLayerSize = 256


class Discriminator(nn.Module):
    # initializer
    def __init__(self):
        super(Discriminator, self).__init__()
        self.data_input_layer = nn.Linear(GeneratorOutputSize, DiscriminatorInputLayerOutputSize)
        self.label_input_layer = nn.Linear(LabelSize, DiscriminatorInputLayerOutputSize)
        self.combine_layer = nn.Linear(DiscriminatorInputLayerOutputSize*2, DiscriminatorCombineLayerSize)
        self.combine_layer_norm = nn.BatchNorm1d(DiscriminatorCombineLayerSize)
        self.output_layer = nn.Linear(DiscriminatorCombineLayerSize, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, data_input, label):
        x = torch_f.relu(self.data_input_layer(data_input))
        y = torch_f.relu(self.label_input_layer(label))
        x = torch.cat([x, y], 1)
        x = torch_f.relu(self.combine_layer(x))
        x = torch_f.sigmoid(self.output_layer(x))
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

