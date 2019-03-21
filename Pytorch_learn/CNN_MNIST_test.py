import torch.cuda
from torch.autograd import Variable
import torchvision
import torch.utils.data as Data
import torch.nn.functional as Functional
import matplotlib.pyplot as plt
import numpy as np
import CNN_MNIST_network
import scipy.io as sio

BatchSize = 50

test_dataset = torchvision.datasets.MNIST(
    root='data/mnist',
    train=False,
    transform=torchvision.transforms.ToTensor()
)


def print_mid_layer(network, x):
    x = Variable(x)
    x1 = network.conv_1(x)
    # x1 = Functional.relu(network.bn_1(x1))

    plt.figure(dpi=1200)
    n = x1.size()[1]
    sub_x = int(np.sqrt(n))
    sub_y = int(np.ceil(n/sub_x))
    print(sub_x, sub_y)
    for i in range(n):
        picture = x1[0][i].data.numpy()
        plt.subplot(sub_y, sub_y, i + 1)
        plt.imshow(picture)
        plt.title("%d" % i)
    plt.tight_layout()
    plt.savefig("layer-1-conv-out.jpg")
    plt.close()

    x2 = network.conv_2(x1)
    # x2 = Functional.relu(network.bn_2(x2))
    plt.figure(dpi=1200)
    n = x2.size()[1]
    sub_x = int(np.sqrt(n))
    sub_y = int(np.ceil(n/sub_x))
    print(sub_x, sub_y)
    for i in range(n):
        picture = x2[0][i].data.numpy()
        plt.subplot(sub_y, sub_y, i + 1)
        plt.imshow(picture)
        plt.title("%d" % i)
    plt.tight_layout()
    plt.savefig("layer-2-conv-out.jpg")
    plt.close()

    # x3 = network.conv_3(x2)
    # x3 = Functional.relu(network.bn_3(x3))
    # plt.figure(dpi=1200)
    # n = x3.size()[1]
    # sub_x = int(np.sqrt(n))
    # sub_y = int(np.ceil(n/sub_x))
    # print(sub_x, sub_y)
    # for i in range(n):
    #     picture = x3[0][i].data.numpy()
    #     plt.subplot(sub_y, sub_y, i + 1)
    #     plt.imshow(picture)
    #     plt.title("%d" % i)
    # plt.tight_layout()
    # plt.savefig("layer-3-conv-out.jpg")
    # plt.close()

a = [
0, 0, 0, 0, 0,
0.456737, 1.0519, 0.860765, 0.680035, 2.33034,
0, 0, 0, 4.39216, 4.4561,
0, 0, 0.0945578, 5.04304, 0.0604394,
0, 0, 5.7971, 4.67026, 0,

0, 0.243865, 0.130953, 0, 0,
0, 0, 0, 0.00336789, 1.3007,
0, 0, 0.459654, 0.183506, 3.05827,
0, 0, 0, 2.93792, 1.66607,
0, 0, 1.62136, 2.60551, 0,

1.64869, 3.80057, 2.74508, 1.92573, 2.0336,
4.37658, 6.69539, 7.68776, 7.89401, 7.01194,
0.126848, 0.127775, 0.875726, 3.82293, 1.42973,
0.126848, 0.126848, 4.77778, 4.11497, 0,
0.126848, 2.3847, 4.97059, 1.5461, 0.126848,

0.492288, 0.21724, 0, 0, 0,
0.50082, 1.08555, 1.05948, 3.05965, 2.54392,
0, 0, 0.508293, 3.69459, 1.56132,
0, 0, 1.50728, 2.36027, 0,
0, 0.00299244, 2.67632, 0.494184, 0,

1.88584, 0, 0, 0, 0,
2.39362, 1.71836, 1.3274, 3.6028, 0,
0, 0.796372, 1.84716, 4.07406, 0,
0, 0, 2.78393, 2.08718, 0,
0, 2.80574, 3.25806, 0, 0,

1.30598, 1.52287, 0, 0, 0,
0, 4.05244, 4.01862, 4.12208, 3.26193,
0, 0, 0, 0, 0.2665,
0, 0, 0, 0, 0,
0, 0, 0, 0, 0,

0.413501, 0.426844, 0.538497, 0.447958, 0.771674,
0, 0, 0.431228, 0.12749, 5.30524,
0.447958, 0.436199, 0.368969, 0, 5.66522,
0.447958, 0.447958, 0.158151, 4.38798, 5.07952,
0.447958, 0.429074, 0, 4.94744, 1.94902,

0, 0, 0, 0, 0,
4.17303, 6.83536, 7.07215, 5.78796, 1.7429,
0.338657, 1.34062, 1.88481, 0.795907, 1.47954,
0, 0, 0, 0.72894, 0.766185,
0, 0, 0.15361, 0.952541, 0.467687,

2.06507, 3.85193, 3.44813, 3.03016, 2.25423,
0.432389, 3.09576, 3.34753, 3.98551, 3.79584,
0, 0, 0.219671, 1.93559, 2.54705,
0, 0, 1.65263, 2.11375, 2.03252,
0, 1.12055, 2.10709, 2.29867, 0,

0, 0, 0, 0, 0.0740748,
3.35757, 5.23769, 5.39562, 4.93268, 3.66873,
0.295777, 1.04923, 1.39284, 2.90344, 3.97879,
0, 0, 0.0388724, 3.6986, 2.35653,
0, 0, 3.97818, 3.9661, 0.305386,
]

if __name__ == "__main__":
    cnn_network = torch.load('results/version-h1-cnn_network.pkl').cuda()
    cnn_network.eval()

    # have a test
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BatchSize,
        shuffle=True,
        num_workers=2)
    accuracy = 0
    for step, (test_x, test_y) in enumerate(test_loader):
        test_x = Variable(test_x).cuda()
        test_out = cnn_network(test_x)
        prediction_y = torch.max(test_out, 1)[1].cpu().data.squeeze()
        accuracy = accuracy + torch.sum(torch.eq(prediction_y, test_y))

    accuracy = float(accuracy)/test_dataset.test_data.size()[0]
    print('accuracy: %.4f' % accuracy)


    x = test_loader.dataset[0][0]
    # print(x)
    x = x.reshape(1, 1, 28, 28)
    x = Variable(x).cuda()
    # plt.figure()
    # plt.imshow(x)
    # plt.show()
    # print(x)

    x = cnn_network.conv_1(x)
    x = Functional.relu(x)

    x = cnn_network.max_pool_1(x)

    # print(x.size())
    x = cnn_network.conv_2(x)
    x = Functional.relu(x)

    x = cnn_network.max_pool_2(x)


    #
    # a = torch.Tensor(np.array(a)).reshape(x.size())
    # a = Variable(a).cuda()
    #
    # print((x - a).abs().max())


    # print(x.size())
    x = cnn_network.conv_3(x)
    x = Functional.relu(x)
    # print(x.size())
    x = x.view(x.size(0), -1)

    # print(x.size())
    x = cnn_network.fc_2(x)
    x = Functional.relu(x)

    print(x)

    # # print(x[0][8])
    # print(x.size())
    # weight = cnn_network._modules['conv_1']._parameters['weight'].data.cpu().numpy().tolist()
    # file = open('result_conv{}.txt'.format(1), 'w')
    # file.write(str(weight))
    # file.close()
    # weight = cnn_network._modules['conv_2']._parameters['weight'].data.cpu().numpy().tolist()
    # file = open('result_conv{}.txt'.format(2), 'w')
    # file.write(str(weight))
    # file.close()
    # weight = cnn_network._modules['conv_3']._parameters['weight'].data.cpu().numpy().tolist()
    # file = open('result_conv{}.txt'.format(3), 'w')
    # file.write(str(weight))
    # file.close()
    # weight = cnn_network._modules['conv_1']._parameters['bias'].data.cpu().numpy().tolist()
    # file = open('result_conv_b{}.txt'.format(1), 'w')
    # file.write(str(weight))
    # file.close()
    # weight = cnn_network._modules['conv_2']._parameters['bias'].data.cpu().numpy().tolist()
    # file = open('result_conv_b{}.txt'.format(2), 'w')
    # file.write(str(weight))
    # file.close()
    # weight = cnn_network._modules['conv_3']._parameters['bias'].data.cpu().numpy().tolist()
    # file = open('result_conv_b{}.txt'.format(3), 'w')
    # file.write(str(weight))
    # file.close()
    # weight = cnn_network._modules['fc_2']._parameters['weight'].data.cpu().numpy().tolist()
    # file = open('result_fc{}.txt'.format(2), 'w')
    # file.write(str(weight))
    # file.close()
    # weight = cnn_network._modules['fc_2']._parameters['bias'].data.cpu().numpy().tolist()
    # file = open('result_fc_b{}.txt'.format(2), 'w')
    # file.write(str(weight))
    # file.close()


