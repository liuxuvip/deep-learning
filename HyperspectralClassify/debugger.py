from torch.autograd import Variable
import matplotlib.pyplot as plt
from NeuralNetwork_2_2 import *
import numpy as np
import GlobalDefine


train_set = np.load("train_set.npy")
train_set_label = np.load("train_set_label.npy")

G = Generator()
G.load_state_dict(torch.load('G_state' + GlobalDefine.run_version + '.pkl'))


num_data = dp.TrainNumPerClass * dp.LabelSize


def check_result():
    noise = torch.rand((num_data, GeneratorNoiseDim))

    y_ = torch.zeros((num_data, 1)).type(torch.LongTensor)
    start = 0
    for i in range(dp.LabelSize):
        y_[start:(start + dp.TrainNumPerClass)] = i
        start += dp.TrainNumPerClass

    y_label = torch.zeros(num_data, LabelSize)
    y_label.scatter_(1, y_.view(num_data, 1), 1)
    noise, y_label = Variable(noise), Variable(y_label)

    fake_data = G(noise, y_label)

    fake_data = fake_data.data.numpy()
    fake_label = y_.numpy().reshape(num_data)

    link = np.array(range(0, num_data, 2))

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.title('real data')

    plt.imshow(train_set[link] * 1000, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.imshow(fake_data[link] * 1000, cmap='gray')
    plt.title('fake data')

    for i in range(2, 15):
        plt.figure(i)
        a = np.random.randint(low=0, high=num_data)
        x_axis = np.array(range(dp.DataSize))
        plt.title('label is %d black is real data, red is fake data' % (train_set_label[a]))
        plt.plot(x_axis, train_set[a], 'black', x_axis, fake_data[a], 'red')
        plt.axis([0, dp.DataSize, 0, 1])
        plt.savefig('label is %d' % train_set_label[a])


        # plt.show()






LabelSize = 3
TrainNumPerClass = 2
DataSize = 3


def train_test_split(data, label):
    import random
    # classify data
    data_classified = list(range(LabelSize))
    for i in range(len(data_classified)):
        data_classified[i] = []
    # rearrange data by its label
    for i in range(len(data)):
        data_classified[label[i] - 1].append(data[i])

    # from each class, extract 200 random samples
    data_extracted = list(range(LabelSize))  # extracted data
    for i in range(len(data_classified)):
        class_i_len = len(data_classified[i])
        data_classified[i] = np.array(data_classified[i]).astype(np.float32)
        extract_nums = list(range(class_i_len))
        extract_subs = random.sample(extract_nums, TrainNumPerClass)
        data_extracted[i] = data_classified[i][extract_subs, :]
        data_classified[i][extract_subs] = np.inf
        data_classified[i] = data_classified[i][data_classified[i][:, 0] != np.inf]
    # flatten the list to array
    train_set = np.zeros((LabelSize*TrainNumPerClass, DataSize))
    train_label = np.zeros(LabelSize*TrainNumPerClass)
    for i in range(LabelSize):
        train_set[i * TrainNumPerClass:(i*TrainNumPerClass + TrainNumPerClass)] = data_extracted[i]
        train_label[i * TrainNumPerClass:(i*TrainNumPerClass + TrainNumPerClass)] = i + 1

    length = 0
    for i in range(len(data_classified)):
        length += len(data_classified[i])
    test_label = np.zeros(length)
    test_set = []
    start = 0
    for i in range(len(data_classified)):
        for j in range(len(data_classified[i])):
            test_set.append(data_classified[i][j])
        test_label[start:(start + len(data_classified[i]))] = i + 1
        start += len(data_classified[i])

    test_set = np.array(test_set)
    return train_set, train_label, test_set, test_label

def check_data_generater_correct():
    # remove all the zeros
    data_set = np.random.rand(15, DataSize)
    data_label = np.random.randint(1,  LabelSize + 1, (15, 1))
    data_label = data_label.reshape(-1)
    print(
        'data set', data_set,
        'label set', data_label
    )
    data_set = data_set[data_label != 0]
    data_label = data_label[data_label != 0]
    train_set, train_label, test_set, test_label = train_test_split(data_set, data_label)
    print(
        'one', train_set,
        'two', train_label,
        'three', test_set,
        'four', test_label)

# for i in range(1, dp.LabelSize):
#     plt.figure(i)
#     plt.plot(train_set[i*dp.TrainNumPerClass])
#     plt.title('label is %d' % i)
#     plt.show()

if __name__ == '__main__':
    check_result()
