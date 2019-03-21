import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.utils.data as Data


# G(z)
class generator(nn.Module):
    # initializers
    def __init__(self):
        super(generator, self).__init__()
        self.fc1_1 = nn.Linear(100, 256)
        self.fc1_1_bn = nn.BatchNorm1d(256)
        self.fc1_2 = nn.Linear(10, 256)
        self.fc1_2_bn = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(512, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc3_bn = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 784)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.fc1_1_bn(self.fc1_1(input)))
        y = F.relu(self.fc1_2_bn(self.fc1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.fc2_bn(self.fc2(x)))
        x = F.relu(self.fc3_bn(self.fc3(x)))
        x = F.tanh(self.fc4(x))
        return x


class discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(discriminator, self).__init__()
        self.fc1_1 = nn.Linear(784, 1024)
        self.fc1_2 = nn.Linear(10, 1024)
        self.fc2 = nn.Linear(2048, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.fc1_1(input), 0.2)
        y = F.leaky_relu(self.fc1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
        x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
        x = F.sigmoid(self.fc4(x))
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


temp_z_ = torch.rand(10, 100)
fixed_z_ = temp_z_
fixed_y_ = torch.zeros(10, 1)
for i in range(9):
    fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
    temp = torch.ones(10,1) + i
    fixed_y_ = torch.cat([fixed_y_, temp], 0)


fixed_z_ = Variable(fixed_z_.cuda(), volatile=True)
fixed_y_label_ = torch.zeros(100, 10)
fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
fixed_y_label_ = Variable(fixed_y_label_.cuda(), volatile=True)


# training parameters
batch_size = 128
lr = 0.0002
train_epoch = 50

# data_loader
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])
train_loader = Data.DataLoader(
    datasets.MNIST('data/mnist', train=True, download=True, transform=transform),
    batch_size=batch_size, shuffle=True)

# network
G = generator()
D = discriminator()
G.weight_init(mean=0, std=0.02)
D.weight_init(mean=0, std=0.02)
G.cuda()
D.cuda()

# Binary Cross Entropy loss
BCE_loss = nn.BCELoss()

# Adam optimizer
G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

print('training start!')
for step, (x_, y_) in enumerate(train_loader):
    mini_batch = x_.size()[0]
    print(mini_batch)
    # train discriminator D
    D.zero_grad()

    y_real_ = torch.ones(mini_batch)
    y_fake_ = torch.zeros(mini_batch)
    y_label_ = torch.zeros(mini_batch, 10)
    y_label_.scatter_(1, y_.view(mini_batch, 1), 1)
    x_ = x_.view(-1, 28*28)
    x_, y_label_, y_real_, y_fake_ = Variable(x_.cuda()), Variable(y_label_.cuda()), \
                                     Variable(y_real_.cuda()), Variable(y_fake_.cuda())

    D_result = D(x_, y_label_).squeeze()
    D_real_loss = BCE_loss(D_result, y_real_)

    z_ = torch.rand((mini_batch, 100))
    y_ = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor)
    y_label_ = torch.zeros(mini_batch, 10)
    y_label_.scatter_(1, y_.view(mini_batch, 1), 1)

    z_, y_label_ = Variable(z_.cuda()), Variable(y_label_.cuda())
    G_result = G(z_, y_label_)

    D_result = D(G_result, y_label_).squeeze()
    D_fake_loss = BCE_loss(D_result, y_fake_)
    D_fake_score = D_result.data.mean()

    D_train_loss = D_real_loss + D_fake_loss

    D_train_loss.backward()
    D_optimizer.step()

    # train generator G
    G.zero_grad()

    z_ = torch.rand((mini_batch, 100))
    y_ = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor)
    y_label_ = torch.zeros(mini_batch, 10)
    y_label_.scatter_(1, y_.view(mini_batch, 1), 1)

    z_, y_label_ = Variable(z_.cuda()), Variable(y_label_.cuda())

    G_result = G(z_, y_label_)
    D_result = D(G_result, y_label_).squeeze()
    G_train_loss = BCE_loss(D_result, y_real_)
    G_train_loss.backward()
    G_optimizer.step()
    if step % 5 == 0:
        print('step is %d hehe' % step)
