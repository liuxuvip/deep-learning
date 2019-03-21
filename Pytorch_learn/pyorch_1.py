import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as nF
import torch.utils.data as Data


# Linear regression
# fake data set
Num_data = 1000
X = torch.linspace(-1, 1, Num_data)
X = torch.unsqueeze(X, dim=1)
Parameters = {'a': 2, 'b': -1, 'c': 0}
Y = Parameters['a']*X*X + Parameters['b']*X + Parameters['c'] + 0.1*torch.randn(X.size())

# load X, Y into data set
DataSet = Data.TensorDataset(data_tensor=X, target_tensor=Y)
X = Variable(X)
Y = Variable(Y)

# load data
# torch.manual_seed(1)    # reproducible
data_loder = Data.DataLoader(dataset=DataSet, batch_size=100, shuffle=True, num_workers=2)


# new a neural network
class Net(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.output = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.hidden(x)
        x = nF.relu(x)  # put x into active function
        x = self.output(x)
        return x


net1 = Net(n_input=1, n_hidden=50, n_output=1)
print(net1)

# optimizer = torch.optim.SGD(net1.parameters(), lr=0.3)
loss_func = torch.nn.MSELoss()
optimizer= torch.optim.SGD(net1.parameters(), lr=0.01)

# begin training
plt.figure(1, [16, 9])
plt.ion()

LossRecord = []
# begin training
for training_times in range(100):
    for step, (batch_x, batch_y) in enumerate(data_loder):
        # print('training', training_times, 'step', step)
        # put the tensors into 'variable'
        batch_x = Variable(batch_x)
        batch_y = Variable(batch_y)
        out = net1(batch_x)
        loss = loss_func(out, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 2 == 1:
            LossRecord.append(loss.data[0])

    plt.cla()
    out = net1(X)
    plt.scatter(X.data.numpy(), Y.data.numpy(), c='blue', linewidths=1.5)
    plt.plot(X.data.numpy(), out.data.numpy(), 'b', lw=2.5)
    loss = loss_func(out, Y)
    plt.text(0.8, -0.8, 'loss = %.5f' % loss.data[0], fontdict={'size': 20, 'color': 'black'})
    plt.pause(0.01)


plt.ioff()


plt.figure(2)
Num_data = 100
test_data = torch.linspace(-1, 1, Num_data)
test_data = torch.unsqueeze(test_data, dim=1)
test_data = Variable(test_data)
out = net1(test_data)
plt.plot(test_data.data.numpy(), out.data.numpy(), 'b', lw=2.5)

# torch.save(net1.state_dict(), 'net1.pkl')
#
#
# net3 = Net(n_input=1, n_hidden=80, n_output=1)
# net3.load_state_dict(torch.load('net1.pkl'))
# plt.figure(2)
# y_new = net3(x)
# plt.scatter(x.data.numpy(), y.data.numpy(), c='blue', linewidths=1.5)
# plt.plot(x.data.numpy(), y_new.data.numpy(), 'b', lw=2.5)
# plt.show()
plt.figure(3, [16, 9])
L = [i for i in range(len(LossRecord))]
plt.plot(L, LossRecord)
plt.show()
