import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as nF
import time

WithGpu = True

# fake data
P1 = (0.1, 0.2)
P2 = (0.4, 0.7)
P3 = (0.8, 0.5)

Num_data = 250

P1_s = torch.ones([Num_data, 2])
P1_s[:, 0] *= P1[0]
P1_s[:, 1] *= P1[1]

P2_s = torch.ones([Num_data, 2])
P2_s[:, 0] *= P2[0]
P2_s[:, 1] *= P2[1]

P3_s = torch.ones([Num_data, 2])
P3_s[:, 0] *= P3[0]
P3_s[:, 1] *= P3[1]

x_1 = torch.normal(P1_s, 0.1)
x_2 = torch.normal(P2_s, 0.09)
x_3 = torch.normal(P3_s, 0.09)

x = torch.cat((x_1, x_2, x_3)).type(torch.FloatTensor)
if WithGpu:
    x = Variable(x).cuda()
else:
    x = Variable(x)
y = torch.squeeze(torch.zeros([3*Num_data]))
y[: Num_data] = 0
y[Num_data:2*Num_data] = 1
y[2*Num_data: 3*Num_data] = 2

net2 = torch.nn.Sequential(
    torch.nn.Linear(2, 30),
    torch.nn.ReLU(),
    torch.nn.Linear(30, 3),
)

if WithGpu:
    y = Variable(y.type(torch.LongTensor)).cuda()
    loss_func = torch.nn.CrossEntropyLoss().cuda()
    net2.cuda()
else:
    y = Variable(y.type(torch.LongTensor))
    loss_func = torch.nn.CrossEntropyLoss()


optimizer = torch.optim.SGD(net2.parameters(), lr=0.1)

last_loss = loss_func(net2(x), y)
optimizer.zero_grad()
last_loss.backward()
optimizer.step()

plt.figure(1, [16, 9])
plt.ion()
plt.show()

dict = {0: 'red', 1: 'black', 2: 'yellow'}

start = time.clock()

Max_iteration_general = 2000
Error_loss = 5e-5
for i in range(Max_iteration_general):
    out = net2(x)
    loss = loss_func(out, y)
    if abs(loss.cpu().data[0] - last_loss.cpu().data[0]) < Error_loss:
        end = time.clock()
        plt.title('Calculation converge\ntotal running time %.1fs' % (end - start), size=30, color='green')
        print(last_loss, loss)
        break
    else:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_loss = loss

        if i % 10 == 0:
            prediction = torch.max(nF.softmax(out), 1)[1]
            pred_y = prediction.data.cpu().numpy().squeeze()
            target_y = y.data.cpu().numpy()
            accuracy = sum(pred_y == target_y) / 3.0 / Num_data
            plt.cla()
            plt.scatter(x.data.cpu().numpy()[:, 0], x.data.cpu().numpy()[:, 1],
                        c=[dict[w] for w in pred_y], lw=2)
            # plt.text(0, 0, 'Calculating loss is %.4f, accuracy is %.4f' % (loss.cpu().data[0], accuracy),
            #          fontdict={'size': 25, 'color': 'red'})
            plt.xlabel('Calculating...   loss is %.4f, accuracy is %.4f' % (loss.cpu().data[0], accuracy),
                       size=25, color='red')
            plt.pause(0.01)

plt.ioff()
plt.show()
