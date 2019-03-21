import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    for i in range(1, 7):
        loss_his = []
        accuracy_his = []
        with open('00{}.txt'.format(i)) as f:
            for step, line in enumerate(f):
                if line[0] != '-':
                    line = line.split('| ')
                    # print(line, step)
                    if step % 4 == 1:
                        loss = float(line[3].split(' ')[1])
                        loss_his.append(loss)
                    elif step % 4 == 3:
                        accuracy = float(line[4].split(' ')[1].split('%')[0])
                        accuracy_his.append(accuracy)

        loss_his = np.array(loss_his)
        accuracy_his = np.array(accuracy_his)
        print(i, "accuracy max: ", accuracy_his.max(), "loss min: ", loss_his.min())
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(accuracy_his)
        plt.title("accuracy record")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.subplot(2, 1, 2)
        plt.plot(loss_his)
        plt.title("loss record")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.tight_layout()
        plt.savefig("00{}.tif".format(i))


