# coding: utf-8
import sys, os

sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from multi_layer_net import MultiLayerNet
from optimizer import *
import matplotlib.pyplot as plt

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

hidden_size_list = [50, 100, 50]
network = MultiLayerNet(input_size=784, hidden_size_list=hidden_size_list, output_size=10)
optimizer = SGD(lr=0.01)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100

train_loss_list = []
train_acc_list = []
test_acc_list = []
epoch_cnt = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 通过误差反向传播法求梯度
    grads = network.gradient(x_batch, t_batch)
    optimizer.update(network.params, grads)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        epoch_cnt.append(i / iter_per_epoch)
        print(
            "epoch" + str(int(i / iter_per_epoch)) + " " + "train accuracy: " + str(
                train_acc) + "," + "train accuracy: " + str(
                test_acc))

# 画图
plt.subplots(1, 1)
plt.plot(epoch_cnt, train_acc_list, label="train accuracy", color="blue")
plt.plot(epoch_cnt, test_acc_list, label="test accuracy", color="red", linestyle="--")
plt.xlabel("epoch")
plt.ylabel("accurcy")
plt.title("accuracy")
plt.legend()
plt.show()
