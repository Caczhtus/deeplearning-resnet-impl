from matplotlib import pyplot as plt

from Residual import Residual
from ResNet18 import ResNet18
from common.layers import *
from common.util import smooth_curve
from mnist import load_mnist
import numpy as np
from common.trainer import Trainer

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=False)

x_train = x_train[:16]
t_train = t_train[:16]
x_test = x_test[:4]
t_test = t_test[:4]

print(x_train.shape)

# x_train = x_train.reshape(2, 1, -1)
# print(x_train.shape)

# w = np.random.randn(784, 1000)
# print(w.shape)

# s = np.dot(x_train, w)
# print(s.shape)

# c = Residual(input_channels=x_train.shape[1], output_channels=64, strides=2, use_1x1conv=True)
# d = c.forward(x_train)
# print(d.shape)
# e = c.backward(d)


# np.random.seed(10)
# a = np.random.randint(1, 10, [5, 3])
# print(a.size)
# print(a.shape)
# b = np.max(a, axis=1)  # 找一个每行最大的
# print(b)

# c = AvgPooling(3, 3, 1)
# x_train = x_train[:, :, 16:20, 16:20]
# print(x_train.shape)
# x = c.forward(x_train)
# y = c.backward(x)
# print(x_train[0], end='\n\n')
# print(x, end='\n\n')
# print(y, end='\n\n')

# d = Affine(w, np.random.randn(1000))
# e = d.forward(x_train)
# print(e.shape)
# e = d.backward(e)
# print(e.shape)
# f = AvgPooling(28, 28)
# g = f.forward(x_train)
# print(g.shape)

# h = ResNet18(input_channels=x_train.shape[1], num_class=10)
# i = h.forward(x_train)
# print(i.shape)
# j = h.backward(i)
# print(j.shape)
# print(t_train.shape)
# print(t_train)
# k = h.last_layer.forward(i, t_train)
# print(np.argmax(i, axis=1))
# print(k)
# l = h.loss(x_train, t_train)
# print(l)
# m = h.accuracy(x_train, t_train)
# print(m)
# n = h.gradient(x_train, t_train)
# print(len(n))

net = ResNet18(input_channels=x_train.shape[1], num_class=10, weight_init_std='he')
trainer = Trainer(net, x_train, t_train, x_test, t_test, epochs=1000, verbose=True, mini_batch_size=1)
trainer.train()
# 绘制图形==========
plt.plot(np.arange(len(trainer.train_loss_list)), smooth_curve(trainer.train_loss_list), markevery=100, label='x')
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0.8, 2.5)
plt.legend()
plt.show()

