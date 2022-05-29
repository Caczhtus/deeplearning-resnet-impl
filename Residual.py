from common.layers import *
import numpy as np

class Residual:
    def __init__(self, input_channels, output_channels, strides=1, padding=1, kernel_size=3, use_1x1conv=False, weight_init_std="relu"):

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.strides = strides
        self.padding = padding
        self.kernel_size = kernel_size
        self.use_1x1conv = use_1x1conv

        self.conv_size_list = [output_channels, output_channels]  # conv
        self.params = {}  # 参数字典
        self.__init_weight(weight_init_std)

        self.conv1 = Convolution(self.params['W1'], self.params['b1'], strides, padding)
        self.bn1 = BatchNormalization(self.params['gamma1'], self.params['beta1'])
        self.relu1 = Relu()
        self.conv2 = Convolution(self.params['W2'], self.params['b2'], 1, padding)
        self.bn2 = BatchNormalization(self.params['gamma2'], self.params['beta2'])
        self.relu2 = Relu()
        self.conv3 = Convolution(self.params['W3'], self.params['b3'], strides, 0) if self.use_1x1conv else None
        # print(self.params)


    def __init_weight(self, weight_init_std):

        conv_size_list = [self.input_channels] + self.conv_size_list
        for idx in range(1, len(conv_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / conv_size_list[idx - 1])  # 使用ReLU的情况下推荐的初始值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / conv_size_list[idx - 1])  # 使用sigmoid的情况下推荐的初始值
            self.params['W' + str(idx)] = scale * np.random.randn(conv_size_list[idx], conv_size_list[idx-1], self.kernel_size, self.kernel_size)
            self.params['b' + str(idx)] = np.zeros(conv_size_list[idx])
            self.params['gamma' + str(idx)] = np.ones(conv_size_list[idx])  # 按通道广播
            self.params['beta' + str(idx)] = np.zeros(conv_size_list[idx])

        if self.use_1x1conv:
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0)  # 使用ReLU的情况下推荐的初始值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = 1.0  # 使用sigmoid的情况下推荐的初始值
            self.params['W' + str(len(conv_size_list))] = scale * np.random.randn(self.output_channels, self.input_channels, 1, 1)
            self.params['b' + str(len(conv_size_list))] = np.zeros(self.output_channels)

    def forward(self, x):
        identity = x
        out = self.conv1.forward(x)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out)

        if self.conv3 is not None:
            identity = self.conv3.forward(x)

        out += identity
        out = self.relu2.forward(out)

        return out

    def backward(self, dout):
        out = self.relu2.backward(dout)
        #  加法层可以忽略,继续把数据回传给上游
        identity = out
        if self.conv3 is not None:
            identity = self.conv3.backward(out)

        out = self.bn2.backward(out)
        out = self.conv2.backward(out)
        out = self.relu1.backward(out)
        out = self.bn1.backward(out)
        out = self.conv1.backward(out)

        return out + identity