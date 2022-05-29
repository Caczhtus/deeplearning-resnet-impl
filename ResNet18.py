from Residual import Residual
from common.layers import *


class ResNet18:
    def __init__(self, input_channels, num_class, weight_init_std='relu'):

        self.input_channels = input_channels
        self.num_class = num_class

        self.params = {}  # 参数字典
        self.__init_weight(weight_init_std)

        self.linear = Linear(self.params['W0'], self.params['b0'])  # 先做一层线性变换，使得符合resnet18的图像尺寸要求  28x28 --> 224x224
        self.conv1 = Convolution(self.params['W1'], self.params['b1'], 2, 3)  # 7x7  s=2  pad=3  out=64  scale=14x14
        self.bn1 = BatchNormalization(self.params['gamma1'], self.params['beta1'])
        self.relu1 = Relu()
        self.max_pool = MaxPooling(3, 3, 2, 1)  # scale=7x7
        self.layer11 = Residual(64, 64)
        self.layer12 = Residual(64, 64)
        self.layer21 = Residual(64, 128, 2, use_1x1conv=True)
        self.layer22 = Residual(128, 128)
        self.layer31 = Residual(128, 256, 2, use_1x1conv=True)
        self.layer32 = Residual(256, 256)
        self.layer41 = Residual(256, 512, 2, use_1x1conv=True)
        self.layer42 = Residual(512, 512)
        self.avg_pool = AvgPooling(7, 7)  # 利用平均池化层模拟自适应平均池化层
        self.fc = Affine(self.params['W2'], self.params['b2'])  # 全连接
        self.last_layer = SoftmaxWithLoss()


    def __init_weight(self, weight_init_std):

        scale0 = np.sqrt(2.0 / (28.0*28.0)) if str(weight_init_std).lower() in ('relu', 'he') else np.sqrt(1.0 / 28.0*28.0)
        scale1 = np.sqrt(2.0 / self.input_channels) if str(weight_init_std).lower() in ('relu', 'he') else np.sqrt(1.0 / self.input_channels)
        scale2 = np.sqrt(2.0 / 512.0) if str(weight_init_std).lower() in ('relu', 'he') else np.sqrt(1.0 / 512.0)

        self.params['W0'] = scale0 * np.random.randn(28*28, 224*224)
        self.params['b0'] = np.zeros(224*224)
        self.params['W1'] = scale1 * np.random.randn(64, self.input_channels, 7, 7)
        self.params['b1'] = np.zeros(64)
        self.params['W2'] = scale2 * np.random.randn(512, self.num_class)
        self.params['b2'] = np.zeros(self.num_class)
        self.params['gamma1'] = np.ones(64)
        self.params['beta1'] = np.zeros(64)

    def forward(self, x):
        out = self.linear.forward(x)
        out = out.reshape(out.shape[0], out.shape[1], 224, 224)
        out = self.conv1.forward(out)
        out = self.bn1.forward(out)
        out = self.relu1.forward(out)
        out = self.max_pool.forward(out)
        out = self.layer11.forward(out)
        out = self.layer12.forward(out)
        out = self.layer21.forward(out)
        out = self.layer22.forward(out)
        out = self.layer31.forward(out)
        out = self.layer32.forward(out)
        out = self.layer41.forward(out)
        out = self.layer42.forward(out)
        out = self.avg_pool.forward(out)
        out = self.fc.forward(out)
        return out

    def backward(self, x):
        out = self.fc.backward(x)
        out = self.avg_pool.backward(out)
        out = self.layer42.backward(out)
        out = self.layer41.backward(out)
        out = self.layer32.backward(out)
        out = self.layer31.backward(out)
        out = self.layer22.backward(out)
        out = self.layer21.backward(out)
        out = self.layer12.backward(out)
        out = self.layer11.backward(out)
        out = self.max_pool.backward(out)
        out = self.relu1.backward(out)
        out = self.bn1.backward(out)
        out = self.conv1.backward(out)
        out = out.reshape(out.shape[0], out.shape[1], -1)
        out = self.linear.backward(out)
        return out

    def loss(self, x, t):
        y = self.forward(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, X, T):
        Y = self.forward(X)
        Y = np.argmax(Y, axis=1)
        if T.ndim != 1:
            T = np.argmax(T, axis=1)
        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t)
        out = 1  # 对自身求导为1
        out = self.last_layer.backward(out)
        out = self.backward(out)

        grads = {}
        grads['linear_w'] = self.linear.dW
        grads['linear_b'] = self.linear.db
        grads['conv1_w'] = self.conv1.dW
        grads['conv1_b'] = self.conv1.db
        grads['bn1_gamma'] = self.bn1.dgamma
        grads['bn1_beta'] = self.bn1.dbeta

        grads['layer11_W1'] = self.layer11.conv1.dW
        grads['layer11_b1'] = self.layer11.conv1.db
        grads['layer11_gamma1'] = self.layer11.bn1.dgamma
        grads['layer11_beta1'] = self.layer11.bn1.dbeta
        grads['layer11_W2'] = self.layer11.conv2.dW
        grads['layer11_b2'] = self.layer11.conv2.db
        grads['layer11_gamma2'] = self.layer11.bn2.dgamma
        grads['layer11_beta2'] = self.layer11.bn2.dbeta
        if self.layer11.use_1x1conv:
            grads['layer11_W3'] = self.layer11.conv3.dW
            grads['layer11_b3'] = self.layer11.conv3.db

        grads['layer12_W1'] = self.layer12.conv1.dW
        grads['layer12_b1'] = self.layer12.conv1.db
        grads['layer12_gamma1'] = self.layer12.bn1.dgamma
        grads['layer12_beta1'] = self.layer12.bn1.dbeta
        grads['layer12_W2'] = self.layer12.conv2.dW
        grads['layer12_b2'] = self.layer12.conv2.db
        grads['layer12_gamma2'] = self.layer12.bn2.dgamma
        grads['layer12_beta2'] = self.layer12.bn2.dbeta
        if self.layer12.use_1x1conv:
            grads['layer12_W3'] = self.layer12.conv3.dW
            grads['layer12_b3'] = self.layer12.conv3.db

        grads['layer21_W1'] = self.layer21.conv1.dW
        grads['layer21_b1'] = self.layer21.conv1.db
        grads['layer21_gamma1'] = self.layer21.bn1.dgamma
        grads['layer21_beta1'] = self.layer21.bn1.dbeta
        grads['layer21_W2'] = self.layer21.conv2.dW
        grads['layer21_b2'] = self.layer21.conv2.db
        grads['layer21_gamma2'] = self.layer21.bn2.dgamma
        grads['layer21_beta2'] = self.layer21.bn2.dbeta
        if self.layer21.use_1x1conv:
            grads['layer21_W3'] = self.layer21.conv3.dW
            grads['layer21_b3'] = self.layer21.conv3.db

        grads['layer22_W1'] = self.layer22.conv1.dW
        grads['layer22_b1'] = self.layer22.conv1.db
        grads['layer22_gamma1'] = self.layer22.bn1.dgamma
        grads['layer22_beta1'] = self.layer22.bn1.dbeta
        grads['layer22_W2'] = self.layer22.conv2.dW
        grads['layer22_b2'] = self.layer22.conv2.db
        grads['layer22_gamma2'] = self.layer22.bn2.dgamma
        grads['layer22_beta2'] = self.layer22.bn2.dbeta
        if self.layer22.use_1x1conv:
            grads['layer22_W3'] = self.layer22.conv3.dW
            grads['layer22_b3'] = self.layer22.conv3.db

        grads['layer31_W1'] = self.layer31.conv1.dW
        grads['layer31_b1'] = self.layer31.conv1.db
        grads['layer31_gamma1'] = self.layer31.bn1.dgamma
        grads['layer31_beta1'] = self.layer31.bn1.dbeta
        grads['layer31_W2'] = self.layer31.conv2.dW
        grads['layer31_b2'] = self.layer31.conv2.db
        grads['layer31_gamma2'] = self.layer31.bn2.dgamma
        grads['layer31_beta2'] = self.layer31.bn2.dbeta
        if self.layer31.use_1x1conv:
            grads['layer31_W3'] = self.layer31.conv3.dW
            grads['layer31_b3'] = self.layer31.conv3.db

        grads['layer32_W1'] = self.layer32.conv1.dW
        grads['layer32_b1'] = self.layer32.conv1.db
        grads['layer32_gamma1'] = self.layer32.bn1.dgamma
        grads['layer32_beta1'] = self.layer32.bn1.dbeta
        grads['layer32_W2'] = self.layer32.conv2.dW
        grads['layer32_b2'] = self.layer32.conv2.db
        grads['layer32_gamma2'] = self.layer32.bn2.dgamma
        grads['layer32_beta2'] = self.layer32.bn2.dbeta
        if self.layer32.use_1x1conv:
            grads['layer32_W3'] = self.layer32.conv3.dW
            grads['layer32_b3'] = self.layer32.conv3.db

        grads['layer41_W1'] = self.layer41.conv1.dW
        grads['layer41_b1'] = self.layer41.conv1.db
        grads['layer41_gamma1'] = self.layer41.bn1.dgamma
        grads['layer41_beta1'] = self.layer41.bn1.dbeta
        grads['layer41_W2'] = self.layer41.conv2.dW
        grads['layer41_b2'] = self.layer41.conv2.db
        grads['layer41_gamma2'] = self.layer41.bn2.dgamma
        grads['layer41_beta2'] = self.layer41.bn2.dbeta
        if self.layer41.use_1x1conv:
            grads['layer41_W3'] = self.layer41.conv3.dW
            grads['layer41_b3'] = self.layer41.conv3.db

        grads['layer42_W1'] = self.layer42.conv1.dW
        grads['layer42_b1'] = self.layer42.conv1.db
        grads['layer42_gamma1'] = self.layer42.bn1.dgamma
        grads['layer42_beta1'] = self.layer42.bn1.dbeta
        grads['layer42_W2'] = self.layer42.conv2.dW
        grads['layer42_b2'] = self.layer42.conv2.db
        grads['layer42_gamma2'] = self.layer42.bn2.dgamma
        grads['layer42_beta2'] = self.layer42.bn2.dbeta
        if self.layer42.use_1x1conv:
            grads['layer42_W3'] = self.layer42.conv3.dW
            grads['layer42_b3'] = self.layer42.conv3.db

        grads['affine_w'] = self.fc.dW
        grads['affine_b'] = self.fc.db

        return grads

    def gg(self):
        g = {}
        g['linear_w'] = self.linear.W
        g['linear_b'] = self.linear.b
        g['conv1_w'] = self.conv1.W
        g['conv1_b'] = self.conv1.b
        g['bn1_gamma'] = self.bn1.gamma
        g['bn1_beta'] = self.bn1.beta

        g['layer11_W1'] = self.layer11.conv1.W
        g['layer11_b1'] = self.layer11.conv1.b
        g['layer11_gamma1'] = self.layer11.bn1.gamma
        g['layer11_beta1'] = self.layer11.bn1.beta
        g['layer11_W2'] = self.layer11.conv2.W
        g['layer11_b2'] = self.layer11.conv2.b
        g['layer11_gamma2'] = self.layer11.bn2.gamma
        g['layer11_beta2'] = self.layer11.bn2.beta
        if self.layer11.use_1x1conv:
            g['layer11_W3'] = self.layer11.conv3.W
            g['layer11_b3'] = self.layer11.conv3.b

        g['layer12_W1'] = self.layer12.conv1.W
        g['layer12_b1'] = self.layer12.conv1.b
        g['layer12_gamma1'] = self.layer12.bn1.gamma
        g['layer12_beta1'] = self.layer12.bn1.beta
        g['layer12_W2'] = self.layer12.conv2.W
        g['layer12_b2'] = self.layer12.conv2.b
        g['layer12_gamma2'] = self.layer12.bn2.gamma
        g['layer12_beta2'] = self.layer12.bn2.beta
        if self.layer12.use_1x1conv:
            g['layer12_W3'] = self.layer12.conv3.W
            g['layer12_b3'] = self.layer12.conv3.b

        g['layer21_W1'] = self.layer21.conv1.W
        g['layer21_b1'] = self.layer21.conv1.b
        g['layer21_gamma1'] = self.layer21.bn1.gamma
        g['layer21_beta1'] = self.layer21.bn1.beta
        g['layer21_W2'] = self.layer21.conv2.W
        g['layer21_b2'] = self.layer21.conv2.b
        g['layer21_gamma2'] = self.layer21.bn2.gamma
        g['layer21_beta2'] = self.layer21.bn2.beta
        if self.layer21.use_1x1conv:
            g['layer21_W3'] = self.layer21.conv3.W
            g['layer21_b3'] = self.layer21.conv3.b

        g['layer22_W1'] = self.layer22.conv1.W
        g['layer22_b1'] = self.layer22.conv1.b
        g['layer22_gamma1'] = self.layer22.bn1.gamma
        g['layer22_beta1'] = self.layer22.bn1.beta
        g['layer22_W2'] = self.layer22.conv2.W
        g['layer22_b2'] = self.layer22.conv2.b
        g['layer22_gamma2'] = self.layer22.bn2.gamma
        g['layer22_beta2'] = self.layer22.bn2.beta
        if self.layer22.use_1x1conv:
            g['layer22_W3'] = self.layer22.conv3.W
            g['layer22_b3'] = self.layer22.conv3.b

        g['layer31_W1'] = self.layer31.conv1.W
        g['layer31_b1'] = self.layer31.conv1.b
        g['layer31_gamma1'] = self.layer31.bn1.gamma
        g['layer31_beta1'] = self.layer31.bn1.beta
        g['layer31_W2'] = self.layer31.conv2.W
        g['layer31_b2'] = self.layer31.conv2.b
        g['layer31_gamma2'] = self.layer31.bn2.gamma
        g['layer31_beta2'] = self.layer31.bn2.beta
        if self.layer31.use_1x1conv:
            g['layer31_W3'] = self.layer31.conv3.W
            g['layer31_b3'] = self.layer31.conv3.b

        g['layer32_W1'] = self.layer32.conv1.W
        g['layer32_b1'] = self.layer32.conv1.b
        g['layer32_gamma1'] = self.layer32.bn1.gamma
        g['layer32_beta1'] = self.layer32.bn1.beta
        g['layer32_W2'] = self.layer32.conv2.W
        g['layer32_b2'] = self.layer32.conv2.b
        g['layer32_gamma2'] = self.layer32.bn2.gamma
        g['layer32_beta2'] = self.layer32.bn2.beta
        if self.layer32.use_1x1conv:
            g['layer32_W3'] = self.layer32.conv3.W
            g['layer32_b3'] = self.layer32.conv3.b

        g['layer41_W1'] = self.layer41.conv1.W
        g['layer41_b1'] = self.layer41.conv1.b
        g['layer41_gamma1'] = self.layer41.bn1.gamma
        g['layer41_beta1'] = self.layer41.bn1.beta
        g['layer41_W2'] = self.layer41.conv2.W
        g['layer41_b2'] = self.layer41.conv2.b
        g['layer41_gamma2'] = self.layer41.bn2.gamma
        g['layer41_beta2'] = self.layer41.bn2.beta
        if self.layer41.use_1x1conv:
            g['layer41_W3'] = self.layer41.conv3.W
            g['layer41_b3'] = self.layer41.conv3.b

        g['layer42_W1'] = self.layer42.conv1.W
        g['layer42_b1'] = self.layer42.conv1.b
        g['layer42_gamma1'] = self.layer42.bn1.gamma
        g['layer42_beta1'] = self.layer42.bn1.beta
        g['layer42_W2'] = self.layer42.conv2.W
        g['layer42_b2'] = self.layer42.conv2.b
        g['layer42_gamma2'] = self.layer42.bn2.gamma
        g['layer42_beta2'] = self.layer42.bn2.beta
        if self.layer42.use_1x1conv:
            g['layer42_W3'] = self.layer42.conv3.W
            g['layer42_b3'] = self.layer42.conv3.b

        g['affine_w'] = self.fc.W
        g['affine_b'] = self.fc.b

        return g
