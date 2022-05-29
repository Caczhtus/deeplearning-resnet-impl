# 基于ResNet18的数字图像检测

## 简介

基于『common』包下的卷积层、池化层等基础层，实现了何凯明博士在『Deep Residual Learning for Image Recognition』论文中提到的残差单元，并拼接为『ResNet18』

## 快速开始

1. 下载数据集

```
python mnist.py
```

2. 训练并测试

```python
python unit_test.py
```

## 代码结构

```
.
├── common
│   ├── functions.py
│   ├── gradient.py
│   ├── __init__.py
│   ├── layers.py
│   ├── multi_layer_net_extend.py
│   ├── multi_layer_net.py
│   ├── optimizer.py
│   ├── __pycache__
│   │   ├── functions.cpython-38.pyc
│   │   ├── __init__.cpython-38.pyc
│   │   ├── layers.cpython-38.pyc
│   │   ├── optimizer.cpython-38.pyc
│   │   ├── trainer.cpython-38.pyc
│   │   └── util.cpython-38.pyc
│   ├── trainer.py
│   └── util.py
├── mnist.pkl
├── mnist.py
├── __pycache__
│   ├── mnist.cpython-38.pyc
│   ├── Residual.cpython-38.pyc
│   └── ResNet18.cpython-38.pyc
├── Residual.py
├── ResNet18.py
├── t10k-images-idx3-ubyte.gz
├── t10k-labels-idx1-ubyte.gz
├── train-images-idx3-ubyte.gz
├── train-labels-idx1-ubyte.gz
└── unit_test.py
```

## 说明

目前本代码还有很多不足，仅供学习参考。后续有能力再进行优化补全。

- base-resnet-block
- resnet18
- 普通卷积层
- 平均池化层
- 最大池化层
- 线性变换层
- 训练器

## 参考

1. [https://arxiv.org/pdf/1512.03385.pdf](https://arxiv.org/pdf/1512.03385.pdf)
2. [ZhangXinNan/deep_learning_from_scratch](https://github.com/ZhangXinNan/deep_learning_from_scratch)
3. 《深度学习入门——基于Python的理论与实现》作者：斋藤康毅 译者：陆宇杰