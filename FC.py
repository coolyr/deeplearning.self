import random
from functools import reduce

import numpy as np

# Sigmoid激活函数类
class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)

# 全连接层实现类
# 这个类一举取代了原先的Layer、Node、Connection等类，不但代码更加容易理解，而且运行速度也快了几百倍
class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, activator):
        '''
        构造函数
        input_size: 本层输入向量的维度
        output_size: 本层输出向量的维度
        activator: 激活函数
        ## 8 - 3 - 8
        ## FullConnectedLayer(8 ,3, SigmoidActivator)
        ## FullConnectedLayer(3 ,8, SigmoidActivator)
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator

        # 权重数组W
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        # 偏置项b(初始化为0)
        self.b = np.zeros((output_size, 1))
        # 输出向量（初始化为0）
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        '''
        前向计算
        input_array: 输入向量，维度必须等于input_size
        '''

        # 式2*
        # o-o  的形式，input为左边节点o的输出值，output为右边节点o的输出值（也是下一层节点的输入值）
        self.input = input_array
        self.output = self.activator.forward(np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        '''
        反向计算W和b的梯度
        delta_array: 从上一层传递过来的误差项
        '''
        # 式8*   delta[l] = a[l](1-a[l]) * W.T * delta[l+1]
        #  o-o  的形式,计算的为左边节点o的delta
        self.delta = self.activator.backward(self.input) * np.dot(self.W.T, delta_array) #连接上游节点的delta
        # 式9*   grad[t+1] = grad[t] + rate * delta * X.T
        #  o-o  的形式,计算的为右边节点o的权重W和偏置项b的梯度grad_w  &  grad_b
        self.W_grad = np.dot(delta_array, self.input.T)
        # 式10*  b = b + rate * delta
        self.b_grad = delta_array


    def update(self, learning_rate):
        '''
        使用梯度下降算法更新权重
        '''
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad

    def dump(self):
        print('W: %s\nb:%s' % (self.W, self.b))

##############################################################################################
##############################################################################################

# 神经网络类
# 我们对Network类稍作修改，使之用到FullConnectedLayer
class Network(object):
    def __init__(self, layers):
        '''
        构造函数
        # 8 - 3 - 8
        # FullConnectedLayer(8 ,3, SigmoidActivator)
        # FullConnectedLayer(3 ,8, SigmoidActivator)
        '''
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(
                    layers[i], layers[i+1],
                    SigmoidActivator()
                )
            )

    # 预测输出
    def predict(self, sample):
        '''
        使用神经网络实现预测
        sample: 输入样本
        '''
        output = sample
        for layer in self.layers:
            layer.forward(output) #上一层的输出 等于 下一层的输入
            output = layer.output
        return output

    # 训练神经网络
    def train(self, labels, data_set, rate, epoch):
        '''
        训练函数
        labels: 样本标签
        data_set: 输入样本
        rate: 学习速率
        epoch: 训练轮数 - mini_batch
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)

    # 训练一个样本
    def train_one_sample(self, label, sample, rate):
        self.predict(sample)        # ->
        self.calc_gradient(label)   # <-
        self.update_weight(rate)    # ->

    # 计算梯度
    def calc_gradient(self, label):
        # 输出层delta = output * (1 - output) * (label - output)
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * \
                (label - self.layers[-1].output)
        for layer in self.layers[::-1]: #逆向计算各层delta
            layer.backward(delta)
            delta = layer.delta
        return delta

    # 前向更新各层权重
    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)

    def dump(self):
        for layer in self.layers:
            layer.dump()

    # 计算损失函数：均方误差
    def loss(self, label, output):
        return 0.5 * ((label - output) * (label - output)).sum()

    # 梯度检查
    def gradient_check(self, sample_feature, sample_label):
        '''
        梯度检查
        network: 神经网络对象
        sample_feature: 样本的特征
        sample_label: 样本的标签
        '''
        # 获取网络在当前样本下每个连接的梯度
        self.predict(sample_feature)
        self.calc_gradient(sample_label)

        # 检查梯度
        epsilon = 10e-4
        for fc in self.layers:
            for i in range(fc.W.shape[0]): #output
                for j in range(fc.W.shape[1]): #input
                    fc.W[i,j] += epsilon
                    output = self.predict(sample_feature)
                    err1 = self.loss(sample_label, output)
                    fc.W[i,j] -= 2*epsilon
                    output = self.predict(sample_feature)
                    err2 = self.loss(sample_label, output)
                    expect_grad = (err1 - err2) / (2 * epsilon)
                    fc.W[i,j] += epsilon
                    # print('weights(%d,%d): <expected:%.4e  -  actural:%.4e>' % (i, j, expect_grad, fc.W_grad[i, j]))
                    print('weights(%d,%d): <expected:%f  -  actural:%f>' % (i, j, expect_grad, fc.W_grad[i, j]))
                    # print('weights(%d,%d): expected - actural %.4e - %.4e' % (i, j, expect_grad, fc.W_grad[i, j]))
                    #?为什么差一个负号呢？？

##############################################################################################
##############################################################################################
# from BP import train_data_set
# 行向量 转置成 列向量
def transpose(args):
    return list(map(
        lambda arg: list(map(lambda line: np.array(line).reshape(len(line), 1), arg)),
        args
    ))
##############################################################################################
##############################################################################################
# 正规化器
class Normalizer(object):
    def __init__(self):
        # mask =  [1, 2, 4, 8, 16, 32, 64, 128]
        self.mask = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]

    def norm(self, number):
        data = list(map(lambda m: 0.9 if number & m else 0.1, self.mask))
        return np.array(data).reshape(8, 1)

    def denorm(self, vec):
        binary = list(map(lambda i: 1 if i > 0.5 else 0, vec[:,0]))
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x,y: x + y, binary)
##############################################################################################
##############################################################################################
# 获取训练数据集
def train_data_set():
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(0, 256):
        n = normalizer.norm(i)
        data_set.append(n)
        labels.append(n)
    return labels, data_set

# 计算准确率
def correct_ratio(network):
    normalizer = Normalizer()
    correct = 0.0
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print('correct_ratio: %.2f%%' % (correct / 256 * 100))


#测试向量化神经网络
def test():
    # 256个样本 转置
    labels, data_set = transpose(train_data_set())
    net = Network([8, 3, 8])
    rate = 0.5
    mini_batch = 20
    epoch = 10
    for i in range(epoch): #实际训练的轮数 turns = mini_batch*epoch
        net.train(labels, data_set, rate, mini_batch)
        print('after epoch %d loss: %f' % (
            (i + 1),
            net.loss(labels[-1], net.predict(data_set[-1]))
        ))
        # 每一次mini_batch学习率要缩小1倍
        rate /= 2
    #计算准确率
    correct_ratio(net)
    return  net

def gradient_check(net):
    '''
    梯度检查
    '''
    labels, data_set = transpose(train_data_set())
    # net = Network([8, 3, 8])
    net.gradient_check(data_set[0], labels[0])
    return net



if __name__ == '__main__':
    #训练神经网络
    net = test()

    #运行梯度检验
    gradient_check(net)
