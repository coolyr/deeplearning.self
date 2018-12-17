import random
from numpy import *
from functools import reduce

# Network 神经网络对象，提供API接口。它由若干层对象组成以及连接对象组成。
# Layer 层对象，由多个节点组成。
# Node 节点对象计算和记录节点自身的信息(比如输出值、误差项等)，以及与这个节点相关的上下游的连接。
# Connection 每个连接对象都要记录该连接的权重。
# Connections 仅仅作为Connection的集合对象，提供一些集合操作。

def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))

##############################################################################################
##############################################################################################
# 节点类，负责记录和维护节点自身信息以及与这个节点相关的上下游连接，实现输出值和误差项的计算。
class Node(object):
    def __init__(self, layer_index, node_index):
        # 构造节点对象。
        # layer_index: 节点所属的层的编号
        # node_index: 节点的编号
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = [] #保存Node的下游连接conn
        self.upstream = []   #保存Node的上游连接conn
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        # 设置节点的输出值。如果节点属于输入层会用到这个函数。
        self.output = output

    def append_downstream_connection(self, conn):
        # 添加一个到下游节点的连接
        self.downstream.append(conn)

    def append_upstream_connection(self, conn):
        # 添加一个到上游节点的连接
        self.upstream.append(conn)

    def calc_output(self):
        # 根据式*1计算节点的输出
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0) #reduce语法没看懂！！
        self.output = sigmoid(output)

    #计算隐层节点的delta
    def calc_hidden_layer_delta(self):
        # 节点属于隐藏层时，根据式4*计算delta
        # delta = y(1-y)*SUM(W[j,i] * DELTA[j]) => 下游所有DELTA[j]通过权向量W[j,i]求和到当前节点i,下游节点不包括偏置节点
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    #计算输出层节点的delta
    def calc_output_layer_delta(self, label):
        # 节点属于输出层时，根据式3*计算delta
        # delta = (t-y)y(1-y)
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        # 打印节点的信息
        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str

##############################################################################################
##############################################################################################
# ConstNode对象，为了实现一个输出恒为1的节点(计算偏置项Wb时需要)
class ConstNode(object):
    def __init__(self, layer_index, node_index):
        # 构造节点对象。
        # layer_index: 节点所属的层的编号
        # node_index: 节点的编号
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = [] # 只有输出conn，没有输入conn
        self.output = 1      # 输出恒为1
        # 无delta,不需要后向传播
        # 无upstream，没有上游conn

    def append_downstream_connection(self, conn):
        # 添加一个到下游节点的连接
        self.downstream.append(conn)

    # 计算隐层偏置节点的delta
    def calc_hidden_layer_delta(self):
        # 节点属于隐藏层时，根据式4*计算偏置节点的delta
        # delta = y(1-y)*SUM(W[j,i] * DELTA[j]) => 下游所有DELTA[j]通过权向量W[j,i]求和到当前节点i,下游节点不包括偏置节点
        downstream_delta = reduce(
            lambda ret, conn: ret + conn.downstream_node.delta * conn.weight,
            self.downstream, 0.0)
        # 因为偏置节点的output=1, 所以delta=0. => 该项计算无实际效果，所以在定义ConstNode时没有定义delta
        # 但是为了和其它节点统一起来，所以也添加了该函数
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        # 打印节点的信息
        node_str = '%u-%u: output: 1' % (self.layer_index, self.node_index)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str

##############################################################################################
##############################################################################################
# Layer对象，负责初始化一层。此外，作为Node的集合对象，提供对Node集合的操作。
class Layer(object):
    def __init__(self, layer_index, node_count):
        # 初始化一层
        # layer_index: 层编号
        # node_count: 层所包含的节点个数
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        self.nodes.append(ConstNode(layer_index, node_count))#添加偏置节点Wb

    def set_output(self, data):
        # 设置层的输出。当层是输入层时会用到。
        # len(data)为输入节点数，不包括偏置节点，偏置节点输出恒为1
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        # 计算层的输出向量
        for node in self.nodes[:-1]:#除了偏置节点[:-1]外计算每个节点的输出
            node.calc_output()

    def dump(self):
        # 打印层的信息
        for node in self.nodes:
            print(node)

##############################################################################################
##############################################################################################
# Connection对象，主要职责是记录连接的权重，以及这个连接所关联的上下游节点。
class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        # 初始化连接，权重初始化为是一个很小的随机数
        # upstream_node: 连接的上游节点
        # downstream_node: 连接的下游节点
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1) #uniform(x, y)方法将随机生成下一个实数，它在[x,y]范围内
        self.gradient = 0.0 # 梯度 grant = delta * x

    def calc_gradient(self):
        # 计算梯度
        # grant = delta * x （grant本身是对权重求导，现在分解成对净输入net求导得到的delta和净输入net对权重求导得到的x的乘积形式）
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def update_weight(self, rate):
        # 获取当前的梯度
        self.calc_gradient()
        self.weight += rate * self.gradient

    def get_gradient(self):
        return self.gradient

    def __str__(self):
        # 打印连接信息
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,
            self.weight)

##############################################################################################
##############################################################################################
# Connections对象，提供Connection集合操作。
class Connections(object):
    def __init__(self):
        self.connections = []

    def add_connection(self, connection):
        self.connections.append(connection)

    def dump(self):
        for conn in self.connections:
            print(conn)

##############################################################################################
##############################################################################################
# Network对象，提供API
class Network(object):
    ## layers = [8, 3, 8]
    def __init__(self, layers):
        # 初始化一个全连接神经网络
        # layers: 二维数组，描述神经网络每层节点数
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        #node_count = 0;
        #初始化各层Node节点,初始化所在层索引layer_index和节点索引node_index
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i])) #输出层也添加了一个偏置节点（不需要！）
        #建立各层Layer之间的连接关系
        for layer in range(layer_count - 1):
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]] #最有一个为计算偏执项Wb的ConstNode节点，不需要输入，所以[:-1]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn) #将连接信息加到上游节点上
                conn.upstream_node.append_downstream_connection(conn) #将连接信息加到下游节点上

    #训练神经网络
    def train(self, labels, data_set, rate, epoch):
        # labels: 数组，训练样本标签。每个元素是一个样本的标签。
        # data_set: 二维数组，训练样本特征。每个元素是一个样本的特征。
        # rate: 学习率
        # epoch: 迭代轮数
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d], data_set[d], rate)
                print('Turns %d : sample %d training finished' % (i, d))
    #训练一个样本
    def train_one_sample(self, label, sample, rate):
        # 内部函数，用一个样本训练网络
        self.predict(sample)        # ->
        self.calc_delta(label)      # <-
        self.update_weight(rate)    # ->

    # 前向传播 - 计算各层输出
    def predict(self, sample):
        # 根据输入的样本预测输出值
        # sample: 数组，样本的特征，也就是网络的输入向量
        self.layers[0].set_output(sample) #设置输入层输出，不包括偏置节点（输出恒为1）
        for i in range(1, len(self.layers)): #前向传播 - 计算各层输出
            self.layers[i].calc_output()
        return list(map(lambda node: node.output, self.layers[-1].nodes[:-1]))#返回输出层除偏置节点ConstNode对象的所有output组成的list

    #计算每个节点的delta
    def calc_delta(self, label):
        # 内部函数，计算每个节点的delta
        output_nodes = self.layers[-1].nodes #返回输出层除偏置节点外的所有输出节点
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        #从后向前计算各层的节点的delta
        for layer in self.layers[-2::-1]: #[::-1]表示逆向列表, 所以[-2::-1]代表逆向从倒数第二个开始直至第一个
            for node in layer.nodes: #包括偏置节点delta的计算
                node.calc_hidden_layer_delta()

    #更新每个连接的权重
    def update_weight(self, rate):
        # 内部函数，更新每个连接权重
        # 从前先后更新
        for layer in self.layers[:-1]:# 不包括输出层
            for node in layer.nodes:# 每个节点
                for conn in node.downstream:# 每一个下游conn
                    conn.update_weight(rate)# 更新conn的权重

    def get_gradient(self, label, sample):
        # 获得网络在一个样本下，每个连接上的梯度
        # label: 样本标签
        # sample: 样本输入
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def calc_gradient(self):
        # 内部函数，计算每个连接的梯度
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()


    def dump(self):
        # 打印网络信息
        for layer in self.layers:
            layer.dump()

##############################################################################################
##############################################################################################
# 正规化器
class Normalizer(object):
    def __init__(self):
        # mask =  [1, 2, 4, 8, 16, 32, 64, 128]
        self.mask = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]

    def norm(self, number):
        # [0.1, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.1]
        return list(map(lambda m: 0.9 if number & m else 0.1, self.mask))

    def denorm(self, vec):
        binary = list(map(lambda i: 1 if i > 0.5 else 0, vec))
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x, y: x + y, binary)

##############################################################################################
##############################################################################################
#计算两个向量的均方误差
def mean_square_error(vec1, vec2):
    return 0.5 * reduce(lambda a, b: a + b,
                        list(map(lambda v: (v[0] - v[1]) * (v[0] - v[1]),
                            zip(vec1, vec2)
                            ))
                        )

#梯度检验
def gradient_check(network, sample_feature, sample_label):
    # 梯度检查
    # network: 神经网络对象
    # sample_feature: 样本的特征
    # sample_label: 样本的标签

    # 计算网络误差
    network_error = lambda vec1, vec2: 0.5 * reduce(lambda a, b: a + b, list(map(lambda v: (v[0] - v[1]) * (v[0] - v[1]), zip(vec1, vec2))))

    # 获取网络在当前样本下每个连接的梯度
    network.get_gradient(sample_feature, sample_label)

    # 对每个权重做梯度检查
    for conn in network.connections.connections:
        # 获取指定连接的梯度
        actual_gradient = conn.get_gradient()
        # 增加一个很小的值，计算网络的误差(至少4位有效数字相同！！)
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature), sample_label)
        # 减去一个很小的值，计算网络的误差
        conn.weight -= 2 * epsilon  # 刚才加过了一次，因此这里需要减去2倍
        error2 = network_error(network.predict(sample_feature), sample_label)
        # 根据式6*计算期望的梯度值
        expected_gradient = (error2 - error1) / (2 * epsilon)
        # 打印
        print('%s:\n\t\t\texpected gradient: \t%f\n\t\t\tactual gradient: \t%f' % (str(conn), expected_gradient, actual_gradient))

#生成（获取）训练数据集
def train_data_set():
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(0, 256, 8): #[0, 8, 16, ... , 240, 248]
        n = normalizer.norm(int(random.uniform(0, 256)))
        data_set.append(n)  # [0.1, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.1]
        labels.append(n)    # [0.1, 0.9, 0.9, 0.1, 0.1, 0.9, 0.9, 0.1]
    return labels, data_set


#训练网络
def train(network):
    labels, data_set = train_data_set()
    network.train(labels, data_set, 0.3, 50)

#测试网络单个样本
def testSample(network, data):
    normalizer = Normalizer()
    norm_data = normalizer.norm(data)
    predict_data = network.predict(norm_data)
    print('\n测试单个样本\ttestdata(%u)\tpredict(%u)' % (data, normalizer.denorm(predict_data)))


#计算准确率
def correct_ratio(network):
    normalizer = Normalizer()
    correct = 0.0;
    for i in range(256):
        if normalizer.denorm(network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print('correct_ratio: %.2f%%' % (correct / 256 * 100))

#运行梯度检验
def gradient_check_test():
    net = Network([2, 2, 2])
    sample_feature = [0.9, 0.1]
    sample_label = [0.9, 0.1]
    gradient_check(net, sample_feature, sample_label)

if __name__ == '__main__':
    #训练神经网络
    net = Network([8, 3, 8])
    train(net)
    net.dump()
    testSample(net, 5)
    #测试准确率
    correct_ratio(net)

    #运行梯度检验
    gradient_check_test()



