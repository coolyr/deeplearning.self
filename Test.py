from functools import reduce
import random
import numpy as np

def char2num(s):
    return {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}[s]

def str2int(s):
    # print( list(map(char2num, s)))
    return reduce(lambda x,y: x*10+y, map(char2num, s))


def testZip():
    X = [[1,1], [1,2], [2, 2]]
    Y = [0, 1, 1]
    saples = zip(X, Y)
    print(list(saples))

class Normalizer(object):
    def __init__(self):
        # mask =  [1, 2, 4, 8, 16, 32, 64, 128]
        self.mask = [0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80]
        # print("mask = ", self.mask)

    def norm(self, number):
        return list(map(lambda m: 0.9 if number & m else 0.1, self.mask))

    def denorm(self, vec):
        binary = map(lambda i: 1 if i > 0.5 else 0, vec)
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x, y: x + y, binary)

#生成（获取）训练数据集
def train_data_set():
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(0, 256, 8):
        # print(i)
        n = normalizer.norm(int(random.uniform(0, 256)))
        # print(int(random.uniform(0, 256)))
        data_set.append(n)
        labels.append(n)

    print("data_set = ",data_set)
    print("labels = ", labels)
    return labels, data_set


# Sigmoid激活函数类
class SigmoidActivator(object):
    def forward(self, weighted_input):
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        return output * (1 - output)

#转化成列向量
def transpose(args):
    return list(map(
        lambda arg: list(map(lambda line: np.array(line).reshape(len(line), 1), arg)),
        args
    ))

def train_data_set():
    normalizer = Normalizer()
    data_set = []
    labels = []
    for i in range(0, 2):
        n = normalizer.norm(i)
        data_set.append(n)
        labels.append(n)
    return labels, data_set


def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)

class ReluActivator(object):
    def forward(self, weighted_input):
        #return weighted_input
        return max(0, weighted_input)
    def backward(self, output):
        return 1 if output > 0 else 0

if __name__ == '__main__':
    # num = str2int("123321")
    # print(num)
    # testZip()
    # input_vec = [1,2,1]
    # weights = [0.2, 0.3, 0.5]
    # # weights = [0.0 for _ in range(2)]
    # print(weights)
    # bias = 0.6
    # print(list(zip(input_vec, weights)))
    # print(list(map(lambda xw: xw[0] * xw[1], list(zip(input_vec, weights)))))
    # print(reduce(lambda a, b: a + b, map(lambda xw: xw[0] * xw[1], list(zip(input_vec, weights)))))
    # print(reduce(lambda a, b: a + b, list(map(lambda xw: xw[0] * xw[1], list(zip(input_vec, weights))))) + bias)
    # print(reduce(lambda x, y: x+y, [1,2,3,4,5]))

    # print(random.uniform(-0.1, 0.1))
    # train_data_set()
    # upstreams = [2, 1, 2]
    # output = reduce(lambda ret, upstream: ret + upstream * upstream, upstreams, 0)
    # print("output = ", output)
    # for i in [1,2,3,4,5][:-1]:
    #     print(i)
    #
    # print([x for x in range(0, 256, 8)])
    # print(len([x for x in range(0, 256, 8)]))
    W = np.random.uniform(-0.1, 0.1, (2, 3))
    print(W)
    b = np.zeros((2, 1))
    print(b)
    activator = SigmoidActivator()
    inputArray = np.random.uniform(-0.1, 0.1, (3, 1))
    print(W.shape, inputArray.shape, b.shape)
    WXb = np.dot(W, inputArray) + b
    print(WXb)
    output = activator.forward(np.dot(W, inputArray) + b)
    print("output = ", output)
    # V = [1,2,3,4,5,6,7,8,9]
    # print(V[::-1])
    # sets = train_data_set()
    # print("sets = ", sets)
    # labels, data_set = transpose(train_data_set())
    # print("labels = ", labels)
    # print("data_set = ", data_set)

    X = np.zeros((2,3,4))
    print("X => ", type(X[0][0][0]))

    weights = np.random.uniform(-1e-4, 1e-4, (2, 3, 3))
    print("weight -> ", weights)
    # print(-1e-4)
    # print(1e-4)
    element_wise_op(weights, ReluActivator().forward)
    print("element_wise_op(weight) -> ", weights)

    print(int(5/2))
    sensitivity_array = np.ones((2,3,3), dtype=np.float64)
    print("sensitivity_array =>\n",sensitivity_array)

    print(" ## ")
    for k in range(2, 0, -1):
        print(k)
    print(" ## ")
    sensitivity_array = np.array([[1], [2]])
    print(sensitivity_array.shape)
    for i in range(2):
        print(i)

    t = [1,2,3,4,5]
    t[-1] = 6
    print("t = ", t)
    print("t[-1] = ", t[-1])

    times = 5
    for k in range(times, 0, -1):
        print(k)


    print(np.random.randn())
    print(-4*np.random.rand())
