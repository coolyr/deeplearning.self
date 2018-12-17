import numpy as np
from activators import ReluActivator, IdentityActivator

## 获取卷积区域
def get_patch(input_array, i, j, filter_width, filter_height, stride):
    '''
    从输入数组中获取本次卷积的区域，
    自动适配输入为2D和3D的情况
    '''
    start_i = i * stride
    start_j = j * stride
    if input_array.ndim == 2:
        return input_array[
               start_i: start_i + filter_height,
               start_j: start_j + filter_width]
    elif input_array.ndim == 3:
        return input_array[:,
               start_i: start_i + filter_height,
               start_j: start_j + filter_width]


## 获取一个2D区域的最大值所在的索引
def get_max_index(array):
    max_i = 0
    max_j = 0
    max_value = array[0, 0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] > max_value:
                max_value = array[i, j]
                max_i, max_j = i, j
    return max_i, max_j


# conv函数实现了2维和3维数组的卷积
## 计算卷积
def conv(input_array, kernel_array, output_array, stride, bias):
    '''
    计算卷积，自动适配输入为2D和3D的情况
    '''
    channel_number = input_array.ndim
    output_width = output_array.shape[1]
    output_height = output_array.shape[0]
    kernel_width = kernel_array.shape[-1]
    kernel_height = kernel_array.shape[-2]
    for i in range(output_height):
        for j in range(output_width):
            output_array[i][j] = (
                                     get_patch(input_array, i, j, kernel_width, kernel_height, stride) * kernel_array
                                 ).sum() + bias

# padding函数实现了zero padding操作
## 为数组增加Zero padding
def padding(input_array, zp):
    '''
    为数组增加Zero padding，自动适配输入为2D和3D的情况
    '''
    print("padding zp = ", zp )
    if zp == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            # 获取width, height, depth
            input_width = input_array.shape[2]
            input_height = input_array.shape[1]
            input_depth = input_array.shape[0]
            # print("input_depth : ", type(input_depth))
            # print("input_height : ", type(input_height))
            # print("input_width : ", type(input_width))
            # print("zp : ", type(zp))
            # 扩充2zp的空间
            padded_array = np.zeros((
                input_depth,
                input_height + 2 * zp,
                input_width + 2 * zp))
            # 将原先的input_array复制到padded_array的中间
            padded_array[:,
            zp: zp + input_height,
            zp: zp + input_width] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_width = input_array.shape[1]
            input_height = input_array.shape[0]
            padded_array = np.zeros((
                input_height + 2 * zp,
                input_width + 2 * zp))
            padded_array[zp: zp + input_height,
            zp: zp + input_width] = input_array
            return padded_array

## 对numpy数组进行element wise操作
def element_wise_op(array, op):
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)

##############################################################################################
##############################################################################################
# Filter类保存了卷积层的参数以及梯度，并且实现了用梯度下降算法来更新参数
class Filter(object):
    # 我们对参数的初始化采用了常用的策略，即：权重随机初始化为一个很小的值<-0.0001, 0.0001>，而偏置项初始化为0。
    def __init__(self, width, height, depth):
        self.weights = np.random.uniform(-1e-4, 1e-4, (depth, height, width))
        self.bias = 0
        self.weights_grad = np.zeros(self.weights.shape)
        self.bias_grad = 0
    ##
    def __repr__(self):
        return 'filter weights:\n%s\nbias:\n%s' % (repr(self.weights), repr(self.bias))

    ##
    def get_weights(self):
        return self.weights

    ##
    def get_bias(self):
        return self.bias
    ##
    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad

##############################################################################################
##############################################################################################
# 卷积层反向传播算法的实现
# step-1、将误差项传递到上一层。
# step-2、计算每个参数的梯度。
# step-3、更新参数。

# 卷积层的实现
# 我们用ConvLayer类来实现一个卷积层。下面的代码是初始化一个卷积层，可以在构造函数中设置卷积层的超参数。
class ConvLayer(object):
    #  cl = ConvLayer(5,5,3, 3,3,2, 1, 2, IdentityActivator(), 0.001)
    def __init__(self, input_width, input_height, channel_number,
                 filter_width, filter_height, filter_number,
                 zero_padding, stride,
                 activator,
                 learning_rate):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = ConvLayer.calculate_output_size(self.input_width, filter_width, zero_padding, stride)
        self.output_height = ConvLayer.calculate_output_size(self.input_height, filter_height, zero_padding, stride)
        self.output_array = np.zeros((self.filter_number,  self.output_height, self.output_width))
        self.filters = []
        for i in range(filter_number):
            self.filters.append(Filter(filter_width, filter_height, self.channel_number))
        self.activator = activator
        self.learning_rate = learning_rate

    ## ConvLayer类的forward方法实现了卷积层的前向计算（即计算根据输入来计算卷积层的输出）
    def forward(self, input_array):
        '''
        # input_array : 输入【3, 5，5】
        计算卷积层的输出
        输出结果保存在self.output_array
        '''
        self.input_array = input_array
        # input_array : 输入【3, 5，5】
        # zero_padding : 1
        print("forward padding!")
        self.padded_input_array = padding(input_array, self.zero_padding)
        for f in range(self.filter_number):
            filter = self.filters[f]
            conv(self.padded_input_array,
                 filter.get_weights(), self.output_array[f],
                 self.stride, filter.get_bias())
        # element_wise_op函数实现了对numpy数组进行按元素操作，并将返回值写回到数组中
        element_wise_op(self.output_array, self.activator.forward)

    ##计算传递给前一层的误差项delta，以及计算每个权重的梯度
    def backward(self, input_array, sensitivity_array, activator):
        '''
        计算传递给前一层的误差项，以及计算每个权重的梯度
        前一层的误差项保存在self.delta_array
        梯度保存在Filter对象的weights_grad
        '''
        # print("backward sensitivity map =>\n", sensitivity_array)
        # <1> 前向计算输出
        self.forward(input_array)
        # <2> 将误差项传递到上一层[计算上一层的 sensitivity map]
        self.bp_sensitivity_map(sensitivity_array, activator)
        # <3> 计算每个参数的梯度
        self.bp_gradient(sensitivity_array)

    ## step-3、更新参数。
    # 按照梯度下降算法更新参数的代码
    def update(self):
        '''
        按照梯度下降，更新权重
        '''
        for filter in self.filters:
            filter.update(self.learning_rate)

    ## step-1 将误差项传递到上一层
    def bp_sensitivity_map(self, sensitivity_array, activator):
        '''
        计算传递到上一层的sensitivity map
        sensitivity_array: 本层的sensitivity map
        activator: 上一层的激活函数
        delta_array: 保存传递到上一层的 sensitivity map
        '''
        # 处理卷积步长，对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(sensitivity_array)

        # full卷积，对sensitivitiy map进行zero padding
        # 虽然原始输入的zero padding单元也会获得残差,但这个残差不需要继续向上传递，因此就不计算了
        ###############################################################################
        ## zp 怎么计算的？ 不是 zp==1 么？
        # expanded_width = (self.input_width - self.filter_width + 2 * self.zero_padding + 1)
        # zp = (self.input_width + self.filter_width - 1 - (self.input_width - self.filter_width + 2 * self.zero_padding + 1)) / 2
        # zp = (2*self.filter_width - 2*self.zero_padding - 2) / 2
        # zp = self.filter_width - self.zero_padding - 1
        expanded_width = expanded_array.shape[2]
        zp = (self.input_width + self.filter_width - 1 - expanded_width) / 2
        print("bp_sensitivity_map padding!")
        padded_array = padding(expanded_array, zp=1)
        # padded_array = padding(expanded_array, zp)
        ###############################################################################
        # 初始化delta_array，用于保存传递到上一层的 sensitivity map
        self.delta_array = self.create_delta_array()
        # 对于具有多个filter的卷积层来说，最终传递到上一层的 sensitivity map相当于所有的filter的 sensitivity map之和
        for f in range(self.filter_number):
            filter = self.filters[f]
            # 将filter权重翻转180度
            flipped_weights = np.array(list(map(
                lambda w: np.rot90(w, 2),
                filter.get_weights())))
            # 计算与一个filter对应的delta_array
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                print("###################bp_sensitivity_map###########################")
                print("filter index = ", f)
                print("depth index = ", d)
                print("padded_array shape = ", padded_array.shape)
                print("flipped_weights shape = ", flipped_weights.shape)
                print("delta_array shape = ", delta_array.shape)
                conv(padded_array[f], flipped_weights[d], delta_array[d], 1, 0)
            self.delta_array += delta_array
        # 将计算结果与激活函数的偏导数做element-wise乘法操作
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array, activator.backward)
        # delta[l-1] = delta[l] * W[l] ⭕ f'(net[l-1])
        self.delta_array *= derivative_array
        print("bp_sensitivity_map delta_array =>\n" , self.delta_array)

    ## step-2 计算每个参数的梯度
    def bp_gradient(self, sensitivity_array):
        # 处理卷积步长，对原始sensitivity map进行扩展,『还原』为步长为1
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        for f in range(self.filter_number):
            # 计算每个权重的梯度
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                conv(self.padded_input_array[d],
                     expanded_array[f],
                     filter.weights_grad[d], 1, 0)
            # 计算偏置项的梯度
            filter.bias_grad = expanded_array[f].sum()

    ## 将步长为S的sensitivity map『还原』为步长为1的sensitivity map
    def expand_sensitivity_map(self, sensitivity_array):
        depth = sensitivity_array.shape[0]
        # 确定扩展后sensitivity map的大小
        # 计算stride为1时sensitivity map的大小
        #  W2 = (W1 - F + 2P)/S + 1    =>     W2 = (W1 - F + 2P) + 1
        #  H2 = (H1 - F + 2P)/S + 1    =>     H2 = (H1 - F + 2P) + 1
        expanded_width = (self.input_width - self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height - self.filter_height + 2 * self.zero_padding + 1)
        # 构建新的sensitivity_map
        expand_array = np.zeros((depth, expanded_height, expanded_width))
        # 从原始sensitivity map拷贝误差值delta
        for i in range(self.output_height):
            for j in range(self.output_width):
                i_pos = i * self.stride
                j_pos = j * self.stride
                expand_array[:, i_pos, j_pos] =  sensitivity_array[:, i, j]
        return expand_array

    ## 创建用来保存传递到上一层的sensitivity map的数组
    def create_delta_array(self):
        return np.zeros((self.channel_number, self.input_height, self.input_width))

    ## calculate_output_size函数用来确定卷积层输出的大小
    #  W2 = (W1 - F + 2P)/S + 1
    #  H2 = (H1 - F + 2P)/S + 1
    @staticmethod
    def calculate_output_size(input_size, filter_size, zero_padding, stride):
        return int((input_size - filter_size + 2 * zero_padding) / stride + 1)

    def __str__(self):
        # 打印信息
        ConvLayer_Info = '::ConvLayer Info::\ninput_width: %d\ninput_height: %d\nchannel_number: %d\nfilter_width: %d\nfilter_height: %d\nfilter_number: %d\nzero_padding: %d\nstride: %d\noutput_width: %d\noutput_height: %d\noutput_number: %d\noutput_array: %s\nlearning_rate: %f\n' % \
                         (self.input_width, self.input_height, self.channel_number, self.filter_width, self.filter_height, self.filter_number, self.zero_padding, self.stride,  self.output_width, self.output_height,self.output_array.shape[0],self.output_array.shape, self.learning_rate)
        return ConvLayer_Info

##############################################################################################
##############################################################################################
# Max Pooling层的实现
class MaxPoolingLayer(object):
    # mpl = MaxPoolingLayer(4,4,2, 2,2, 2)
    def __init__(self, input_width, input_height,
                 channel_number, filter_width,
                 filter_height, stride):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = int((input_width -  filter_width) / self.stride + 1)
        self.output_height = int((input_height - filter_height) / self.stride + 1)
        # print(self.channel_number, self.output_height, self.output_width)
        self.output_array = np.zeros((self.channel_number, self.output_height, self.output_width))

    ## MaxPooling的forward
    # 输入【2, 4，4】
    def forward(self, input_array):
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d, i, j] = (
                        get_patch(input_array[d], i, j,
                                  self.filter_width,
                                  self.filter_height,
                                  self.stride).max())

    ## MaxPooling的bp
    # 输入【2, 4，4】
    # sensitivity map【2, 2，2】
    def backward(self, input_array, sensitivity_array):
        self.delta_array = np.zeros(input_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    patch_array = get_patch(
                        input_array[d], i, j,
                        self.filter_width,
                        self.filter_height,
                        self.stride)
                    x, y = get_max_index(patch_array)
                    self.delta_array[d, i * self.stride + x, j * self.stride + y] =  sensitivity_array[d, i, j]

##############################################################################################
##############################################################################################
# 卷积层的测试
def init_test():
    # 输入【3, 5，5】
    a = np.array(
        [[[0, 1, 1, 0, 2],
          [2, 2, 2, 2, 1],
          [1, 0, 0, 2, 0],
          [0, 1, 1, 0, 0],
          [1, 2, 0, 0, 2]],

         [[1, 0, 2, 2, 0],
          [0, 0, 0, 2, 0],
          [1, 2, 1, 2, 1],
          [1, 0, 0, 0, 0],
          [1, 2, 1, 1, 1]],

         [[2, 1, 2, 0, 0],
          [1, 0, 0, 1, 0],
          [0, 2, 1, 0, 1],
          [0, 1, 2, 2, 2],
          [2, 1, 0, 0, 1]]])
    #sensitivity map【2, 3，3】
    b = np.array(
        [[[0, 1, 1],
          [2, 2, 2],
          [1, 0, 0]],

         [[1, 0, 2],
          [0, 0, 0],
          [1, 2, 1]]])
    cl = ConvLayer(5, 5, 3, 3, 3, 2, 1, 2, IdentityActivator(), 0.001)
    # cl = ConvLayer(5, 5, 3, 3, 3, 2, 1, 2, ReluActivator(), 0.001)
    #重新对Filter的权重进行赋值
    cl.filters[0].weights = np.array(
        [[[-1, 1, 0],
          [0, 1, 0],
          [0, 1, 1]],

         [[-1, -1, 0],
          [0, 0, 0],
          [0, -1, 0]],

         [[0, 0, -1],
          [0, 1, 0],
          [1, -1, -1]]], dtype=np.float64)
    cl.filters[0].bias = 1
    cl.filters[1].weights = np.array(
        [[[1, 1, -1],
          [-1, -1, 1],
          [0, -1, 1]],

         [[0, 1, 0],
          [-1, 0, -1],
          [-1, 1, 0]],

         [[-1, 0, 0],
          [-1, 0, 1],
          [-1, 0, 0]]], dtype=np.float64)
    return a, b, cl

##测试卷积层的forward
def test():
    # a : 输入【3, 5，5】
    # sensitivity map【2, 3，3】
    a, b, cl = init_test()
    cl.forward(a)
    print(cl.output_array)

##测试卷基层的bp
def test_bp():
    # a : 输入【5，5，3】
    # sensitivity map【2, 3，3】
    a, b, cl = init_test()
    print(cl)
    cl.backward(a, b, IdentityActivator())
    cl.update()
    print(cl.filters[0])
    print(cl.filters[1])

## 传递给卷积层的sensitivity map是全1数组，留给读者自己推导一下为什么是这样（提示：激活函数选择了identity函数：f(x) = x ）
def gradient_check():
    '''
    梯度检查
    '''
    # 设计一个误差函数，取所有节点输出项之和
    error_function = lambda o: o.sum()
    # 计算forward值
    a, b, cl = init_test()
    # cl.forward(a)
    # 求取sensitivity map
    sensitivity_array = np.ones(cl.output_array.shape, dtype=np.float64)
    # 计算梯度
    cl.backward(a, sensitivity_array, IdentityActivator())
    # 检查梯度
    epsilon = 10e-4
    for f in range(cl.filter_number):
        for d in range(cl.filters[f].weights_grad.shape[0]):
            for i in range(cl.filters[f].weights_grad.shape[1]):
                for j in range(cl.filters[f].weights_grad.shape[2]):
                    cl.filters[f].weights[d, i, j] += epsilon
                    cl.forward(a)
                    err1 = error_function(cl.output_array)
                    cl.filters[f].weights[d, i, j] -= 2 * epsilon
                    cl.forward(a)
                    err2 = error_function(cl.output_array)
                    expect_grad = (err1 - err2) / (2 * epsilon)
                    cl.filters[f].weights[d, i, j] += epsilon
                    print('filter(%d) weights(%d,%d,%d)::   expected : actural  <=> %f : %f' % (f, d, i, j, expect_grad, cl.filters[0].weights_grad[d, i, j]))

##############################################################################################
##############################################################################################
# 池化层的测试
def init_pool_test():
    # 输入【2, 4，4】
    a = np.array(
        [[[1, 1, 2, 4],
          [5, 6, 7, 8],
          [3, 2, 1, 0],
          [1, 2, 3, 4]],

         [[0, 1, 2, 3],
          [4, 5, 6, 7],
          [8, 9, 0, 1],
          [3, 4, 5, 6]]], dtype=np.float64)
    # sensitivity map【2, 2，2】
    b = np.array(
        [[[1, 2],
          [2, 4]],

         [[3, 5],
          [8, 2]]], dtype=np.float64)
    mpl = MaxPoolingLayer(4, 4, 2, 2, 2, 2)
    return a, b, mpl

def test_pool():
    a, b, mpl = init_pool_test()
    mpl.forward(a)
    print('input array:\n%s\noutput array:\n%s' % (a, mpl.output_array))

def test_pool_bp():
    a, b, mpl = init_pool_test()
    mpl.backward(a, b)
    print('input array:\n%s\nsensitivity array:\n%s\ndelta array:\n%s' % (a, b, mpl.delta_array))

if __name__ == '__main__':
    #测试卷积层
    # test()
    # test_bp()
    # gradient_check()

    #测试池化层
    # test_pool()
    test_pool_bp()

