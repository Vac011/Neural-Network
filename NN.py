import numpy
import scipy.special
# import matplotlib.pyplot

class NeuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learnrate):
        # 设置输入层、隐藏层和输出层节点数
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # 设置学习率
        self.l = learnrate
        # 初始化权重
        # 以较复杂的方式初始化权重，使用正态分布，均值为0，标准差为1/sqrt(链接数)
        self.wi2h = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # 以较简单的方式初始化权重，使用均匀分布，范围为[-0.5, 0.5]
        self.wh2o = (numpy.random.rand(self.onodes, self.hnodes) - 0.5)
        # 设置激活函数为sigmoid函数
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        # 训练过程分为两步：第一步前向计算输出，第二步根据输出计算得到的误差反向传播来更新权重
        # 将输入列表转换为2维数组（n*1）
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # 计算隐藏层
        hidden_inputs = numpy.dot(self.wi2h, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层
        final_inputs = numpy.dot(self.wh2o, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        # 计算输出层误差
        output_errors = targets - final_outputs
        # 计算隐藏层误差
        hidden_errors = numpy.dot(self.wh2o.T, output_errors)
        # 更新隐藏层和输出层之间的权重
        self.wh2o += self.l * numpy.dot(output_errors * final_outputs * (1.0 - final_outputs), hidden_outputs.T)
        # 更新输入层和隐藏层之间的权重
        self.wi2h += self.l * numpy.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), inputs.T)

    def predict(self, inputs_list):
        # 预测过程即训练过程的前向计算部分
        # 将输入列表转换为2维数组（n*1）
        inputs = numpy.array(inputs_list, ndmin=2).T
        # 计算隐藏层
        hidden_inputs = numpy.dot(self.wi2h, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算输出层
        final_inputs = numpy.dot(self.wh2o, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
    
    def save_parameters(self, filename):
        # 保存参数到文件
        numpy.savez(filename, wi2h=self.wi2h, wh2o=self.wh2o)

    def load_parameters(self, filename):
        # 从文件中加载参数
        data = numpy.load(filename)
        self.wi2h = data['wi2h']
        self.wh2o = data['wh2o']

# 定义参数
inputnodes = 28 * 28  # 输入层节点数，每个手写图片分辨率为28*28，即每个输入节点对应一个像素点
hiddennodes = 200     # 隐藏层节点数，由于隐藏层一般对输入进行特征提取，所以节点数一般小于输入层节点数
outputnodes = 10      # 输出层节点数，每个输出节点代表对应数字的概率
learnrate = 0.02      # 学习率，即每次更新权重时的步长(系数)，过高可能会在最低点附近震荡，过低则收敛速度慢
epochs = 5            # 训练次数，通常训练次数越多，模型越准确，但也可能导致过拟合

# 创建神经网络对象
nn = NeuralNetwork(inputnodes, hiddennodes, outputnodes, learnrate)

# 训练过程
train_data_file = open("mnist_dataset/mnist_train.csv", 'r')
train_data_lists = train_data_file.readlines()
train_data_file.close()
for e in range(epochs):
    for data_list in train_data_lists:
        # 规范化输入数据，将像素值从0-255转换到0.01-1.0之间，先乘0.99再加0.01将数据限制在0.01-1.0之间，避免0值输入导致权重无法更新
        inputs = (numpy.asarray(list(map(float, data_list.split(',')[1:]))) / 255.0 * 0.99) + 0.01
        # 将目标值限制在0.01和0.99之间，由于sigmoid函数输出为(0,1)，避免0或1目标输出导致权重过分更新
        targets = numpy.zeros(outputnodes) + 0.01
        targets[int(data_list.split(',')[0])] = 0.99
        # 训练网络
        nn.train(inputs, targets)

# 测试过程
test_data_file = open("mnist_dataset/mnist_test.csv", 'r')
test_data_lists = test_data_file.readlines()
test_data_file.close()
score = 0
for data_list in test_data_lists:
    inputs = (numpy.asarray(list(map(float, data_list.split(',')[1:]))) / 255.0 * 0.99) + 0.01
    targetnum = int(data_list.split(',')[0])
    outputs = nn.predict(inputs)
    label = numpy.argmax(outputs)
    if (label == targetnum):
        score += 1
# 计算准确率
print("performance = ", score / len(test_data_lists) * 100, "%")
