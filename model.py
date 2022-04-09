# numpy 实现两层神经网络
import numpy as np

class fc2(object):
    def __init__(self, input_size, hidden_size, output_size, beta):
        self.w1 = 1e-3*np.random.rand(input_size, hidden_size)
        self.w2 = 1e-3*np.random.rand(hidden_size, output_size)
        self.b1 = 1e-3*np.zeros(hidden_size)
        self.b2 = 1e-3*np.zeros(output_size)
        self.beta = beta  # l2系数
        self.nuerous = {}
        self.gradients = {}

    def update_param(self, lr, gradients):
        self.w1 -= lr * gradients['w1']
        self.b1 -= lr * gradients['b1']
        self.w2 -= lr * gradients['w2']
        self.b2 -= lr * gradients['b2']

    def get_weights(self):
        return {'w1': self.w1,
                'b1': self.b1,
                'w2': self.w2,
                'b2': self.b2}

    def fc(self, x, W, b):
        return np.dot(x, W) + b

    def fc_backward(self, next_dx, W, x):
        N = x.shape[0]
        dx = np.dot(next_dx, W.T)  # 当前层的梯度
        dw = np.dot(x.T, next_dx) + self.beta*W  # 当前层权重的梯度
        db = np.sum(next_dx, axis=0)  # 当前层偏置的梯度, N个样本的梯度求和
        return dw / N, db / N, dx

    def relu(self, x):
        return np.maximum(0, x)

    def relu_backward(self, next_dx, x):
        dx = np.where(np.greater(x, 0), next_dx, 0)
        return dx

    def softmax(self, x):
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    def cross_entropy_loss(self, y_predict, y_true):
        y_shift = y_predict - np.max(y_predict, axis=-1, keepdims=True)
        y_exp = np.exp(y_shift)
        y_probability = y_exp / np.sum(y_exp, axis=-1, keepdims=True)
        loss = np.mean(np.sum(-y_true * np.log(y_probability), axis=-1))  # 损失函数
        l2_loss = self.beta*(np.sum(np.square(self.w1))+np.sum(np.square(self.w2)))/y_true.shape[0]/2
        loss += l2_loss
        dy = y_probability - y_true
        return loss, dy


    def model_forward(self, X):
        self.nuerous['z1'] = self.fc(X, self.w1, self.b1)
        self.nuerous['z1_relu'] = self.relu(self.nuerous['z1'])

        self.nuerous['y'] = self.fc(self.nuerous['z1_relu'], self.w2, self.b2)
        return self.nuerous['y']


    def model_backward(self, X, y_true):
        loss, dy = self.cross_entropy_loss(self.nuerous['y'], y_true)
        self.gradients['w2'], self.gradients['b2'], self.gradients['z1_relu'] = \
        self.fc_backward(dy, self.w2, self.nuerous['z1_relu'])
        self.gradients['z1'] = self.relu_backward(self.gradients['z1_relu'], self.nuerous['z1'])
        self.gradients["w1"], self.gradients["b1"], _ = \
            self.fc_backward(self.gradients["z1"], self.w1, X)
        return loss, self.gradients


    def predict_accuracy(self, X, y_true):
        y_predict = self.model_forward(X)
        return np.mean(np.equal(np.argmax(y_predict, axis=-1),
                                np.argmax(y_true, axis=-1)))







