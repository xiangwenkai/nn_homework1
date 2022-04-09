import gzip
import pickle
import numpy as np
from model import fc2
import sys
import matplotlib.pyplot as plt
import pickle

plt.style.use('ggplot')

def to_categorical(y):
    num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    return categorical

def batch_choice(train_size, batch_size):
    idx = np.random.choice(train_size, batch_size)
    return train_set[0][idx], train_y[idx]


if __name__ == "__main__":

    # 获取mnist数据
    path = 'data/mnist.pkl.gz'
    with gzip.open(path, 'rb') as f:
        train_set, val_set, test_set = pickle.load(f, encoding='latin1')

    train_y, val_y, test_y = to_categorical(train_set[1]), to_categorical(val_set[1]), to_categorical(test_set[1])
    train_size = train_set[0].shape[0]

    input_size = train_set[0].shape[1]
    num_class = train_y.shape[1]

    def train(lr, hidden_size, beta):
        net = fc2(input_size, hidden_size, num_class, beta=beta)

        batch_size = 32
        epoch = 5
        steps = train_size // batch_size

        for e in range(epoch):
            lr_decay = lr*0.6**e
            for s in range(steps):
                X, y = batch_choice(train_size, batch_size)

                net.model_forward(X)

                loss, gradients = net.model_backward(X=X, y_true=y)

                net.update_param(lr_decay, gradients)
                #
                # if s % 1000 == 0:
                #     print(f"epoch:{e} step:{s} ; loss:{loss}")
                #     print(f"train_acc:{net.predict_accuracy(X, y)};  val_acc:{net.predict_accuracy(val_set[0], val_y)}")

        print("result train_acc:{};  val_acc:{}".
              format(net.predict_accuracy(train_set[0], train_y), net.predict_accuracy(val_set[0], val_y)))
        return net.predict_accuracy(val_set[0], val_y), net

    # 参数搜索
    def params_tuning():
        max_acc = 0
        for lr in [0.3, 0.2, 0.1]:
            for hidden_size in [32,64,128]:
                for beta in [0.1, 0.05, 0.03, 0.01]:
                    val_acc, _ = train(lr, hidden_size, beta)
                    if val_acc > max_acc:
                        max_acc = val_acc
                        best_lr = lr
                        best_hidden_size = hidden_size
                        best_beta = beta

        # 用最优参数训练并预测test
        _, net = train(best_lr, best_hidden_size, best_beta)
        test_acc = net.predict_accuracy(test_set[0], test_y)
        print(f"The best params are \n lr: {best_lr} \n hidden_size: {best_hidden_size} \n "
              f"beta: {best_beta} \n The test accuracy is {test_acc: .4f}")
        return best_lr, best_hidden_size, best_beta

    # 训练loss曲线和测试正确率曲线
    def train_plot(lr, hidden_size, beta):
        net = fc2(input_size, hidden_size, num_class, beta=beta)

        batch_size = 32
        epoch = 5
        steps = train_size // batch_size
        train_loss = []
        val_loss = []
        test_acc = []
        for e in range(epoch):
            lr_decay = lr * 0.8 ** e
            for s in range(steps):
                X, y = batch_choice(train_size, batch_size)

                net.model_forward(X)

                loss, gradients = net.model_backward(X=X, y_true=y)

                net.update_param(lr_decay, gradients)

                if s % 50 == 0:
                    trl, _ = net.cross_entropy_loss(net.model_forward(train_set[0]), train_y)
                    vall, _ = net.cross_entropy_loss(net.model_forward(test_set[0]), test_y)
                    tea = net.predict_accuracy(test_set[0], test_y)
                    train_loss.append(trl)
                    val_loss.append(vall)
                    test_acc.append(tea)
        k = len(train_loss)
        xaxis = [i for i in range(k)]
        plt.figure()
        plt.plot(xaxis, train_loss, label='train')
        plt.plot(xaxis, val_loss, label='val')
        plt.legend()
        plt.savefig('Train_Val_Loss.png', dpi=300)
        plt.close()

        plt.figure()
        plt.plot(xaxis, test_acc)
        plt.savefig('Test_Accuracy.png', dpi=300)
        plt.close()
        return net

    def save_model(model_file, path='model.pkl'):
        with open(path, 'wb') as f:
            pickle.dump(model_file, f)


    def read_model(path):
        with open(path, 'rb') as file:
            net = pickle.load(file)
        return net

    # 搜索最佳参数
    best_lr, best_hidden_size, best_beta = params_tuning()

    # 用最佳参数训练并画图，获取最终模型
    net = train_plot(best_lr, best_hidden_size, best_beta)

    # 模型保存和读取
    save_model(net, path='model.pkl')
    init_args = {'hidden_size': best_hidden_size, 'beta': best_beta}
    net = read_model('model.pkl')





