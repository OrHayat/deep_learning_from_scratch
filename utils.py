import numpy as np
import numpy.linalg as la
import scipy.io as sio


def vec2mat(y, labels_count):
    sample_count = y.shape[1]
    m = np.zeros((labels_count, sample_count))
    for i in range(sample_count):
        m[y[0, i], i] = 1
    return m


def load_mnist():
    mnist = sio.loadmat('resources/MNISTData.mat')
    trainX = mnist['trainX']
    trainY = mnist['trainY']
    testX = mnist['testX']
    testY = mnist['testY']

    labels_count = 10
    trainX = trainX.T / 255
    trainY = vec2mat(trainY, labels_count)
    testX = testX.T / 255
    testY = vec2mat(testY, labels_count)
    return trainX, trainY, testX, testY


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x,0)


def softmax(x, w, b):
    sample_count = x.shape[1]
    classes_count = w.shape[1]
    log_numerator = np.matmul(w.T, x) + np.tile(b, (1, sample_count))
    log_numerator -= np.tile(np.max(log_numerator), (classes_count, 1))

    numerator = np.exp(log_numerator)
    denominator = np.sum(numerator, axis=0)

    return numerator * denominator ** (-1)


def loss(x, c, w, b, epsilon=1e-8, regularization=0):
    probabilities = softmax(x, w, b)
    return -(np.sum(c * np.log(probabilities + epsilon)) + regularization * np.power(la.norm(w), 2)) / x.shape[1]


def gradient_loss_x(x, c, w, b):
    probabilities = softmax(x, w, b)
    return np.matmul(w, probabilities - c)


def gradient_loss_theta(x, c, w, b):
    probabilities = softmax(x, w, b)
    diff = probabilities.T - c.T
    gw = np.matmul(x, diff)
    gb = np.sum(diff, axis=0).reshape(diff.shape[1], 1)
    return np.concatenate((gb.flatten('F'), gw.flatten('F'))).reshape((-1, 1))
    # \.reshape(gb.shape[0] * gb.shape[1] + gw.shape[0] * gw.shape[1], 1)


class linear_model:
    def __init__(self, theta, batch_size, learning_rate, iterations, freq):
        self.theta = theta
        self.theta_k = self.theta
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.freq = freq
        self.training_records = []

    def train(self, x, c, test_x, test_y):
        sample_size, sample_count = x.shape
        theta_size = self.theta.shape[0]
        labels_count = c.shape[0]
        batch_count = int(sample_count / self.batch_size)
        for i in range(self.iterations):
            idxs = np.random.permutation(sample_count)
            for j in range(0, batch_count):
                idx_k = idxs[j * self.batch_size: (j + 1) * self.batch_size]
                x_k = x[:, idx_k]
                c_k = c[:, idx_k]
                b_k = self.theta_k[0: labels_count]
                w_k = self.theta_k[labels_count: theta_size].reshape(labels_count, sample_size).T
                g_k = gradient_loss_theta(x_k, c_k, w_k, b_k)
                a_k = self.learning_rate if i <= 100 else (10 * self.learning_rate) / np.sqrt(i)
                self.theta_k = self.theta_k - a_k * g_k
            if np.mod(i, self.freq) == 0:
                train_loss, train_accuracy = self.evaluate(x, c, sample_size, labels_count, theta_size)
                test_loss, test_accuracy = self.evaluate(test_x, test_y, sample_size, labels_count, theta_size)
                print(i, train_loss, train_accuracy, test_loss, test_accuracy)
                self.training_records.append((i, train_loss, train_accuracy, test_loss, test_accuracy))

    def evaluate(self, x, c, sample_size, labels_count, theta_size):
        b_k = self.theta_k[0: labels_count]
        w_k = self.theta_k[labels_count: theta_size].reshape(labels_count, sample_size).T

        loss_val = loss(x, c, w_k, b_k)
        probabilities = softmax(x, w_k, b_k)

        test_count = probabilities.shape[1]
        top_scores = [np.argmax(probabilities[:, i]) for i in range(test_count)]
        actual_scores = [np.argmax(c[:, i]) for i in range(test_count)]
        accuracy = np.sum([1 if top_scores[i] == actual_scores[i] else 0
                           for i in range(test_count)]) / test_count
        return loss_val, accuracy


def jacMV_x(x, w1, b, v, type='sigmoid'):
    n = w1.shape[0]
    if type == 'sigmoid':
        inner = sigmoid(np.matmul(w1, x) + b)
        derivative = np.multiply(inner, (1 - inner))
    if type == 'relu':
        inner = relu(np.matmul(w1, x) + b)
        inner[inner > 0] = 1
        derivative = inner
    jac = np.matmul(np.diagflat(derivative), w1)
    return np.matmul(jac, v)


def jacTMV_x(x, w1, b, v, type='sigmoid'):
    sample_count = x.shape[1]
    if type == 'sigmoid':
        inner = sigmoid(np.matmul(w1, x) + np.tile(b, (1, sample_count)))
        derivative = np.multiply(inner, (1 - inner))
    if type == 'relu':
        inner = relu(np.matmul(w1, x) + np.tile(b, (1, sample_count)))
        inner[inner > 0] = 1
        derivative = inner
    return np.matmul(w1.T, np.multiply(derivative, v))


def jacMV_theta(x, w1, b, v, type='sigmoid'):
    n = w1.shape[0]
    m = x.shape[0]

    db = v[0: n]
    dw1 = v[n: n + n ** 2]
    if type == 'sigmoid':
        inner = sigmoid(np.matmul(w1, x) + b)
        derivative = np.multiply(inner, 1 - inner)
    if type == 'relu':
        inner = relu(np.matmul(w1, x) + b)
        inner[inner > 0] = 1
        derivative = inner
    diagonal = np.diagflat(derivative)

    jbv = np.matmul(diagonal, db)

    jw1v = np.matmul(diagonal, np.matmul(np.kron(x.T, np.eye(m)), dw1))

    return jbv + jw1v


def jacTMV_theta(x, w1, b, v, type='sigmoid'):
    n, m = x.shape
    if type == 'sigmoid':
        inner = sigmoid(np.matmul(w1, x) + np.tile(b, (1, m)))
        derivative = np.multiply(inner, 1 - inner)
    if type == 'relu':
        inner = relu(np.matmul(w1, x) + np.tile(b, (1, m)))
        inner[inner > 0] = 1
        derivative = inner
    x_rep = np.tile(x.T, (n, 1)).reshape(m, n ** 2, order='F')
    jbTv = derivative * v  # np.multiply(derivative, np.matmul(w2.T, v))
    jbTv_avg = np.average(jbTv, 1).reshape(-1, 1)
    jbTv_rep = np.tile(jbTv, (n, 1))
    jw1Tv = np.multiply(x_rep.T, jbTv_rep)
    jw1Tv_avg = np.average(jw1Tv, 1).reshape(jw1Tv.shape[0], 1)
    return np.concatenate((jbTv_avg.flatten('F'), jw1Tv_avg.flatten('F'),)).reshape((-1, 1))
    # .reshape(jbTv_avg.shape[0] * jbTv_avg.shape[1] +
    #          jw1Tv_avg.shape[0] * jw1Tv_avg.shape[1] +
    #          jw2Tv_avg.shape[0] * jw2Tv_avg.shape[1], 1)
# def jacTMV_theta(x, w1, b, v):
#     n, m = x.shape
#     inner = sigmoid(np.matmul(w1, x) + np.tile(b, (1, m)))
#     derivative = np.multiply(inner, 1 - inner)
#     jwTv=np.matmul(derivative*v,x.T)
#     inner=sigmoid(np.matmul(w1,x)+b)
#     derivative = np.multiply(inner, 1 - inner)
#     jbTv =derivative*v
#     return np.concatenate((jbTv.flatten('F'),jwTv.flatten('F'))).reshape((-1,1))
