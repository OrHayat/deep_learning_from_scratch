import numpy as np
import scipy.io as sio

from utils import stochastic_gradient_descent
from test_utils import plot_sgd_results


def vec2mat(y, labels_count):
    sample_count = y.shape[1]
    m = np.zeros((labels_count, sample_count))
    for i in range(sample_count):
        m[y[0, i], i] = 1
    return m


def load_mnist():
    mnist = sio.loadmat('resources/mnist.mat')
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


if __name__ == "__main__":
    # data = sio.loadmat('resources/PeaksData.mat')
    # trainX = data['Yt']
    # trainY = data['Ct']
    # testX = data['Yv']
    # testY = data['Cv']

    trainX, trainY, testX, testY = load_mnist()

    labels_count = trainY.shape[0]
    sample_size, sample_count = trainX.shape

    iterations = 100
    max_epoch = 100
    freq = 1
    learning_rate = 0.005

    w = np.random.randn(sample_size, labels_count)
    b = np.random.randn(labels_count, 1)
    theta = np.concatenate((b.flatten('F'), w.flatten('F'))) \
        .reshape(b.shape[0] * b.shape[1] + w.shape[0] * w.shape[1], 1)

    iterations, accuracy, loss = zip(
        *stochastic_gradient_descent(trainX, trainY, theta, iterations, learning_rate, max_epoch, testX, testY, freq))

    plot_sgd_results(iterations, loss, accuracy, 'SGD results (MNIST)', 'q3_mnist', True)
