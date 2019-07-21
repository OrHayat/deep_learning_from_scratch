import numpy as np

from utils import linear_model
from utils import load_mnist
from test_utils import plot_sgd_results
import scipy.io as sio


def test_mnist():
    trainX, trainY, testX, testY = load_mnist()

    labels_count = trainY.shape[0]
    sample_size, sample_count = trainX.shape

    iterations = 100
    batch_size = 100
    freq = 1
    learning_rate = 0.03

    w = np.random.randn(sample_size, labels_count)
    b = np.random.randn(labels_count, 1)
    theta = np.concatenate((b.flatten('F'), w.flatten('F'))) \
        .reshape(b.shape[0] * b.shape[1] + w.shape[0] * w.shape[1], 1)

    model = linear_model(theta, batch_size, learning_rate, iterations, freq)
    model.train(trainX, trainY, testX, testY)
    iterations, train_loss, train_accuracy, test_loss, test_accuracy = zip(*model.training_records)
    plot_sgd_results(iterations, train_loss, train_accuracy, test_loss, test_accuracy,
                     'SGD results (MNIST)', 'q3_mnist_data', False)


def test_PeaksData():
    data = sio.loadmat('resources/PeaksData.mat')
    trainX = data['Yt']
    trainY = data['Ct']
    testX = data['Yv']
    testY = data['Cv']

    labels_count = trainY.shape[0]
    sample_size, sample_count = trainX.shape

    iterations = 100
    batch_size = 100
    freq = 1
    learning_rate = 0.03

    w = np.random.randn(sample_size, labels_count)
    b = np.random.randn(labels_count, 1)
    theta = np.concatenate((b.flatten('F'), w.flatten('F'))) \
        .reshape(b.shape[0] * b.shape[1] + w.shape[0] * w.shape[1], 1)

    model = linear_model(theta, batch_size, learning_rate, iterations, freq)
    model.train(trainX, trainY, testX, testY)
    iterations, train_loss, train_accuracy, test_loss, test_accuracy = zip(*model.training_records)
    plot_sgd_results(iterations, train_loss, train_accuracy, test_loss, test_accuracy,
                     'SGD results (PeaksData)', 'q3_peaks_data', False)


if __name__ == "__main__":
    test_mnist()
    test_PeaksData()
