import numpy as np

from utils import linear_model
from utils import load_mnist
from test_utils import plot_sgd_results
import scipy.io as sio


def test_mnist(iterations, learning_rate, batch_size):
    trainX, trainY, testX, testY = load_mnist()

    labels_count = trainY.shape[0]
    sample_size, sample_count = trainX.shape

    freq = 1

    w = np.random.randn(sample_size, labels_count)
    b = np.random.randn(labels_count, 1)
    theta = np.concatenate((b.flatten('F'), w.flatten('F'))) \
        .reshape(b.shape[0] * b.shape[1] + w.shape[0] * w.shape[1], 1)

    model = linear_model(theta, batch_size, learning_rate, iterations, freq)
    model.train(trainX, trainY, testX, testY)
    plot_sgd_results(model, 'SGD results (MNIST)', 'q3_mnist_data', False)

    iterations, train_loss, train_accuracy, test_loss, test_accuracy = zip(*model.training_records)
    return train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[1]


def test_GMMData(iterations, learning_rate, batch_size):
    data = sio.loadmat('resources/GMMData.mat')
    trainX = data['Yt']
    trainY = data['Ct']
    testX = data['Yv']
    testY = data['Cv']

    labels_count = trainY.shape[0]
    sample_size, sample_count = trainX.shape

    freq = 1

    w = np.random.randn(sample_size, labels_count)
    b = np.random.randn(labels_count, 1)
    theta = np.concatenate((b.flatten('F'), w.flatten('F'))) \
        .reshape(b.shape[0] * b.shape[1] + w.shape[0] * w.shape[1], 1)

    model = linear_model(theta, batch_size, learning_rate, iterations, freq)
    model.train(trainX, trainY, testX, testY)

    plot_sgd_results(model, f'SGD results (GMMData)',
                     f'q3_gmm_data_{iterations}_{learning_rate}_{batch_size}', False)

    iterations, train_loss, train_accuracy, test_loss, test_accuracy = zip(*model.training_records)
    return train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[1]


def test_PeaksData(iterations, learning_rate, batch_size):
    data = sio.loadmat('resources/PeaksData.mat')
    trainX = data['Yt']
    trainY = data['Ct']
    testX = data['Yv']
    testY = data['Cv']

    labels_count = trainY.shape[0]
    sample_size, sample_count = trainX.shape

    freq = 1

    w = np.random.randn(sample_size, labels_count)
    b = np.random.randn(labels_count, 1)
    theta = np.concatenate((b.flatten('F'), w.flatten('F'))) \
        .reshape(b.shape[0] * b.shape[1] + w.shape[0] * w.shape[1], 1)

    model = linear_model(theta, batch_size, learning_rate, iterations, freq)
    model.train(trainX, trainY, testX, testY)

    plot_sgd_results(model, 'SGD results (PeaksData)',
                     f'q3_peaks_data_{iterations}_{learning_rate}_{batch_size}', False)

    iterations, train_loss, train_accuracy, test_loss, test_accuracy = zip(*model.training_records)
    return train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[1]



def test_SwissRollData(iterations, learning_rate, batch_size):
    data = sio.loadmat('resources/SwissRollData.mat')
    trainX = data['Yt']
    trainY = data['Ct']
    testX = data['Yv']
    testY = data['Cv']

    labels_count = trainY.shape[0]
    sample_size, sample_count = trainX.shape

    freq = 1

    w = np.random.randn(sample_size, labels_count)
    b = np.random.randn(labels_count, 1)
    theta = np.concatenate((b.flatten('F'), w.flatten('F'))) \
        .reshape(b.shape[0] * b.shape[1] + w.shape[0] * w.shape[1], 1)

    model = linear_model(theta, batch_size, learning_rate, iterations, freq)
    model.train(trainX, trainY, testX, testY)

    plot_sgd_results(model, 'SGD results (SwissRollData)',
                     f'q3_swiss_roll_data_{iterations}_{learning_rate}_{batch_size}', False)

    iterations, train_loss, train_accuracy, test_loss, test_accuracy = zip(*model.training_records)
    return train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[1]


if __name__ == "__main__":
    iterations = [100, 200]
    learning_rates = [0.3, 0.03, 0.003]
    batch_sizes = [50, 100]

    gmm = []
    peaks = []
    swiss = []

    for i in iterations:
        for l in learning_rates:
            for b in batch_sizes:
                gmm.append((i, l, b, test_GMMData(i, l, b)))
                peaks.append((i, l, b, test_PeaksData(i, l, b)))
                swiss.append((i, l, b, test_SwissRollData(i, l, b)))

    with open('q3_gmm_data.txt', 'a') as file:
        file.truncate(0)
        for s in [f'{i} & {l} & {b} & {train_loss} & {train_accuracy} & {test_loss} & {test_accuracy} \\\\\n' for (i, l, b, (train_loss, train_accuracy, test_loss, test_accuracy)) in gmm]:
            file.write(s)

    with open('q3_peaks_data.txt', 'a') as file:
        file.truncate(0)
        for s in [f'{i} & {l} & {b} & {train_loss} & {train_accuracy} & {test_loss} & {test_accuracy} \\\\\n' for (i, l, b, (train_loss, train_accuracy, test_loss, test_accuracy)) in peaks]:
            file.write(s)

    with open('q3_swiss_roll_data.txt', 'a') as file:
        file.truncate(0)
        for s in [f'{i} & {l} & {b} & {train_loss} & {train_accuracy} & {test_loss} & {test_accuracy} \\\\\n' for (i, l, b, (train_loss, train_accuracy, test_loss, test_accuracy)) in swiss]:
            file.write(s)

