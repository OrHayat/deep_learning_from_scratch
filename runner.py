import numpy as np
import scipy.io as sio

from test_utils import plot_sgd_results
from utils import load_mnist
from neural_network import rnn_model


def testGMMData():
    print('Neural network is running on GMM dataset')
    data = sio.loadmat('resources/GMMData.mat')
    trainX = data['Yt']
    trainY = data['Ct']
    testX = data['Yv']
    testY = data['Cv']

    labels_count = trainY.shape[0]
    sample_size = trainX.shape[0]

    layers_count = 5
    batch_size = 100
    learning_rate = 0.03
    iterations = 100
    freq = 1

    theta_layer_size = sample_size + 2 * (sample_size ** 2)
    loss_layer_size = labels_count * (sample_size + 1)

    theta = np.random.randn(layers_count * theta_layer_size + loss_layer_size, 1)

    model = rnn_model(theta, layers_count, batch_size, learning_rate, iterations, freq)
    model.train(trainX, trainY, testX, testY)
    iterations, train_loss, train_accuracy, test_loss, test_accuracy = zip(*model.training_records)
    plot_sgd_results(iterations, train_loss, train_accuracy, test_loss, test_accuracy, 'SGD results (GMM Data)',
                     'q5_gmm_data', False)


def testPeaksData():
    print('Neural network is running on Peaks dataset')
    data = sio.loadmat('resources/PeaksData.mat')
    trainX = data['Yt']
    trainY = data['Ct']
    testX = data['Yv']
    testY = data['Cv']

    labels_count = trainY.shape[0]
    sample_size = trainX.shape[0]

    layers_count = 7
    batch_size = 100
    learning_rate = 0.03
    iterations = 200
    freq = 1

    theta_layer_size = sample_size + 2 * (sample_size ** 2)
    loss_layer_size = labels_count * (sample_size + 1)

    theta = np.random.randn(layers_count * theta_layer_size + loss_layer_size, 1)

    model = rnn_model(theta, layers_count, batch_size, learning_rate, iterations, freq)
    model.train(trainX, trainY, testX, testY)
    iterations, train_loss, train_accuracy, test_loss, test_accuracy = zip(*model.training_records)
    plot_sgd_results(iterations, train_loss, train_accuracy, test_loss, test_accuracy, 'SGD results (Peaks Data)',
                     'q5_peaks_data', False)


def testSwissRollData():
    print('Neural network is running on SwissRoll dataset')
    data = sio.loadmat('resources/SwissRollData.mat')
    trainX = data['Yt']
    trainY = data['Ct']
    testX = data['Yv']
    testY = data['Cv']

    labels_count = trainY.shape[0]
    sample_size = trainX.shape[0]

    layers_count = 9
    batch_size = 100
    learning_rate = 0.03
    iterations = 300
    freq = 1

    theta_layer_size = sample_size + 2 * (sample_size ** 2)
    loss_layer_size = labels_count * (sample_size + 1)

    theta = np.random.randn(layers_count * theta_layer_size + loss_layer_size, 1)

    model = rnn_model(theta, layers_count, batch_size, learning_rate, iterations, freq, moment=0.7)
    model.train(trainX, trainY, testX, testY)
    iterations, train_loss, train_accuracy, test_loss, test_accuracy = zip(*model.training_records)
    plot_sgd_results(iterations, train_loss, train_accuracy, test_loss, test_accuracy, 'SGD results (Swiss Roll Data)',
                     'q5_swiss_roll_data', False)


if __name__ == "__main__":
    # testGMMData()
    # testPeaksData()
    testSwissRollData()
