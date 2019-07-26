import numpy as np
import scipy.io as sio

from test_utils import plot_sgd_results
from test_utils import compare_sgd_result
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

    plot_sgd_results(model, 'SGD results (GMM Data)', f'q5_gmm_data', False)

    model_momentum = rnn_model(theta, layers_count, batch_size, learning_rate, iterations, freq, gamma=0.5)
    model_momentum.train(trainX, trainY, testX, testY)
    compare_sgd_result([model, model_momentum], 'SGD with / without momentum (GMM Data)',
                       'q5_gmm_data_momentum', False)


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

    plot_sgd_results(model, 'SGD results (Peaks Data)',
                     f'q5_peaks_data', False)

    model_momentum = rnn_model(theta, layers_count, batch_size, learning_rate, iterations, freq, gamma=0.5)
    model_momentum.train(trainX, trainY, testX, testY)
    compare_sgd_result([model, model_momentum], 'SGD with / without momentum (Peaks Data)',
                       'q5_peaks_data_momentum', False)


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

    model = rnn_model(theta, layers_count, batch_size, learning_rate, iterations, freq)
    model.train(trainX, trainY, testX, testY)

    plot_sgd_results(model, 'SGD results (Swiss Roll Data)',
                     f'q5_swiss_roll_data', False)

    model_momentum = rnn_model(theta, layers_count, batch_size, learning_rate, iterations, freq, gamma=0.5)
    model_momentum.train(trainX, trainY, testX, testY)
    compare_sgd_result([model, model_momentum], 'SGD with / without momentum (Swiss Roll Data)', 'q5_swiss_roll_data_momentum', False)


if __name__ == "__main__":
    testGMMData()
    testPeaksData()
    testSwissRollData()
