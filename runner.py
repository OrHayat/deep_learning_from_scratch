import numpy as np
import scipy.io as sio

from test_utils import plot_sgd_results
from test_utils import compare_sgd_result
from neural_network import rnn_model


def testGMMData(iterations, learning_rate, batch_size, layers_count):
    print(f'Neural network is running on GMM dataset with {iterations}, {learning_rate}, {batch_size}, {layers_count}')
    data = sio.loadmat('resources/GMMData.mat')
    trainX = data['Yt']
    trainY = data['Ct']
    testX = data['Yv']
    testY = data['Cv']

    labels_count = trainY.shape[0]
    sample_size = trainX.shape[0]

    freq = 1

    theta_layer_size = sample_size + 2 * (sample_size ** 2)
    loss_layer_size = labels_count * (sample_size + 1)

    theta = np.random.randn(layers_count * theta_layer_size + loss_layer_size, 1)

    model = rnn_model(theta, layers_count, batch_size, learning_rate, iterations, freq)
    model.train(trainX, trainY, testX, testY)

    plot_sgd_results(model, 'SGD results (GMM Data)',
                     f'q5_gmm_data_{iterations}_{learning_rate}_{batch_size}_{layers_count}', False)

    model_momentum = rnn_model(theta, layers_count, batch_size, learning_rate, iterations, freq, gamma=0.5)
    model_momentum.train(trainX, trainY, testX, testY)
    compare_sgd_result([model, model_momentum], 'SGD with / without momentum (GMM Data)',
                       f'q5_gmm_data_momentum_{iterations}_{learning_rate}_{batch_size}_{layers_count}', False)

    iterations, train_loss, train_accuracy, test_loss, test_accuracy = zip(*model.training_records)
    return train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1]


def testPeaksData(iterations, learning_rate, batch_size, layers_count):
    print(f'Neural network is running on Peaks dataset with {iterations}, {learning_rate}, {batch_size}, {layers_count}')
    data = sio.loadmat('resources/PeaksData.mat')
    trainX = data['Yt']
    trainY = data['Ct']
    testX = data['Yv']
    testY = data['Cv']

    labels_count = trainY.shape[0]
    sample_size = trainX.shape[0]

    freq = 1

    theta_layer_size = sample_size + 2 * (sample_size ** 2)
    loss_layer_size = labels_count * (sample_size + 1)

    theta = np.random.randn(layers_count * theta_layer_size + loss_layer_size, 1)

    model = rnn_model(theta, layers_count, batch_size, learning_rate, iterations, freq)
    model.train(trainX, trainY, testX, testY)

    plot_sgd_results(model, 'SGD results (Peaks Data)',
                     f'q5_peaks_data_{iterations}_{learning_rate}_{batch_size}_{layers_count}', False)

    model_momentum = rnn_model(theta, layers_count, batch_size, learning_rate, iterations, freq, gamma=0.5)
    model_momentum.train(trainX, trainY, testX, testY)
    compare_sgd_result([model, model_momentum], 'SGD with / without momentum (Peaks Data)',
                       f'q5_peaks_data_momentum_{iterations}_{learning_rate}_{batch_size}_{layers_count}', False)

    iterations, train_loss, train_accuracy, test_loss, test_accuracy = zip(*model.training_records)
    return train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1]


def testSwissRollData(iterations, learning_rate, batch_size, layers_count):
    print(f'Neural network is running on SwissRoll dataset with {iterations}, {learning_rate}, {batch_size}, {layers_count}')
    data = sio.loadmat('resources/SwissRollData.mat')
    trainX = data['Yt']
    trainY = data['Ct']
    testX = data['Yv']
    testY = data['Cv']

    labels_count = trainY.shape[0]
    sample_size = trainX.shape[0]

    freq = 1

    theta_layer_size = sample_size + 2 * (sample_size ** 2)
    loss_layer_size = labels_count * (sample_size + 1)

    theta = np.random.randn(layers_count * theta_layer_size + loss_layer_size, 1)

    model = rnn_model(theta, layers_count, batch_size, learning_rate, iterations, freq)
    model.train(trainX, trainY, testX, testY)

    plot_sgd_results(model, 'SGD results (Swiss Roll Data)',
                     f'q5_swiss_roll_data_{iterations}_{learning_rate}_{batch_size}_{layers_count}', False)

    model_momentum = rnn_model(theta, layers_count, batch_size, learning_rate, iterations, freq, gamma=0.5)
    model_momentum.train(trainX, trainY, testX, testY)
    compare_sgd_result([model, model_momentum], 'SGD with / without momentum (Swiss Roll Data)',
                       f'q5_swiss_roll_data_momentum_{iterations}_{learning_rate}_{batch_size}_{layers_count}', False)

    iterations, train_loss, train_accuracy, test_loss, test_accuracy = zip(*model.training_records)
    return train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1]


if __name__ == "__main__":
    iterations = [100, 200]
    learning_rates = [0.03, 0.003]
    batch_sizes = [50, 100]
    layers = [5, 7, 9]

    gmm = []
    peaks = []
    swiss = []

    for i in iterations:
        for r in learning_rates:
            for b in batch_sizes:
                for l in layers:
                    # gmm.append((i, r, b, l, testGMMData(i, r, b, l)))
                    peaks.append((i, r, b, l, testPeaksData(i, r, b, l)))
                    swiss.append((i, r, b, l, testSwissRollData(i, r, b, l)))

    with open('q5_gmm_data.txt', 'a') as file:
        file.truncate(0)
        for s in [f'{i} & {r} & {b} & {l} & {np.round(train_loss, 4)} & {np.round(train_accuracy, 4)} & {np.round(test_loss, 4)} & {np.round(test_accuracy, 4)} \\\\\n'
                  for (i, r, b, l, (train_loss, train_accuracy, test_loss, test_accuracy)) in gmm]:
            file.write(s)

    with open('q5_peaks_data.txt', 'a') as file:
        file.truncate(0)
        for s in [
            f'{i} & {r} & {b} & {l} & {np.round(train_loss, 4)} & {np.round(train_accuracy, 4)} & {np.round(test_loss, 4)} & {np.round(test_accuracy, 4)} \\\\\n'
                for (i, r, b, l, (train_loss, train_accuracy, test_loss, test_accuracy)) in peaks]:
            file.write(s)

    with open('q5_swiss_roll_data.txt', 'a') as file:
        file.truncate(0)
        for s in [
            f'{i} & {r} & {b} & {l} & {np.round(train_loss, 4)} & {np.round(train_accuracy, 4)} & {np.round(test_loss, 4)} & {np.round(test_accuracy, 4)} \\\\\n'
                for (i, r, b, l, (train_loss, train_accuracy, test_loss, test_accuracy)) in swiss]:
            file.write(s)
