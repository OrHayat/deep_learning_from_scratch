import numpy as np
import scipy.io as sio

from test_utils import plot_sgd_results
from test_utils import compare_sgd_result
from neural_network import nn_model


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

    theta_layer_size = sample_size + 1 * (sample_size ** 2)
    loss_layer_size = labels_count * (sample_size + 1)

    theta = np.random.randn(layers_count * theta_layer_size + loss_layer_size, 1)

    model = nn_model(theta, layers_count, batch_size, learning_rate, iterations, freq)
    model.train(trainX, trainY, testX, testY)

    plot_sgd_results(model, 'SGD results (GMM Data)',
                     f'q5_gmm_data_{iterations}_{learning_rate}_{batch_size}_{layers_count}', False)

    model_momentum = nn_model(theta, layers_count, batch_size, learning_rate, iterations, freq, gamma=0.5)
    model_momentum.train(trainX, trainY, testX, testY)
    compare_sgd_result([model, model_momentum], 'SGD with / without momentum (GMM Data)',
                       f'q5_gmm_data_momentum_{iterations}_{learning_rate}_{batch_size}_{layers_count}', False)

    iterations, train_loss, train_accuracy, test_loss, test_accuracy = zip(*model.training_records)
    return train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1]


def testPeaksData(iterations, learning_rate, batch_size, layers_count):
    print(
        f'Neural network is running on Peaks dataset with {iterations}, {learning_rate}, {batch_size}, {layers_count}')
    data = sio.loadmat('resources/PeaksData.mat')
    trainX = data['Yt']
    trainY = data['Ct']
    testX = data['Yv']
    testY = data['Cv']

    labels_count = trainY.shape[0]
    sample_size = trainX.shape[0]

    freq = 1

    theta_layer_size = sample_size + 1 * (sample_size ** 2)
    loss_layer_size = labels_count * (sample_size + 1)

    theta = np.random.randn(layers_count * theta_layer_size + loss_layer_size, 1)

    model = nn_model(theta, layers_count, batch_size, learning_rate, iterations, freq)
    model.train(trainX, trainY, testX, testY)

    plot_sgd_results(model, 'SGD results (Peaks Data)',
                     f'q5_peaks_data_{iterations}_{learning_rate}_{batch_size}_{layers_count}', False)

    model_momentum = nn_model(theta, layers_count, batch_size, learning_rate, iterations, freq, gamma=0.5)
    model_momentum.train(trainX, trainY, testX, testY)
    compare_sgd_result([model, model_momentum], 'SGD with / without momentum (Peaks Data)',
                       f'q5_peaks_data_momentum_{iterations}_{learning_rate}_{batch_size}_{layers_count}', False)

    iterations, train_loss, train_accuracy, test_loss, test_accuracy = zip(*model.training_records)
    return train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1]


def testkeras():
    data = sio.loadmat('resources/SwissRollData.mat')
    trainX = data['Yt'].T
    trainY = data['Ct'].T
    testX = data['Yv'].T
    testY = data['Cv'].T
    # first neural network with keras tutorial
    print(trainX.shape, 'trX')  # x,y
    print(trainY.shape, 'try')
    print(testX.shape, 'teX')  # x,y
    print(testY.shape, 'tey')
    print(trainY)
    from numpy import loadtxt
    from keras.models import Sequential
    from keras.layers import Dense, Input
    # load the dataset
    # define the keras model
    print(trainX.shape)
    model = Sequential()
    model.add(Dense(12, input_shape=(2,), activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    # compile the keras model
    print("compiling")
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    print("compiled succesfully")
    # fit the keras model on the dataset
    print(trainX.shape)
    model.fit(trainX, trainY, epochs=150, batch_size=50)
    # evaluate the keras model
    loss, accuracy = model.evaluate(testX, testY)
    print('Accuracy: %.2f', (accuracy * 100), 'loss', loss)
    # exit()


def testSwissRollData(iterations, learning_rate, batch_size, layers_count):
    print(
        f'Neural network is running on SwissRoll dataset with {iterations}, {learning_rate}, {batch_size}, {layers_count}')
    data = sio.loadmat('resources/SwissRollData.mat')
    trainX = data['Yt']
    trainY = data['Ct']
    testX = data['Yv']
    testY = data['Cv']
    labels_count = trainY.shape[0]
    sample_size = trainX.shape[0]

    freq = 1

    theta_layer_size = sample_size + 1 * (sample_size ** 2)
    loss_layer_size = labels_count * (sample_size + 1)

    theta = np.random.randn(layers_count * theta_layer_size + loss_layer_size, 1)

    model = nn_model(theta, layers_count, batch_size, learning_rate, iterations, freq)
    model.train(trainX, trainY, testX, testY)

    plot_sgd_results(model, 'SGD results (Swiss Roll Data)',
                     f'q5_swiss_roll_data_{iterations}_{learning_rate}_{batch_size}_{layers_count}', False)

    model_momentum = nn_model(theta, layers_count, batch_size, learning_rate, iterations, freq, gamma=0.5)
    model_momentum.train(trainX, trainY, testX, testY)
    compare_sgd_result([model, model_momentum], 'SGD with / without momentum (Swiss Roll Data)',
                       f'q5_swiss_roll_data_momentum_{iterations}_{learning_rate}_{batch_size}_{layers_count}', False)

    iterations, train_loss, train_accuracy, test_loss, test_accuracy = zip(*model.training_records)
    return train_loss[-1], train_accuracy[-1], test_loss[-1], test_accuracy[-1]


if __name__ == "__main__":
    # testkeras()
    iterations = [100, 200]
    learning_rates = [0.01, 0.001]
    batch_sizes = [50, 100, 500]
    layers = [2, 3, 5, 7]

    gmm = None
    peaks = None
    swiss = None

    for i in iterations:
        for r in learning_rates:
            for b in batch_sizes:
                for l in layers:
                    res = [testGMMData(i, r, b, l)]
                    data = (i, r, b, l)
                    res.insert(0, data)
                    if gmm is None or res[-1][-1] > gmm[-1][-1]:
                        gmm = res
                    res = [testPeaksData(i, r, b, l)]
                    res.insert(0, data)
                    if peaks is None or res[-1][-1] > peaks[-1][-1]:
                        peaks = res
                    res = [testSwissRollData(i, r, b, l)]
                    res.insert(0, data)
                    if swiss is None or res[-1][-1] > swiss[-1][-1]:
                        swiss = res
    s1 = (
        "highest testing accuracy for gmm happent at num_epoch={},learning_rate={},batch_size={} layer count={}\nresults were training loss={},training acc={},testing loss={},testing acc={}".format(
            gmm[0][0], gmm[0][1], gmm[0][2], gmm[0][3], gmm[1][0], gmm[1][1], gmm[1][2], gmm[1][3]))
    s2 = (
        "highest testing accuracy for peaks happent at num_epoch={},learning_rate={},batch_size={} layer count={}\nresults were training loss={},training acc={},testing loss={},testing acc={}".format(
            peaks[0][0], peaks[0][1], peaks[0][2], peaks[0][3], peaks[1][0], peaks[1][1], peaks[1][2], peaks[1][3]))
    s3 = (
        "highest testing accuracy for swiss happent at num_epoch={},learning_rate={},batch_size={} layer count={}\nresults were training loss={},training acc={},testing loss={},testing acc={}".format(
            swiss[0][0], swiss[0][1], swiss[0][2], swiss[0][3], swiss[1][0], swiss[1][1], swiss[1][2], swiss[1][3]))
    print(s1)
    print(s2)
    print(s3)
    with open("report_netwrok.txt", "w") as f:
        f.write(s1)
        f.write("\n")
        f.write(s2)
        f.write("\n")
        f.write(s3)
        f.write("\n")
