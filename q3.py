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
    iterations = [100,200]
    learning_rates = [0.1, 0.01, 0.001]
    batch_sizes = [50, 100]

    gmm = None
    peaks = None
    swiss = None

    for i in iterations:
        for l in learning_rates:
            for b in batch_sizes:
                res=[test_GMMData(i,l,b)]
                data=(i,l,b)
                res.insert(0,data)
                if gmm is None or res[-1][-1]>gmm[-1][-1]:
                    gmm=res
                res=[test_PeaksData(i,l,b)]
                res.insert(0,data)
                if peaks is None or res[-1][-1]>peaks[-1][-1]:
                    peaks=res
                res=[test_SwissRollData(i,l,b)]
                res.insert(0,data)
                if swiss is None or res[-1][-1]>swiss[-1][-1]:
                    swiss=res
    s1=("highest testing accuracy for gmm happent at num_epoch={},learning_rate={},batch_size={}\nresults were training loss={},training acc={},testing loss={},testing acc={}".format(gmm[0][0],gmm[0][1],gmm[0][2],gmm[1][0],gmm[1][1],gmm[1][2],gmm[1][3]))
    s2=("highest testing accuracy for peaks happent at num_epoch={},learning_rate={},batch_size={}\nresults were training loss={},training acc={},testing loss={},testing acc={}".format(peaks[0][0],peaks[0][1],peaks[0][2],peaks[1][0],peaks[1][1],peaks[1][2],peaks[1][3]))
    s3=("highest testing accuracy for swiss happent at num_epoch={},learning_rate={},batch_size={}\nresults were training loss={},training acc={},testing loss={},testing acc={}".format(swiss[0][0],swiss[0][1],swiss[0][2],swiss[1][0],swiss[1][1],swiss[1][2],swiss[1][3]))
    print(s1)
    print(s2)
    print(s3)
    with open("report.txt","w") as f:
        f.write(s1)
        f.write("\n")
        f.write(s2)
        f.write("\n")
        f.write(s3)
        f.write("\n")
