import numpy as np
import numpy.matlib


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x, w, b):
    samples_count = x.shape[1]
    classes_count = w.shape[1]
    log_numerator = np.matmul(w.T, x) + np.matlib.repmat(b, 1, samples_count)
    log_numerator -= np.matlib.repmat(np.max(log_numerator), classes_count, 1)

    numerator = np.exp(log_numerator)
    denominator = np.sum(numerator, axis=0)

    return numerator * denominator**(-1)


def loss(x, c, w, b, epsilon=1e-8):
    proba = softmax(x, w, b)
    return -np.sum(c * np.log(proba + epsilon))


def gradient_loss_x(x, c, w, b):
    proba = softmax(x, w, b)
    return np.matmul(w, proba - c)


def gradient_loss_w(x, c, w, b):
    proba = softmax(x, w, b)
    diff = proba.T - c.T
    gw = x * diff
    gb = np.sum(diff, axis=0)
    return np.concatenate((gb.flatten('F'), gw.flatten('F')))

def stochastic_gradient_descent(x, c, w, batch_size, learning_rate, yv, cv, iterations, freq):
    sample_count = x.shape[0]
    sample_size = x.shape[1]
    weight_size = w.shape[1]
    labels_count = c.shape[0]
    batch_count = sample_count / batch_size

    records = np.zeros(np.ceil(iterations / freq), 2)

    w_k = w
    for i in range(iterations):
        idxs = np.random.permutation(sample_count)
        for j in range(batch_count):
            idx_k = idxs[(j-1) * batch_size + 1 : j * batch_size]
            x_k = x[:, idx_k]
            c_k = x[:, idx_k]
            b_k = w_k[1 : labels_count]
            w_k = np.reshape(w_k[labels_count+1 : weight_size], [sample_size, labels_count])
            g_k = gradient_loss_w(x_k, c_k, w_k, b_k)