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
