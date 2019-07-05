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
    denominator = np.sum(numerator)

    return numerator * denominator**(-1)


def loss(x, c, w, b, epsilon=1e-3):
    proba = softmax(x, w, b)
    return -np.sum(c * np.log(proba + epsilon))


def gradient_loss_x(x, c, w, b):
    proba = softmax(x, w, b)
    return np.matmul(w, proba - c)


def gradient_loss_w(x, c, w, b):
    proba = softmax(x, w, b)
    diff = proba.T - c.T
    gw = x * diff
    gb = sum(diff)

    return np.concatenate((gb.flatten('F'), gw.flatten('F')))


def gradient_test(f, g, dim, iterations=20, epsilon=1):
    x = np.random.randn(dim, 1)
    d = np.random.randn(dim, 1)
    delta = []
    for i in range(iterations):
        epsilon *= 0.5
        res = f(x + epsilon * d) - f(x)
        df = np.abs(res)
        dg = np.abs(res - epsilon * np.matmul(d.T, g(x)))
        delta.append((df, dg))
    return delta
