import numpy as np
from utils import loss
from utils import softmax
from utils import gradient_loss_theta
from utils import gradient_loss_x
from utils import sigmoid
from utils import jacTMV_x
from utils import jacTMV_theta


def rnn(x, w1, w2, b):
    sample_count = x.shape[1]
    inner = np.matmul(w1, x) + np.tile(b, (1, sample_count))
    return x + np.matmul(w2, sigmoid(inner))


def get_rnn_weights(theta, k, sample_size):
    layer_size = sample_size + 2 * (sample_size ** 2)
    layer_weights = theta[k * layer_size: (k + 1) * layer_size]
    b = layer_weights[0: sample_size]
    w1 = layer_weights[sample_size: sample_size + sample_size ** 2].reshape(sample_size, sample_size).T
    w2 = layer_weights[sample_size + sample_size ** 2: layer_size].reshape(sample_size, sample_size).T

    return w1, w2, b


def forward_pass(x, c, theta, layers_count):
    sample_size, sample_count = x.shape
    labels_count = c.shape[0]

    xis = np.zeros((sample_size, sample_count, layers_count + 1))
    xis[:, :, 0] = x
    for k in range(1, layers_count + 1):
        w1, w2, b = get_rnn_weights(theta, k - 1, sample_size)
        xis[:, :, k] = rnn(xis[:, :, k - 1], w1, w2, b)

    theta_layer_size = sample_size + 2 * (sample_size ** 2)
    loss_weights_idx = layers_count * theta_layer_size
    b = theta[loss_weights_idx: loss_weights_idx + labels_count]
    w = theta[loss_weights_idx + labels_count: loss_weights_idx + labels_count + sample_size * labels_count] \
        .reshape(labels_count, sample_size).T

    loss_val = loss(xis[:, :, layers_count], c, w, b)
    probabilities = softmax(xis[:, :, layers_count], w, b)
    return probabilities, loss_val, xis


def get_loss_weights(theta, sample_size, labels_count, layers_count):
    theta_layer_size = sample_size + 2 * (sample_size ** 2)
    loss_weights_idx = layers_count * theta_layer_size

    b = theta[loss_weights_idx: loss_weights_idx + labels_count]
    w = theta[loss_weights_idx + labels_count: loss_weights_idx + (sample_size + 1) * labels_count]
    w = w.reshape((labels_count, sample_size)).T

    return w, b


def back_propagation(xis, c, theta, layers_count):
    sample_size = xis.shape[0]
    labels_count = c.shape[0]

    theta_layer_size = sample_size + 2 * (sample_size ** 2)
    loss_weights_idx = layers_count * theta_layer_size
    weights_size = loss_weights_idx + labels_count + sample_size * labels_count
    w_loss, b_loss = get_loss_weights(theta, sample_size, labels_count, layers_count)

    g_theta = np.zeros(theta.shape)
    g_w_loss = gradient_loss_theta(xis[:, :, layers_count], c, w_loss, b_loss)
    g_theta[loss_weights_idx: weights_size] = g_w_loss

    g_x = gradient_loss_x(xis[:, :, layers_count], c, w_loss, b_loss)

    for k in range(layers_count - 1, -1, -1):
        w1, w2, b = get_rnn_weights(theta, k, sample_size)

        g_w_layer = jacTMV_theta(xis[:, :, k], w1, w2, b, g_x)
        g_theta[k * theta_layer_size: (k + 1) * theta_layer_size] = g_w_layer
        g_x = jacTMV_x(xis[:, :, k], w1, w2, b, g_x)
    return g_theta


class rnn_model:
    def __init__(self, theta, layers_count, batch_size, learning_rate, iterations, freq):
        self.theta = theta
        self.theta_k = self.theta
        self.layers_count = layers_count
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.freq = freq
        self.training_records = []

    def train(self, x, c, test_x, test_c):
        sample_count = x.shape[1]
        batch_count = int(sample_count / self.batch_size)
        self.theta_k = self.theta
        # idxs_lst = np.genfromtxt('idxs.csv', delimiter=',').astype('int') - 1
        for i in range(self.iterations):
            idxs = np.random.permutation(sample_count)
            # idxs = idxs_lst[i]
            for j in range(0, batch_count):
                idx_k = idxs[j * self.batch_size: (j + 1) * self.batch_size]
                x_k = x[:, idx_k]
                c_k = c[:, idx_k]

                _, _, xis = forward_pass(x_k, c_k, self.theta_k, self.layers_count)
                g_k = back_propagation(xis, c_k, self.theta_k, self.layers_count)

                a_k = self.learning_rate if i <= 100 else (5 * self.learning_rate) / np.sqrt(i)

                self.theta_k = self.theta_k - a_k * g_k

            if np.mod(i, self.freq) == 0:
                train_loss, train_accuracy = self.evaluate(x, c)
                test_loss, test_accuracy = self.evaluate(test_x, test_c)

                print(i, train_loss, train_accuracy, test_loss, test_accuracy)
                self.training_records.append((i, train_loss, train_accuracy, test_loss, test_accuracy))

    def evaluate(self, x, c):
        probabilities, loss_val, _ = forward_pass(x, c, self.theta_k, self.layers_count)

        test_count = probabilities.shape[1]
        top_scores = [np.argmax(probabilities[:, i]) for i in range(test_count)]
        actual_scores = [np.argmax(c[:, i]) for i in range(test_count)]
        accuracy = np.sum([1 if top_scores[i] == actual_scores[i] else 0
                           for i in range(test_count)]) / test_count
        return loss_val, accuracy

    def predict(self, x, c):
        probabilities, loss_val, _ = forward_pass(x, c, self.theta_k, self.layers_count)
        return probabilities, loss_val
