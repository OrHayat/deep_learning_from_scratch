import numpy as np

from neural_network import back_propagation
from neural_network import forward_pass
from neural_network import net
from test_utils import gradient_test
from test_utils import jacobian_test
from test_utils import transpose_test
from utils import jacMV_theta
from utils import jacMV_x
from utils import jacTMV_theta
from utils import jacTMV_x



def network_test():
    sample_size = 5
    sample_count = 1

    x = np.random.randn(sample_size, sample_count)
    w1 = np.random.randn(sample_size, sample_size)
    b = np.random.randn(sample_size, 1)

    dx = np.random.randn(sample_size, sample_count)
    db = np.random.randn(sample_size, sample_count)
    dw1 = np.random.randn(sample_size, sample_size)
    d_theta = np.concatenate((db.flatten('F'), dw1.flatten('F'))).reshape((-1,1))
    jacobian_test(lambda e: net(x + dx * e, w1, b),
                  lambda e: jacMV_x(x, w1, b, e * dx),
                  'Jacobian Test w.r.t x', 'q4j_x')
    transpose_test(lambda v: jacMV_x(x, w1, b, v),
                   lambda v: jacTMV_x(x, w1, b, v),
                   lambda: np.random.randn(dx.shape[0], dx.shape[1]),
                   lambda: np.random.randn(dx.shape[0], dx.shape[1])
                   ,file_name="transpose_test w.r.t x"
                   ,title="transpose_test w.r.t x")

    jacobian_test(lambda e: net(x, w1 + dw1 * e, b + db * e),
                  lambda e: jacMV_theta(x, w1, b, e * d_theta),
                  'Jacobian test w.r.t w', 'q4j_w')
    transpose_test(lambda v: jacMV_theta(x, w1,b, v),
                   lambda v: jacTMV_theta(x, w1,b, v),
                   lambda: np.random.randn(d_theta.shape[0], d_theta.shape[1]),
                   lambda: np.random.randn(dx.shape[0], dx.shape[1]),
                   file_name="transpose_test w.r.t w"
                   ,title="transpose_test w.r.t w")


def layers_test():
    c = np.array([1, 0, 0, 0, 0]).reshape(5, 1)
    labels_count, sample_count = c.shape
    sample_size = 3
    x = np.random.randn(sample_size, sample_count)
    layers_count = 5

    theta_layer_size = sample_size + 2 * (sample_size ** 2)
    loss_layer_size = labels_count * (sample_size + 1)

    theta = np.random.randn(layers_count * theta_layer_size + loss_layer_size, 1)
    d_theta = np.random.randn(layers_count * theta_layer_size + loss_layer_size, 1)

    probabilities, base_loss, xs = forward_pass(x, c, theta, layers_count)
    grad = back_propagation(xs, c, theta, layers_count)

    gradient_test(lambda e: forward_pass(x, c, theta + e * d_theta, layers_count)[1],
                  lambda e: e * np.matmul(d_theta.T, grad).item(),
                  "Net test", "q4net")


if __name__ == "__main__":
    network_test()
    layers_test()
