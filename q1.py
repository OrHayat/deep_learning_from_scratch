import numpy as np

from test_utils import gradient_test
from utils import gradient_loss_theta
from utils import gradient_loss_x
from utils import loss

if __name__ == "__main__":
    sample_size = 5
    c = np.array([[1], [0], [0]])
    labels_count = c.shape[0]
    sample_count = c.shape[1]

    x = np.random.randn(sample_size, sample_count)
    w = np.random.randn(sample_size, labels_count)
    b = np.random.randn(labels_count, 1)

    dx = np.random.randn(sample_size, sample_count)
    dw = np.random.randn(sample_size, labels_count)
    db = np.random.randn(labels_count, 1)
    dtheta = np.concatenate((db.flatten('F'), dw.flatten('F'))) \
        .reshape(db.shape[0] * db.shape[1] + dw.shape[0] * dw.shape[1], 1)

    gradient_test(lambda e: loss(x + dx * e, c, w, b),
                  lambda e: e * np.matmul(dx.T, gradient_loss_x(x, c, w, b)).item(),
                  'Softmax gradient test w.r.t to x', 'q1_x')
    gradient_test(lambda e: loss(x, c, w + dw * e, b + db * e),
                  lambda e: e * np.matmul(dtheta.T, gradient_loss_theta(x, c, w, b)).item(),
                  'Softmax gradient test w.r.t to w', 'q1_w')
