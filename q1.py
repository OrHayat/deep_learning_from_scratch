import numpy as np
from numpy import matlib
import utils as u

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

    d = np.concatenate((db.flatten('F'), dw.flatten('F')))

    x = np.array([1, 2, 3, 4, 5]).reshape((5, 1))
    w = np.array([[1, 1, 1],
                  [2, 2, 2],
                  [3, 3, 3],
                  [4, 4, 4],
                  [5, 5,5]])
    b = np.array([0.5, 0.5, 0.5]).reshape((3, 1))

    iterations = 10

    epsilon = 1
    result_w = []
    for i in range(iterations):
        epsilon *= 0.5
        _f = u.loss(x, c, w + dw * epsilon, b + db * epsilon) - u.loss(x, c, w, b)
        df = np.abs(_f)
        dg = np.abs(_f - epsilon * np.matmul(d.T, u.gradient_loss_w(x, c, w, b)))
        result_w.append((df, dg))

    epsilon = 1
    result_x = []
    # for i in range(iterations):
    #     epsilon *= 0.5
    #     _f = u.loss(x + dx * epsilon, c, w, b) - u.loss(x, c, w, b)
    #     df = np.abs(_f)
    #     dg = np.abs(_f - epsilon * np.matmul(d.T, u.gradient_loss_x(x + dx * epsilon, c, w, b)))
    #     result_x.append((df, dg))

    for r in result_w:
        print(r)
    print(result_x)
