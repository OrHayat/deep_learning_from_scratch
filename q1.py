import numpy as np
from numpy import matlib
from utils import loss
from utils import gradient_loss_w
from utils import gradient_loss_x
import matplotlib.pyplot as plt

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
    dwb = np.concatenate((db.flatten('F'), dw.flatten('F'))).reshape(18, 1)

    # x = np.array([-2.5, -0.15, 0.35, 0.7, -1.3]).reshape(5, 1)
    # w = np.array([[-1, -1.6, -1.3], [0.8, 0.75, -1.5], [-0.1, 1.2, -0.04], [0.5, 1.6, -0.6], [-0.96, -1.5, 1.3]])
    # b = np.array([1.6, -2, -0.5]).reshape(3, 1)
    # dx = np.array([0.2, -0.6, 0.5, 0.01, -0.04]).reshape(5, 1)
    # dw = np.array([[0.2, 0.3, -0.7], [-0.8, 0.4, 0.3], [-1.3, -0.9, 2.4], [0.6, -0.5, -0.5], [0.6, -0.1, 0.7]])
    # db = np.array([-1, 1.4, -1]).reshape(3, 1)
    # dwb = np.concatenate((db.flatten('F'), dw.flatten('F'))).reshape(18, 1)

    iterations = 15
    epsilons = [0.5 ** e for e in range(6, iterations)]

    a1 = gradient_loss_w(x, c, w, b)
    a2 = gradient_loss_x(x, c, w, b)
    a3 = np.matmul(dwb.T, a1)
    a4 = np.matmul(dx.T, a2)

    df_w = [np.abs(loss(x, c, w + dw * e, b + db * e) - loss(x, c, w, b))
            for e in epsilons]
    dg_w = [np.abs(loss(x, c, w + dw * e, b + db * e) - loss(x, c, w, b)
                   - e * np.matmul(dwb.T, gradient_loss_w(x, c, w, b)))
            for e in epsilons]

    df_x = [np.abs(loss(x + dx * e, c, w, b) - loss(x, c, w, b))
            for e in epsilons]
    dg_x = [np.abs(loss(x + dx * e, c, w, b) - loss(x, c, w, b)
                   - e * np.matmul(dx.T, gradient_loss_x(x, c, w, b))[0][0])
            for e in epsilons]

    plt.loglog(epsilons, df_x)
    plt.loglog(epsilons, dg_x)
    plt.title = 'Gradient test - x'
    plt.xlabel = 'Iterations'
    plt.ylabel = 'Value'
    plt.legend([r'$\left|f\left(x + \epsilon d\right) - f\left(x\right)\right|$',
                r'$\left|f\left(x + \epsilon d\right) - f\left(x\right) - \epsilon d^T g\left(x\right)\right|$'])
    plt.savefig('submission/q1x.png')
    plt.show()

    plt.loglog(epsilons, df_w)
    plt.loglog(epsilons, dg_w)
    plt.title = 'Gradient test - w, b'
    plt.xlabel = 'Iterations'
    plt.ylabel = 'Value'
    plt.legend([r'$\left|f\left(x + \epsilon d\right) - f\left(x\right)\right|$',
                r'$\left|f\left(x + \epsilon d\right) - f\left(x\right) - \epsilon d^T g\left(x\right)\right|$'])
    plt.savefig('submission/q1wb.png')
    plt.show()
