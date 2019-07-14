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

    # x = np.random.randn(sample_size, sample_count)
    # w = np.random.randn(sample_size, labels_count)
    # b = np.random.randn(labels_count, 1)

    # dx = np.random.randn(sample_size, sample_count)
    # dw = np.random.randn(sample_size, labels_count)
    # db = np.random.randn(labels_count, 1)
    # dwb = np.concatenate((db.flatten('F'), dw.flatten('F'))).reshape(18, 1)

    x = np.array([-2.5, -0.15, 0.35, 0.7, -1.3]).reshape(5, 1)
    w = np.array([[-1, -1.6, -1.3], [0.8, 0.75, -1.5], [-0.1, 1.2, -0.04], [0.5, 1.6, -0.6], [-0.96, -1.5, 1.3]])
    b = np.array([1.6, -2, -0.5]).reshape(3, 1)
    dx = np.array([0.2, -0.6, 0.5, 0.01, -0.04]).reshape(5, 1)
    dw = np.array([[0.2, 0.3, -0.7], [-0.8, 0.4, 0.3], [-1.3, -0.9, 2.4], [0.6, -0.5, -0.5], [0.6, -0.1, 0.7]])
    db = np.array([-1, 1.4, -1]).reshape(3, 1)
    dwb = np.concatenate((db.flatten('F'), dw.flatten('F'))).reshape(18, 1)

    iterations = 10
    epsilons = [0.5 ** e for e in range(iterations)]

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

    plt.semilogy(range(iterations), df_x)
    plt.semilogy(range(iterations), dg_x)
    plt.legend(['f', 'g'])
    plt.show()
    # fig, axs = plt.subplots(1, 2)
    # axs[0].plot(range(iterations), df_w)
    # axs[0].plot(range(iterations), dg_w)
    # axs[0].set_title('w, b')
    # axs[0].set(xlabel='Iterations', ylabel='Gradient test')
    # axs[0].legend(['f', 'g'])
    #
    # axs[1].plot(range(iterations), df_x)
    # axs[1].plot(range(iterations), dg_x)
    # axs[1].set_title('x')
    # axs[1].set(xlabel='Iterations', ylabel='Gradient test')
    # axs[1].legend(['f', 'g'])

    plt.show()

    # result_w = []
    # for e in epsilons:
    #     _f = loss(x, c, w + dw * e, b + db * e) - loss(x, c, w, b)
    #     df = np.abs(_f)
    #     dg = np.abs(_f - e * np.matmul(d.T, gradient_loss_w(x, c, w, b)))
    #     result_w.append((df, dg))
    # for (df, dg) in result_w:
    #     print(df, dg)
    # for i in range(1, len(result_w)):
    #     (df1, dg1) = result_w[i-1]
    #     (df2, dg2) = result_w[i]
    #     print(df1 / df2, dg1 / dg2)
    # print()
    # epsilon = 1
    # result_x = []
    # for i in range(iterations):
    #     epsilon *= 0.5
    #     _f = loss(x + dx * epsilon, c, w, b) - loss(x, c, w, b)
    #     df = np.abs(_f)
    #     a = gradient_loss_x(x, c, w, b)
    #     b = np.matmul(dx.T, a)[0]
    #     dg = np.abs(_f - epsilon * np.matmul(dx.T, gradient_loss_x(x, c, w, b))[0][0])
    #     result_x.append((df, dg))
    # for (df, dg) in result_x:
    #     print(df, dg)
    # for i in range(1, len(result_x)):
    #     (df1, dg1) = result_x[i-1]
    #     (df2, dg2) = result_x[i]
    #     print(df1 / df2, dg1 / dg2)
