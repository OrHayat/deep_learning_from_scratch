import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from numpy import linalg as la
from scipy.sparse import spdiags


def jacobi(A, b, x0, max_iter, e, w=1.0):
    residual_norm = []
    convergence_factor = []

    k = 0
    x = {k: x0}

    D = np.diagflat(np.diag(A))
    LU = A - D
    for k in range(1, max_iter):
        x[k] = np.dot(1 - w, x[k - 1]) + np.dot(w, np.matmul(la.inv(D), (b - np.matmul(LU, x[k - 1]))))

        residual_norm.append(la.norm(np.matmul(A, x[k]) - b, ord=2))
        convergence_factor.append(la.norm(np.matmul(A, x[k]) - b, ord=2) / la.norm(np.matmul(A, x[k - 1]) - b, ord=2))

        if la.norm(np.matmul(A, x[k]) - b, ord=2) / la.norm(b, ord=2) < e:
            break
    return x[k], residual_norm, convergence_factor, k


def gauss_seidel(A, b, x0, max_iter, e):
    residual_norm = []
    convergence_factor = []

    k = 0
    x = {k: x0}

    DL = np.tril(A)
    U = A - DL
    for k in range(1, max_iter):
        x[k] = np.matmul(la.inv(DL), (b - np.matmul(U, x[k - 1])))

        residual_norm.append(la.norm(np.matmul(A, x[k]) - b, ord=2))
        convergence_factor.append(la.norm(np.matmul(A, x[k]) - b, ord=2) / la.norm(np.matmul(A, x[k - 1]) - b, ord=2))

        if la.norm(np.matmul(A, x[k]) - b, ord=2) / la.norm(b, ord=2) < e:
            break
    return x[k], residual_norm, convergence_factor, k


def steepest_descent(A, b, x0, max_iter, e):
    residual_norm = []
    convergence_factor = []

    k = 0
    r = {}
    x = {k: x0}

    r[0] = b - np.matmul(A, x[0])
    for k in range(1, max_iter):
        alpha = np.dot(r[k - 1], r[k - 1]) / np.dot(r[k - 1], np.matmul(A, r[k - 1]))
        x[k] = x[k - 1] + alpha * r[k - 1]
        r[k] = b - np.matmul(A, x[k])

        residual_norm.append(la.norm(r[k], ord=2))
        convergence_factor.append(la.norm(r[k], ord=2) / la.norm(r[k - 1], ord=2))

        if la.norm(np.matmul(A, x[k]) - b, ord=2) / la.norm(b, ord=2) < e:
            break
    return x[k], residual_norm, convergence_factor, k


def conjugate_gradient(A, b, x0, max_iter, e):
    residual_norm = []
    convergence_factor = []

    k = 0
    p = {}
    r = {}
    x = {k: x0}

    p[0] = b - np.matmul(A, x[0])
    r[0] = b - np.matmul(A, x[0])
    for k in range(1, max_iter):
        Ap_k_prev = np.matmul(A, p[k - 1])
        alpha = np.dot(r[k - 1], p[k - 1]) / np.dot(p[k - 1], Ap_k_prev)
        x[k] = x[k - 1] + alpha * p[k - 1]
        r[k] = b - np.matmul(A, x[k])

        residual_norm.append(la.norm(r[k], ord=2))
        convergence_factor.append(la.norm(r[k], ord=2) / la.norm(r[k - 1], ord=2))

        if la.norm(np.matmul(A, x[k]) - b, ord=2) / la.norm(b, ord=2) < e:
            break
        beta = -np.dot(r[k], Ap_k_prev) / np.dot(p[k - 1], Ap_k_prev)
        p[k] = r[k] + beta * p[k - 1]
    return x[k], residual_norm, convergence_factor, k


def GMRES(A, b, x0, max_iter, e):
    residual_norm = []
    convergence_factor = []

    k = 0
    r = {}
    x = {k: x0}

    r[0] = b - np.matmul(A, x[0])
    for k in range(1, max_iter):
        Ar_k = np.matmul(A, r[k - 1])
        alpha = np.dot(r[k - 1], Ar_k) / np.dot(Ar_k, Ar_k)
        x[k] = x[k - 1] + alpha * r[k - 1]
        r[k] = b - np.matmul(A, x[k])

        residual_norm.append(la.norm(r[k], ord=2))
        convergence_factor.append(la.norm(r[k], ord=2) / la.norm(r[k - 1], ord=2))

        if la.norm(np.matmul(A, x[k]) - b, ord=2) / la.norm(b, ord=2) < e:
            break
    return x[k], residual_norm, convergence_factor, k


def sanity_check():
    A = np.array([[1, 2, -3],
                  [2, -5, 1],
                  [-3, 1, 5]])
    b = np.array([-20, -25, 70])
    x0 = np.array([0, 0, 0])
    # x = np.array([5, 10, 15])
    max_iter = 1000
    e = 10e-5

    res = {1: jacobi(A, b, x0, max_iter, e)[0],
           2: gauss_seidel(A, b, x0, max_iter, e)[0],
           3: steepest_descent(A, b, x0, max_iter, e)[0],
           4: conjugate_gradient(A, b, x0, max_iter, e)[0]
           }
    print(res)


def plot_result(title, result, filename):
    residual_norm = result[1]
    convergence_factor = result[2]
    iterations = range(result[3])

    fig, axs = plt.subplots(2, 1, constrained_layout=True)
    fig.suptitle(title, fontsize=16)

    axs[0].semilogy(iterations, residual_norm)
    axs[0].set_title('Residual Norm')
    axs[0].set_xlabel('Iterations')
    axs[0].set_ylabel(r'$||Ax^{(k)}-b||$')

    axs[1].semilogy(iterations, convergence_factor)
    axs[1].set_title('Convergence Factor')
    axs[1].set_xlabel('Iterations')
    axs[1].set_ylabel(r'$\frac{||Ax^{(k)}-b||}{||Ax^{(k-1)}-b||}$')

    fig.savefig(f"resources/{filename}.pdf", bbox_inches='tight')
    plt.show(block=True)


def q1():
    n = 100
    data = np.array([-np.ones(n), 2.1 * np.ones(n), -np.ones(n)])
    diags = np.array([-1, 0, 1])
    A = spdiags(data, diags, n, n).toarray()
    b = np.random.rand(n)
    x0 = np.zeros(n)
    max_iter = 100
    e = 1e-5

    results = {
        "Jacobi w=1.0": jacobi(A, b, x0, max_iter, e, 1.0),
        "Jacobi w=0.75": jacobi(A, b, x0, max_iter, e, 0.75),
        "Gauss Seidel": gauss_seidel(A, b, x0, max_iter, e),
        "Steepest Descent": steepest_descent(A, b, x0, max_iter, e),
        "Conjugate Gradient": conjugate_gradient(A, b, x0, max_iter, e)
    }

    i = 0
    for title, result in results.items():
        i += 1
        plot_result(title, result, f"Q1b{i}")


def q3():
    A = np.array([[5, 4, 4, -1, 0],
                  [3, 12, 4, -5, -5],
                  [-4, 2, 6, 0, 3],
                  [4, 5, -7, 10, 2],
                  [1, 2, 5, 3, 10]])
    b = np.transpose(np.array([1, 1, 1, 1, 1]))
    x0 = np.transpose(np.array([0, 0, 0, 0, 0]))
    max_iter = 50
    e = 1e-5

    plot_result("GMRES", GMRES(A, b, x0, max_iter, e), f"Q3c")


def load_mnist(label1, label2):
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    trainX = mnist.train.images
    trainY = np.asarray(mnist.train.labels, dtype=np.int32)
    testX = mnist.test.images
    testY = np.asarray(mnist.test.labels, dtype=np.int32)

    train_idx = [idx for idx, label in enumerate(trainY) if label == label1 or label == label2]
    test_idx = [idx for idx, label in enumerate(testY) if label == label1 or label == label2]

    trainX = trainX[train_idx]
    trainY = trainY[train_idx]
    testX = testX[test_idx]
    testY = testY[test_idx]

    trainY[trainY == label1] = -1
    trainY[trainY == label2] = -2
    trainY[trainY == -1] = 0
    trainY[trainY == -2] = 1
    trainY = trainY.reshape(trainY.shape[0], 1)

    testY[testY == label1] = -1
    testY[testY == label2] = -2
    testY[testY == -1] = 0
    testY[testY == -2] = 1
    testY = testY.reshape(testY.shape[0], 1)

    trainX = trainX.T
    testX = testX.T

    return trainX, trainY, testX, testY


def sigmoid(X):
    return 1 / (1 + np.exp(-1 * X))


def cost(X, C, W):
    m = X.shape[1]
    A = sigmoid(np.matmul(X.T, W))

    return (-1 / m) * (np.dot(C.T, np.log(A)) +
                       np.dot((1 - C).T, np.log(1 - A)))


def gradient(X, C, W):
    m = X.shape[1]
    A = sigmoid(np.matmul(X.T, W))
    B = (1 / m) * np.dot(X, A - C)
    return B / la.norm(B, ord=2)


def hessian(X, C, W):
    m = X.shape[1]
    A = sigmoid(np.matmul(X.T, W))
    D = np.diag((A * (1 - A)).reshape((-1)))
    return (1 / m) * np.matmul(X, np.matmul(D, X.T))


def gradient_test(X, C, W, d, epsilon, max_iter, title, filename):
    f = cost(X, C, W)
    linear = []
    quadratic = []

    for i in range(0, max_iter):
        e = epsilon ** i
        f_delta = cost(X + (e * d), C, W)
        linear.append(np.abs(f_delta - f)[0])
        quadratic.append(np.abs(f_delta - f - np.dot((e * d.T), gradient(X, C, W)))[0])

    plt.semilogy(range(1, max_iter + 1), linear, 'r')
    plt.title(title + ' (linear)')
    plt.xlabel('Iterations')
    plt.ylabel(r'$|f(x + \epsilon d) - f(x)|$')
    plt.savefig(f"resources/{filename}_linear.pdf", bbox_inches='tight')
    plt.show()

    plt.plot(range(1, max_iter + 1), quadratic, 'b')
    plt.title(title + ' (quadratic)')
    plt.xlabel('Iterations')
    plt.ylabel(r'$|f(x + \epsilon d) - f(x) - \epsilon d^T grad(x)|$')
    plt.savefig(f"resources/{filename}_quadratic.pdf", bbox_inches='tight')
    plt.show()


def hessian_test(X, C, W, d, epsilon, max_iter, title, filename):
    f = cost(X, C, W)
    linear = []
    quadratic = []

    for i in range(0, max_iter):
        e = epsilon ** i
        f_delta = cost(X + (e * d), C, W)
        linear.append(la.norm(f_delta - f))
        quadratic.append(la.norm(f_delta - f - np.dot((e * d.T), hessian(X, C, W))))

    plt.semilogy(range(1, max_iter + 1), linear, 'r')
    plt.title(title + ' (linear)')
    plt.xlabel('Iterations')
    plt.ylabel(r'$||f(x + \epsilon d) - f(x)||$')
    plt.savefig(f"resources/{filename}_linear.pdf", bbox_inches='tight')
    plt.show()

    plt.plot(range(1, max_iter + 1), quadratic, 'b')
    plt.title(title + ' (quadratic)')
    plt.xlabel('Iterations')
    plt.ylabel(r'$||f(x + \epsilon d) - f(x) - JacMV(x, \epsilon d)||$')
    plt.savefig(f"resources/{filename}_quadratic.pdf", bbox_inches='tight')
    plt.show()


def q5b():
    trainX, trainY, testX, testY = load_mnist(0, 1)
    m = testX.shape[0]
    n = testX.shape[1]
    W = np.zeros((m, 1)) + 1e-5
    d = np.random.randn(m, 1)
    epsilon = 0.5
    max_iter = 30

    gradient_test(testX, testY, W, d, epsilon, max_iter, "Gradient test", "Q5b1")
    hessian_test(testX, testY, W, d, epsilon, max_iter, "Jacobian test", "Q5b2")


def armijo_linesearch(X, C, W, grad, d, max_iter=4, alpha0=1.0, beta=0.5, c=1e-4):
    alpha = alpha0
    for i in range(max_iter):
        if cost(X, C, W + alpha * d) <= cost(X, C, W) + c * alpha * np.dot(grad.T, d):
            return alpha
        else:
            alpha = beta * alpha
    return alpha


def gradient_descent_model(trainX, trainY, testX, testY, W, epsilon, max_iter, label, filename):
    train_costs = []
    test_costs = []

    train_cost = 0.0
    test_cost = 0.0
    for i in range(max_iter):
        train_cost, g = cost(trainX, trainY, W).reshape((-1)), \
                        gradient(trainX, trainY, W)
        test_cost = cost(testX, testY, W).reshape((-1))
        if train_cost < epsilon:
            break

        train_costs.append(train_cost)
        test_costs.append(test_cost)
        alpha = armijo_linesearch(trainX, trainY, W, g, g)
        W = W - alpha * g

    train_costs = np.abs(np.array(train_costs) - train_cost)
    test_costs = np.abs(np.array(test_costs) - test_cost)
    plt.semilogy(train_costs, 'r')
    plt.semilogy(test_costs, 'b')
    plt.title('Gradient Descent Convergence: ' + label)
    plt.legend(['Train', 'Test'])
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.savefig(f'resources/{filename}.pdf')
    plt.show()
    return W


def newton_model(trainX, trainY, testX, testY, W, epsilon, max_iter, label, filename):
    train_costs = []
    test_costs = []

    train_cost = 0.0
    test_cost = 0.0
    for i in range(max_iter):
        print(f'iter: {i}')
        train_cost, grad, hess = cost(trainX, trainY, W).reshape((-1)), \
                                 gradient(trainX, trainY, W), \
                                 hessian(trainX, trainY, W)
        test_cost = cost(testX, testY, W).reshape((-1))
        if train_cost < epsilon:
            break

        train_costs.append(train_cost)
        test_costs.append(test_cost)
        W = W - armijo_linesearch(trainX, trainY, W, grad, grad) *\
            np.linalg.pinv(hess + 0.009 * np.eye(hess.shape[0])).dot(grad)

    train_costs = np.abs(np.array(train_costs) - train_cost)
    test_costs = np.abs(np.array(test_costs) - test_cost)
    plt.semilogy(train_costs, 'r')
    plt.semilogy(test_costs, 'b')
    plt.title('Newton Convergence: ' + label)
    plt.legend(['Train', 'Test'])
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.savefig(f'resources/{filename}.pdf')
    plt.show()

    return W


def q5c():
    epsilon = 1e-10
    max_iter = 100

    trainX, trainY, testX, testY = load_mnist(0, 1)
    m = testX.shape[0]
    W = np.zeros((m, 1)) + 1e-5
    W_gd = gradient_descent_model(trainX, trainY, testX, testY, W, epsilon, max_iter, "0/1", "Q5c1a")
    W_n = newton_model(trainX, trainY, testX, testY, W, epsilon, 15, "0/1", "Q5c1b")

    trainX, trainY, testX, testY = load_mnist(8, 9)
    m = testX.shape[0]
    W = np.zeros((m, 1)) + 1e-5
    W_gd = gradient_descent_model(trainX, trainY, testX, testY, W, epsilon, max_iter, "8/9", "Q5c2a")
    W_n = newton_model(trainX, trainY, testX, testY, W, epsilon, 15, "8/9", "Q5c2b")


if __name__ == '__main__':
    plt.interactive(False)
    # q5b()
    q5c()
