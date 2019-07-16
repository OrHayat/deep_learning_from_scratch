import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt


def display(title, xlabel, ylabel, legend=None, path=None):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend is not None:
        plt.legend(legend)
    if path is not None:
        plt.savefig(path, bbox_inches='tight')
    plt.show()

def armijo_linesearch(f, x, grad, d, max_iter=100, alpha0=1.0, beta=0.5, c=1e-2):
    alpha = alpha0
    for i in range(max_iter):
        if f(x + alpha * d) <= f(x) + alpha * c * np.dot(d.T, grad):
            break
        alpha = alpha * beta
    return alpha


def steepest_descent(f, gf, x0, max_iter=1000, epsilon=1e-3):
    x = x0
    costs = [f(x)]
    k = 0
    for k in range(1, max_iter):
        gk = gf(x)
        d = -gk
        alpha = armijo_linesearch(f, x, gk, d, max_iter=20, beta=0.2)
        x = x + alpha * d

        costs.append(f(x))
        if costs[k - 1] - costs[k] <= epsilon:
            break
    return x, costs, k + 1


def q2c():
    max_iter = 1000
    epsilon = 1e-3
    x = np.array([1, 1])

    rho = lambda x: x ** 2
    c_eq1 = lambda x: (3 * x[0] + x[1] - 6)
    c_eq2 = lambda x: x[0] ** 2 + x[1] ** 2 - 5
    c_ieq = lambda x: x[0] ** 2 + x[1] ** 2 - 5 <= 0
    f = lambda mu: lambda x: (x[0] + x[1]) ** 2 - 10 * (x[0] + x[1]) + \
                             mu[0] * rho(c_eq1(x)) + \
                             mu[1] * rho(max(0, c_eq2(x)))
    gf = lambda mu: lambda x: np.matmul(np.array([[2, 2], [2, 2]]), x) + np.array([-10, -10]) + \
                              mu[0] * (np.matmul(np.array([[18, 6], [6, 2]]), x) + np.array([-36, -12])) + \
                              (0 if c_ieq(x) else 1) * mu[1] * 4 * c_eq2(x) * x
    mus = []
    for i in range(-2, 3):
        mus.append(np.array([10 ** i, 10 ** i]))

    for mu in mus:
        x, costs, iterations = steepest_descent(f(mu), gf(mu), x, max_iter, epsilon)
        plt.plot(range(iterations), costs)
    display(title='Convergence',
            xlabel='Iterations',
            ylabel=r'$\left(x_1+x_2\right)^2 - 10\left(x_1+x_2\right), \quad 3x_1+x_2=6, x_1^2+x_2^2\leq5, -x_1\leq0$',
            legend=[r'$\mu = $' + str(mu[0]) for mu in mus],
            path='resources/q2c.pdf')


def projected_steepest_descent(f, gf, x0, max_iter=1000, epsilon=1e-3):
    x = x0
    costs = [f(x)]
    k = 0
    for k in range(1, max_iter):
        gk = gf(x)
        d = -gk
        alpha = armijo_linesearch(f, x, gk, d, max_iter=5)
        x = x + alpha * d
        x[x < 0] = 0

        costs.append(f(x)[0])
        if costs[k - 1] - costs[k] <= epsilon:
            break
    return x, costs, k + 1


def q4c():
    A = np.random.randn(100, 200)
    x = np.random.randn(200)
    perm = np.random.permutation(x.shape[0])
    x[:180] = 0
    x = x[perm]
    eta = np.sqrt(0.01) * np.random.random(100)
    b = (A.dot(x) + eta).reshape((100, 1))

    l = 25
    u = np.zeros((200, 1))
    v = np.zeros((200, 1))
    x0 = np.row_stack((u, v))

    def objective(A, x, b, l):
        u = x[:200, 0].reshape((200, 1))
        v = x[200:, 0].reshape((200, 1))
        return np.linalg.norm(A.dot(u - v) - b) ** 2 + l * np.ones((1, u.shape[0])).dot(u + v)

    def gradient(A, x, b, l):
        u = x[:200, 0].reshape((200, 1))
        v = x[200:, 0].reshape((200, 1))
        du = A.T.dot(A.dot(u - v) - b) + l
        du = du / np.linalg.norm(du)
        dv = -1 * A.T.dot(A.dot(u - v) - b) + l
        dv = dv / np.linalg.norm(dv)
        return np.row_stack((du, dv)).reshape((400, 1))

    f = lambda x: objective(A, x, b, l)
    gf = lambda x: gradient(A, x, b, l)

    lambdas = [0, 1, 5, 10, 25, 50, 100, 200]
    for l in lambdas:
        x_hat, costs, iterations = projected_steepest_descent(f, gf, x0)
        x_hat = x_hat[:200, 0] - x_hat[200:, 0]
        print(r'norm = ' + str(np.linalg.norm(x - x_hat)))
        print(r'avg = ' + str(np.mean(x_hat != 0)))
        plt.plot(range(iterations), costs)
    display(title='Convergence',
            xlabel='Iterations',
            ylabel=r'$||Ax - b||_2^2 + \lambda ||x||_1$',
            legend=[r'$\lambda = $' + str(l) for l in lambdas],
            path='resources/q4c.pdf')


def projected_coordinate_descent(f, gf, x0, max_iter=1000, epsilon=1e-2):
    n = x0.shape[0]
    x = x0
    k = 0
    costs = [f(x)]

    for k in range(1, max_iter):
        x_temp = x.copy()
        for i in range(n):
            x_temp[i] = gf(x_temp, i)
        if la.norm(np.subtract(x, x_temp)) < epsilon:
            break
        else:
            x = x_temp
            costs.append(f(x))
    return x, costs, k


def q3d():
    H = np.array([
        [5, -1, -1, -1, -1],
        [-1, 5, -1, -1, -1],
        [-1, -1, 5, -1, -1],
        [-1, -1, -1, 5, -1],
        [-1, -1, -1, -1, 5],
    ])
    g = np.array([18, 6, -12, -6, 18])
    a = np.array([0, 0, 0, 0, 0])
    b = np.array([5, 5, 5, 5, 5])
    x0 = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

    f = lambda x: 1 / 2 * np.matmul(x.T, np.matmul(H, x)) - np.matmul(x.T, g)
    gf = lambda x, i: max(min((g[i] - np.dot(H[i], x) + H[i][i] * x[i]) / H[i][i], b[i]), a[i])

    x_hat, costs, iterations = projected_coordinate_descent(f, gf, x0)
    print(x_hat)
    print(f(x_hat))

    plt.plot(range(iterations), costs)
    display(title='Convergence',
            xlabel='Iterations',
            ylabel=r'$\frac{1}{2}x^THx - x^Tg, \quad a \leq x \leq b$',
            path='resources/q3d.pdf')


if __name__ == '__main__':
    q2c()
    q3d()
    q4c()