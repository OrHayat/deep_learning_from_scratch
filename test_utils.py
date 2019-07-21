import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la


def plot_test_results(epsilons, df, dg, func, title, filename, show):
    plt.loglog(epsilons, df)
    plt.loglog(epsilons, dg)
    plt.title = title
    plt.xlabel = 'Epsilon'
    plt.ylabel = 'Value'
    plt.legend([r'$\left|f\left(x + \epsilon d\right) - f\left(x\right)\right|$',
                r'$\left|f\left(x + \epsilon d\right) - f\left(x\right) - ' + func + r'\right|$'])
    plt.savefig(f'results/{filename}.png')
    if show:
        plt.show()


def plot_sgd_results(iterations, train_loss, train_accuracy, test_loss, test_accuracy, title, filename, show):
    fig, ax1 = plt.subplots()
    fig.title = title

    color = 'tab:red'
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(iterations, train_loss, color=color, linestyle='-')
    ax1.plot(iterations, test_loss, color=color, linestyle='--')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(['Train', 'Test'])

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)
    ax2.plot(iterations, train_accuracy, color=color, linestyle='-')
    ax2.plot(iterations, test_accuracy, color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(['Train', 'Test'])

    fig.tight_layout()
    plt.savefig(f'results/{filename}.png')
    if show:
        plt.show()


def gradient_test(f, g, title, filename, iterations=10):
    epsilons = [0.5 ** e for e in range(5, 5 + iterations)]

    g1 = [la.norm(f(e) - f(0)) for e in epsilons]
    g2 = [la.norm(f(e) - f(0) - g(e)) for e in epsilons]

    for i in range(1, iterations):
        print(g1[i - 1] / g1[i], g2[i - 1] / g2[i])
    plot_test_results(epsilons, g1, g2, r'\epsilon d^T g\left(x\right)', title, filename, True)


def jacobian_test(f, j, title, filename, iterations=15):
    epsilons = [0.5 ** e for e in range(5, 5 + iterations)]
    j1 = [la.norm(f(e) - f(0)) for e in epsilons]
    j2 = [la.norm(f(e) - f(0) - j(e)) for e in epsilons]

    for i in range(1, iterations):
        print(j1[i - 1] / j1[i], j2[i - 1] / j2[i])
    plot_test_results(epsilons, j1, j2, r'JacMV\left(x, \epsilon d\right)', title, filename, True)


def transpose_test(f1, f2, u_gen, v_gen, iterations=15):
    u = [u_gen() for _ in range(iterations)]
    v = [v_gen() for _ in range(iterations)]

    res = [np.abs(np.matmul(vi.T, f1(ui)) - np.matmul(ui.T, f2(vi))).item()
           for ui, vi in zip(u, v)]
    for r in res:
        print(r)
