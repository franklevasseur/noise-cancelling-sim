import numpy as np
import matplotlib.pyplot as plt

from filterMemory import FilterMemory

N: int = 8000
sampling_freq: float = 48000
p: int = 10
base_learn_factor: float = 1
size_penality_factor: float = 0.02

draw_weights: bool = False
draw_signals: bool = True


def h(x: np.ndarray) -> np.ndarray:
    return 0.8 * x


def create_ambiant_noise() -> np.ndarray:
    mu, sigma = 0, 0.01
    return 100 * np.random.normal(mu, sigma, N)


if __name__ == '__main__':

    print("#################################")
    print("###### Program Starting... ######")
    print("#################################\n")

    time = np.arange(N).reshape(N) / sampling_freq

    x = create_ambiant_noise()
    d = h(x)

    ybar = np.zeros(N + p)
    hbar = np.zeros(p)
    xbold = np.zeros(p)
    mem: FilterMemory = FilterMemory(p)

    for n, xn in enumerate(x):
        # append current x to xbold window
        xbold = np.roll(xbold, -1)
        xbold[p - 1] = xn

        xnorm = np.sum(xbold ** 2)
        learn_factor = (1 / xnorm) if xnorm > 0 else base_learn_factor

        ybar[n] = np.sum(np.multiply(xbold, hbar))

        # update weights
        residual = d[n] - ybar[n]
        hbar = hbar * (1 - size_penality_factor) + learn_factor * residual * xbold
        mem.append(hbar)

    residuals = d - ybar[:N]

    if draw_signals:
        # plt.plot(time, x, linewidth=0.1, label="x(t) actual noise")
        plt.plot(time, d, color="green", linewidth=0.8, label="d(t) desired")
        plt.plot(time, ybar[:N], color="blue", label="y_hat(t) cancelling signal")
        plt.plot(time, residuals, color="r", linewidth=0.3, label="e(t) residual")

        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('ACN with LMS')
        plt.legend()
        plt.show()

    if draw_weights:
        allMemories = mem.getAll()
        for i, m in enumerate(allMemories):
            plt.plot(m, label="h({})".format(i))

        plt.legend()
        plt.show()

    print("\n#################################")
    print("####### Program Ending... #######")
    print("#################################\n")
