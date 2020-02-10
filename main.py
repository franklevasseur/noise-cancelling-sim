import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

N: int = 8000
sampling_freq: float = 50
p: int = 100
base_learn_factor: float = 1


# butterworth d'ordre n = 2 et
# fréquence de coupure wc = 0.3 * nyquist
def h(x: np.ndarray) -> np.ndarray:
    b, a = sig.butter(1, 0.01, analog=False, output='ba')
    return sig.lfilter(b, a, x)


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

    for n, xn in enumerate(x):
        # append current x to xbold window
        xbold = np.roll(xbold, -1)
        xbold[p - 1] = xn

        xnorm = np.sum(xbold ** 2)
        learn_factor = (1 / xnorm) if xnorm > 0 else base_learn_factor

        ybar[n] = np.sum(np.multiply(xbold, hbar))

        # update weights
        residual = d[n] - ybar[n]
        hbar += learn_factor * residual * xbold

    residuals = d - ybar[:N]

    # plt.plot(time, x, linewidth=0.1, label="x(t) actual noise")
    plt.plot(time, d, color="green", linewidth=0.8, label="d(t) desired")
    plt.plot(time, ybar[:N], color="blue", label="y_hat(t) cancelling signal")
    plt.plot(time, residuals, color="r", linewidth=0.3, label="e(t) residual")

    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('ACN with LMS')
    plt.legend()
    plt.show()

    # yf = ft.fft(weights)
    # xf = np.linspace(0.0, 1.0 / (2.0 / sampling_freq), p)
    # yf_norm = (2.0 / N) * np.abs(yf[:N // 2])
    #
    # plt.xlabel('frequency (unknown units...)')
    # plt.ylabel('|H(jw)|')
    # plt.title('fourier transform of H(n)')
    # plt.plot(xf, yf)
    # plt.show()

    print("\n#################################")
    print("####### Program Ending... #######")
    print("#################################\n")
