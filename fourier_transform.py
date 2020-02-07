import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as ft

if __name__ == '__main__':

    print("#################################")
    print("###### Program Starting... ######")
    print("#################################\n")

    N: int = 600

    T = 1.0 / 800.0

    x = np.linspace(0.0, N * T, N)
    y = np.sin(50.0 * 2.0 * np.pi * x) + 0.5 * np.sin(80.0 * 2.0 * np.pi * x)
    yf = ft.fft(y)
    xf = np.linspace(0.0, 1.0 / (2.0 * T), int(N / 2))
    yf_norm = (2.0 / N) * np.abs(yf[:N // 2])

    fig, ax = plt.subplots(2)
    ax[1].plot(xf, yf_norm, label="fourier transform")
    ax[0].plot(x, y, label="time domain")
    plt.legend()
    plt.show()

    print("\n#################################")
    print("####### Program Ending... #######")
    print("#################################\n")