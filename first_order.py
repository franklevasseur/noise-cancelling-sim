import matplotlib.pyplot as plt
import scipy.signal as sig
import numpy as np

N: int = 2000
sampling_freq: float = 100

if __name__ == '__main__':

    print("#################################")
    print("###### Program Starting... ######")
    print("#################################\n")

    time = np.arange(N).reshape(N) / sampling_freq
    x = np.ones(N)
    x[0] = 0
    x[1] = 0.5

    b, a = sig.butter(3, 0.001, analog=False, output='ba')
    y = sig.lfilter(b, a, x)

    plt.plot(time, x)
    plt.plot(time, y)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Step response for 1. Order Lowpass')
    plt.grid()
    plt.show()

    print("\n#################################")
    print("####### Program Ending... #######")
    print("#################################\n")