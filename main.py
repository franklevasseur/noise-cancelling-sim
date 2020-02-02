import numpy as np
import matplotlib.pyplot as plt

N: int = 1000

if __name__ == '__main__':

    print("#################################")
    print("###### Program Starting... ######")
    print("#################################\n")

    time = np.arange(N).reshape(N)

    mu, sigma = 0, 0.1  # mean and standard deviation
    noise = np.random.normal(mu, sigma, N)

    plt.plot(time, noise)
    plt.show()

    print("\n#################################")
    print("####### Program Ending... #######")
    print("#################################\n")