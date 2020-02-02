import numpy as np
import matplotlib.pyplot as plt

N: int = 1000
Kp: float = 0.01

Kproc: float = -10.0
tau: float = 5


def process(input_cmd):
    return Kproc * (np.exp(- input_cmd / tau) - 1)


if __name__ == '__main__':

    print("#################################")
    print("###### Program Starting... ######")
    print("#################################\n")

    time = np.arange(N)

    ref = 5 * np.ones(N)
    answer = np.zeros(N + 1)
    command = np.zeros(N + 1)

    for t, i in enumerate(time):
        residual = ref[i] - answer[i]
        command[i + 1] = command[i] + Kp * residual
        answer[i + 1] = process(command[i + 1])

    plt.plot(time, ref, label="reference")
    plt.plot(time, answer[0:N], label="answer")
    plt.legend()
    plt.show()

    print("\n#################################")
    print("####### Program Ending... #######")
    print("#################################\n")