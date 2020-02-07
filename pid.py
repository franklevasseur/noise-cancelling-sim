import numpy as np
import matplotlib.pyplot as plt

N: int = 500
sampling_freq: float = 10

Kp: float = 0.1
Ki: float = 0.8
Kd: float = 1


def process(input_cmd):
    Kproc = -10.0
    tau = 5
    return Kproc * (np.exp(- input_cmd / tau) - 1)


if __name__ == '__main__':

    print("#################################")
    print("###### Program Starting... ######")
    print("#################################\n")

    time = np.arange(N) / sampling_freq

    ref = 5 * np.ones(N)
    answer = np.zeros(N + 1)

    command = 0
    cumulative_error = 0
    previous_residuals = list()

    for i, t in enumerate(time):
        residual = ref[i] - answer[i]

        previous_residuals.append(residual)
        derivative = 0 if i < 1 else (previous_residuals[i] - previous_residuals[i - 1]) * sampling_freq

        cumulative_error += (residual / sampling_freq)

        command = (Kp * residual) + (cumulative_error * Ki)
        answer[i + 1] = process(command)

    plt.plot(time, ref, "r--", linewidth=2, label="reference")
    plt.plot(time, answer[0:N], linewidth=1, label="answer")
    # plt.plot(time, (ref - answer[0:N]), "g-", linewidth=1, label="result")
    plt.legend()
    plt.show()

    print("\n#################################")
    print("####### Program Ending... #######")
    print("#################################\n")