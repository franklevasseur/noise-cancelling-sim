import matplotlib.pyplot as plt
import scipy.signal as sig

if __name__ == '__main__':

    print("#################################")
    print("###### Program Starting... ######")
    print("#################################\n")

    lti = sig.lti([1.0], [1.0, 1.0]) # LIT === linéaire et invariant en temps
    t, y = sig.step(lti)
    plt.plot(t, y)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Step response for 1. Order Lowpass')
    plt.grid()
    plt.show()

    print("\n#################################")
    print("####### Program Ending... #######")
    print("#################################\n")