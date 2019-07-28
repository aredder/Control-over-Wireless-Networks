from network_env.network_classes import pac_generator
import matplotlib.pyplot as plt
import numpy as np


def main():

    users = [12, 6, 2]
    packets = [2, 6, 10]
    pac_num = np.inner(users, packets)
    time = pac_generator(5, pac_num, 2, offset_start=0)

    _, _, _ = plt.hist(time, 10, density=True)
    plt.show()
    print(time)

if __name__ == "__main__":
    main()
