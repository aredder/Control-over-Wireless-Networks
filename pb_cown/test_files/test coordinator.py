from network_env.network_classes import Coordinator
from network_env.network_classes import generate_bounded_random_walk, generate_user_request_traffic
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Assume that the network capacity is given according to the expected user requests at every time-step + N_CFP
    class_dist = [0.6, 0.3, 0.1]
    p_val_list = [[2/3, 1/3], [1/6, 1/3, 1/3, 1/6],[1/12, 1/6, 1/4, 1/4, 1/6, 1/12]]
    user_bounds = [10, 30]
    n_cfp = np.int(np.rint(np.sum(np.inner(np.arange(len(p_val_list[k])), p_val_list[k])*round(class_dist[k]*20)
                    for k in range(len(class_dist)))) + 5)
    network_coordinator = Coordinator(n_cfp)

    horizon = 1000
    number_of_users = generate_bounded_random_walk(horizon, lower_bound=user_bounds[0], upper_bound=user_bounds[1])
    user_request_traffic = generate_user_request_traffic(number_of_users, class_dist=class_dist, p_val_list=p_val_list)

    plt.figure(figsize=(8, 6))
    plt.plot(user_request_traffic)
    plt.show()

    limit_ind = []
    n_cfp_k = []
    for k in range(horizon):
        request_in = [0]*5 + [1]*user_request_traffic[k][0]
        network_coordinator.add_requests(request_in)
        request_out = network_coordinator.pop_requests()
        limit_ind.append(len(request_out) == n_cfp)
        n_cfp_k.append(request_out.count(0))

    plt.figure(figsize=(8, 6))
    plt.plot(limit_ind)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(n_cfp_k)
    plt.show()


if __name__ == "__main__":
    main()
