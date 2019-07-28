from network_env.network_classes import generate_bounded_random_walk, generate_user_request_traffic
import matplotlib.pyplot as plt


def main():
    number_of_users = generate_bounded_random_walk(1000, lower_bound=10, upper_bound=30)
    plt.figure(figsize=(8, 6))
    plt.plot(number_of_users)
    plt.show()

    random_walk = generate_user_request_traffic(number_of_users, class_dist=[0.6, 0.3, 0.1],
                                                p_val_list=[[2/3, 1/3], [1/6, 1/3, 1/3, 1/6],
                                                            [1/12, 1/6, 1/4, 1/4, 1/6, 1/12]])
    plt.figure(figsize=(8, 6))
    plt.plot(random_walk)
    plt.show()


if __name__ == "__main__":
    main()
