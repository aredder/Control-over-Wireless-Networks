from collections import deque
import numpy as np


class Coordinator:
    def __init__(self, cfp_channels):
        self.cfp = cfp_channels
        self.queue = deque()

    def add_requests(self, requests):
        self.queue.extend(requests)

    def pop_requests(self):
        return [self.queue.popleft() for _ in range(min(self.cfp, len(self.queue)))]


class CapChannels:
    def __init__(self, size):
        self.n_CAP = size
        self.state = [0]*size

    def sense(self):
        return 0

    def transmit(self):
        return 0


def generate_user_request_traffic(number_of_users, class_dist, p_val_list):
    assert (len(class_dist) == len(p_val_list))
    request = []
    for k in range(len(number_of_users)):
        number_in_class = [np.rint(x) for x in class_dist*number_of_users[k]]

        request.append(np.sum(np.inner(np.arange(len(p_val_list[k])),
                                       np.random.multinomial(number_in_class[k], p_val_list[k], size=1))
                              for k in range(len(class_dist))))
    return request


def generate_bounded_random_walk(length, lower_bound, upper_bound):
    random_walk = [np.random.randint(lower_bound, upper_bound, size=1)]
    for k in range(length):
        if random_walk[k] == lower_bound:
            random_walk.append(random_walk[k] + np.random.choice([0, 1]))
        elif random_walk[k] == upper_bound:
            random_walk.append(random_walk[k] + np.random.choice([-1, 0]))
        else:
            random_walk.append(random_walk[k] + np.random.choice([-1, 0, 1]))
    return random_walk


def pac_generator(poisson_interval, pac_num, offset_end, offset_start=0):
    seq = np.random.poisson(poisson_interval, pac_num)
    time_sum = np.random.randint(offset_start, offset_end)
    time = []
    for i in range(len(seq)):
        time_sum = time_sum + seq[i]
        time.append(time_sum)
    return time
