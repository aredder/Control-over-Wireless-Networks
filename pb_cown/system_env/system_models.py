import random
import time

from scipy.linalg import block_diag
import numpy as np
import scipy as sp

from system_env import tools


class LinearSystem(object):
    def __init__(self, dimension, subsystems, init_cov, noise_cov, dependency=0,
                 stability=0.75, optional_seed=None):
        """
        This is a class for discrete time linear dynamical systems.

        Parameters
        ----------
        dimension : int
            Dimension of the dynamical system (state space).
        subsystems : int
            Number of subsystems.
        dependency : float
            Activate week dependency between subsystems.
        stability: float
            Ratio of stable subsystems.
        """

        if subsystems == 8:
            seed = 2108
            np.random.seed(seed)
            random.seed(seed)
        elif subsystems == 12:
            seed = 87
            np.random.seed(seed)
            random.seed(seed)
        elif subsystems == 16:
            seed = 25
            np.random.seed(seed)
            random.seed(seed)

        if optional_seed is not None:
            np.random.seed(optional_seed)
            random.seed(optional_seed)

        self.dim = dimension
        self.subsystems = subsystems
        self.dependency = dependency
        self.stability = stability
        self.noise_cov = noise_cov
        self.init_cov = init_cov

        self.state = None
        self.q_system = None
        self.r_system = None

        if self.dim < 4:
            raise ValueError('Dimension must be greater than or equal to 4.')
        if self.dim < self.subsystems:
            raise ValueError('Dimension less than number of subsystems.')
        if 0 > self.stability or self.stability > 1:
            raise ValueError('Stability ratio not in [0,1].')

        # Generate random subsystems
        partitions = list(tools.partition(n=self.dim, k=self.subsystems, sing=2))
        partition = partitions[np.random.choice(len(partitions), 1)[0]]
        n_stable = np.int(np.around(self.subsystems*self.stability, decimals=0))
        stability = np.array([1]*n_stable + [0]*np.int(self.subsystems-n_stable))
        np.random.shuffle(stability)

        dyn_mat = []
        inp_mat = []
        for i in range(self.subsystems):
            a, b = tools.generate_random_system(partition[i], 1, marginally_stable=stability[i])
            dyn_mat.append(a)
            inp_mat.append(b)

        self.A = block_diag(*dyn_mat)
        self.subA = dyn_mat
        self.B = block_diag(*inp_mat)
        self.subB = inp_mat
        self.adjacency = np.zeros(self.dim)

        # Add dependency
        self.adjacency = tools.random_graph(n=self.subsystems)
        for i, j in np.argwhere(self.adjacency):
            pos1 = int(np.sum(partition[:-i]))
            pos2 = int(np.sum(partition[:-j]))
            self.A[pos1:pos1+partition[i], pos2:pos2+partition[j]] \
                += self.dependency*(2*np.random.rand(partition[i], partition[j])-1)

        seed = int(time.time())
        np.random.seed(seed)
        random.seed(seed)

    def state_update(self, control, noise=None):
        """
        This function defines the state_update as a function of the control input. Optionally, the external noise
        sequence can be specified manually.

        Parameters
        ----------
        control : float
            system control input
        noise :
            external noise

        Returns
        -------
            Next system.
        """
        if np.shape(control)[0] != self.subsystems:
            raise ValueError('Control dimension does not fit input dimension of the system.')
        if noise is None:
            noise = np.random.multivariate_normal(np.zeros((self.dim,)), self.noise_cov)
        self.state \
            = np.einsum('ij,j->i', self.A, self.state) + np.einsum('ij,j->i', self.B, control) \
            + noise
        return self.state

    def reset_state(self):
        """
        This function resets the state according to the initial state covariance.

        Returns
        -------

        """
        self.state = np.random.multivariate_normal(np.zeros((self.dim,)), self.init_cov)

    def set_state(self, new_state):
        """
        This function allows to set the system state manually.

        Parameters
        ----------
        new_state: float
            Selected external new system state
        """
        if np.shape(new_state)[0] != self.dim:
            raise ValueError('Selected state does not fit defined system dimension.')
        self.state = new_state

    def set_system(self, a, b):
        """
        This function allows to set the system dynamics manually.

        Parameters
        ----------
        a: float
            Selected external system dynamic matrix
        b: float
            Selected external input dynamic matrix
        """
        self.A = a
        self.B = b


class AdaptiveLinearController(object):
    def __init__(self, linear_system, success_rates, adaptive):
        """
        This is a class for an adaptive linear quadratic controller, with independent success rates for each actuator
        dynamic.

        Parameters
        ----------
        linear_system: class
            Associated linear system according to the class LinearSystem.
        success_rates: iterable float
            Specified success rates for each controller-actuator link.
        adaptive: bool
            Option to activate controller updates.
        """
        self.A = linear_system.A
        self.subB = linear_system.subB
        self.Q = linear_system.q_system
        self.R = linear_system.r_system
        self.success_rates = success_rates
        self.adaptive = adaptive

        input_matrix_list = []
        for b, p in zip(self.subB, self.success_rates):
            input_matrix_list.append(b * p)
        input_matrix = block_diag(*input_matrix_list)

        self.K_inf = sp.linalg.solve_discrete_are(self.A, input_matrix, self.Q, self.R)
        self.controller \
            = -np.linalg.inv(input_matrix.transpose() @ self.K_inf @ input_matrix + self.R) \
            @ input_matrix.transpose() @ self.K_inf @ self.A

    def controller_update(self, new_success_rates):
        """

        Parameters
        ----------
        new_success_rates: iterable float
            New success rate.
        Returns
        -------

        """
        if self.adaptive:
            self.success_rates = new_success_rates
            input_matrix_list = []
            for b, p in zip(self.subB, self.success_rates):
                input_matrix_list.append(b * p)
            input_matrix = block_diag(*input_matrix_list)

            try:
                self.K_inf = sp.linalg.solve_discrete_are(self.A, input_matrix, self.Q, self.R)
                self.controller \
                    = -np.linalg.inv(input_matrix.transpose() @ self.K_inf @ input_matrix + self.R) \
                    @ input_matrix.transpose() @ self.K_inf @ self.A
            except ValueError:
                raise ValueError('No ARE solution')

    def controller_one_step(self, actor_decision):
        """

        Parameters
        ----------
        actor_decision: iterable binary
            Decision variable to activate specific actuators.
        Returns
        -------
            Controller associated with decision variable.
        """
        input_matrix_list = []
        for b, p, d in zip(self.subB, self.success_rates, actor_decision):
            input_matrix_list.append(b * p * d)
        input_matrix = block_diag(*input_matrix_list)
        return \
            -np.linalg.inv(input_matrix.transpose() @ self.K_inf @ input_matrix + self.R) \
            @ input_matrix.transpose() @ self.K_inf @ self.A


class BaseNetworkGE(object):
    def __init__(self, channels):
        """
        This is a class to store a collection of network channels
        Parameters
        ----------
        channels
        """
        self.channels = channels

    def output(self, schedule, samples):
        successful_transmissions = sum([np.array(x) * y.transmission(z) for x, y, z in zip(schedule, self.channels,
                                                                                           samples)])
        return [1 if x > 0 else 0 for x in successful_transmissions]


class GilbertElliot(object):
    def __init__(self, p, r, k, h):
        """
        This class specifies a basic two state Markov channel (Gilbert Elliot Model).
        Parameters
        ----------
        p: float
            Transition probability from good state to bad state.
        r: float
            Transition probability from bad state to good state.
        k: float
            Success rate in good state.
        h: float
            Success rate in good state.
        """
        self.GtB = p
        self.BtG = r
        self.G_success = k
        self.B_success = h
        self.state = None
        self.error_stationary = (1-k)*r/(p+r) + (1-h)*p/(p+r)

    def initialize(self, sample):
        if sample < self.BtG / (self.GtB + self.BtG):
            self.state = True
        else:
            self.state = False

    def transmission(self, sample):
        if self.state:
            if sample[0] < self.GtB:
                self.state = False
            if sample[1] < self.G_success:
                return 1
            else:
                return 0
        else:
            if sample[0] < self.BtG:
                self.state = True
            if sample[1] < self.B_success:
                return 1
            else:
                return 0
