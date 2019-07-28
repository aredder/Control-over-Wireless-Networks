import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import csv
from scipy.linalg import block_diag
from datetime import datetime

import system_models
import dqn_iterative_agent
import dqn_iterative_agent_noSL


def main():
    # System and network hyper-parameter
    system_hp = {'system_dimension': 16,
                 'number_of_subsystems': 8,
                 'resource_quantities': [2, 4],
                 'GE_0': [0.1, 0.9, 0.95, 0.75],
                 'GE_1': [0.2, 0.8, 0.6, 0.4],
                 'dependency': True,
                 'ratio_of_stable_subsystems': 0.5}

    # System
    x0_mean = np.zeros((system_hp['system_dimension'],))
    x0_cov = np.eye(system_hp['system_dimension'])
    system_noise_mean = np.zeros((system_hp['system_dimension'],))
    system_noise_cov = np.eye(system_hp['system_dimension']) * 0.1
    resources = sum(system_hp['resource_quantities'])

    system = system_models.LinearSystem(dimension=system_hp['system_dimension'],
                                        subsystems=system_hp['number_of_subsystems'],
                                        init_mean=x0_mean, init_cov=x0_cov,
                                        n_mean=system_noise_mean, n_cov=system_noise_cov,
                                        dependent=system_hp['dependency'],
                                        stability=system_hp['ratio_of_stable_subsystems'])
    system.q_system = np.eye(np.shape(system.A)[0]) * 10
    system.r_system = np.eye(np.shape(system.B)[1]) * 5

    # Network
    good_channel = system_models.GilbertElliot(system_hp['GE_0'][0], system_hp['GE_0'][1],
                                               system_hp['GE_0'][2], system_hp['GE_0'][3])
    bad_channel = system_models.GilbertElliot(system_hp['GE_1'][0], system_hp['GE_1'][1],
                                              system_hp['GE_1'][2], system_hp['GE_1'][3])
    channels = \
        [good_channel] * system_hp['resource_quantities'][0] + [bad_channel] * system_hp['resource_quantities'][1]

    # Compute initial controller based on success rate of intelligent random policy
    action_size = system.subsystems
    lambda_max = np.array([np.max(np.abs(np.linalg.eigvals(a))) for a in system.subA])
    int_rnd_policy = np.array([x / sum(lambda_max) for x in lambda_max])

    monte_carlo_average = 50000
    p_success = np.zeros((action_size,))
    for _ in range(monte_carlo_average):
        p_temp = np.zeros((action_size,))
        for channel in channels:
            action = np.random.choice(action_size, 1, p=int_rnd_policy)
            p_temp[action[0]] = 1 - (1-p_temp[action[0]])*channel.error_stationary
        p_success += p_temp
    p_success = p_success / monte_carlo_average

    input_matrix_list = []
    for b, p in zip(system.subB, p_success):
        input_matrix_list.append(b*p)
    input_matrix = block_diag(*input_matrix_list)

    are = sp.linalg.solve_discrete_are(system.A, input_matrix, system.q_system, system.r_system)
    controller \
        = -np.linalg.inv(input_matrix.transpose() @ are @ input_matrix + system.r_system)\
        @ input_matrix.transpose() @ are @ system.A

    optimal_avg_loss = np.trace(sp.linalg.solve_discrete_are(system.A, system.B, system.q_system, system.r_system)
                                @system_noise_cov)
    # Agent hyper-parameter
    learning_hp = {'memory': 30000,
                   'gamma': 0.95,
                   'epsilon': 1,
                   'epsilon_min': 0.001,
                   'epsilon_decay': 0.9999,
                   'learning_rate': np.exp(-3.5),
                   'learning_rate_decay': 0,
                   'hidden_layers': 1,
                   'neurons_per_layer': 1024,
                   'epochs': 30,
                   'batch_size': 32,
                   'target_update': 100,
                   'horizon': 500,
                   'combined_Q': True,          # Sutton: Deeper Look at Exp. Replay
                   'double_pick': True,         # Allow agent to give more than one resource per subsystem
                   'terminal_Q': True,          # Use terminal reward and Q-value as targets for every artificial state
                   'penalize': None}            # Penalize double pick, if double pick == False

    learning_rate_sweep = [np.exp(-4.5)]
    number_of_training_runs = 1
    save_path = 'C:/Users/adria/sciebo/9_Ma_Thesis/Training_results/'

    for lr in learning_rate_sweep:
        path = os.path.join(save_path, datetime.now().strftime('%d%m%Y_%H-%M-%S') + '_comp_lr' + str(np.round(lr, 8))
                            + '/')
        os.makedirs(path)
        learning_hp['learning_rate'] = lr

        w = csv.writer(open(os.path.join(path, 'hyperP.csv'), "w"))
        for key, val in learning_hp.items():
            w.writerow([key, val])

        loss_data_iterative_agent = np.zeros((number_of_training_runs, learning_hp['epochs']))
        loss_data_vanilla_agent = np.zeros((number_of_training_runs, learning_hp['epochs']))

        for i in range(number_of_training_runs):
            sub_path = os.path.join(path, 'run' + str(i) + '/')
            os.makedirs(sub_path)

            # Generate system noise and channel transitions
            system_noise = np.random.multivariate_normal(system_noise_mean, system_noise_cov,
                                                         learning_hp['epochs']*learning_hp['horizon'])
            channel_init = np.random.sample((resources, learning_hp['epochs']))
            channel_transitions = np.random.sample((2*resources, learning_hp['epochs']*learning_hp['horizon']))

            # DQN agent
            loss, avg_act, agent = \
                dqn_iterative_agent_noSL.train_iterative_agent(learning_hp, system_hp, system, controller, int_rnd_policy,
                                                               channels, system_noise, channel_init, channel_transitions)
            loss_data_vanilla_agent[i, :] = np.array(loss)
            np.savetxt(os.path.join(sub_path, 'loss_data_dqn_agent.csv'), np.array(loss), delimiter=",")
            if agent is not None:
                agent.save(os.path.join(sub_path, 'vanilla_model.h5'))

            # Reset channels
            for channel in channels:
                channel.state = None

            # Iterative agent
            loss, avg_act, agent = \
                dqn_iterative_agent.train_iterative_agent(learning_hp, system_hp, system, controller,
                                                          int_rnd_policy,
                                                          channels, system_noise, channel_init, channel_transitions)
            loss_data_iterative_agent[i, :] = np.array(loss)
            np.savetxt(os.path.join(sub_path, 'loss_data_iterative_agent.csv'), np.array(loss), delimiter=",")
            if agent is not None:
                agent.save(os.path.join(sub_path, 'dira_model.h5'))

            # Reset channels
            for channel in channels:
                channel.state = None

            plt.figure()
            plt.plot(loss_data_iterative_agent[i, :], label='DIRA')
            plt.plot(loss_data_vanilla_agent[i, :], label='Vanilla')
            plt.axhline(y=optimal_avg_loss, color='r')
            plt.ylim(0, 2500)
            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('Empirical average loss')
            plt.savefig(os.path.join(sub_path, 'loss_fig.png'), bbox_inches='tight')

        fig1, ax1 = plt.subplots()
        for i in range(number_of_training_runs):
            if i == 0:
                ax1.plot(loss_data_iterative_agent[i, :], color='blue', label='DIRA')
                ax1.plot(loss_data_vanilla_agent[i, :], color='red', label='Vanilla')
            else:
                ax1.plot(loss_data_iterative_agent[i, :], color='blue')
                ax1.plot(loss_data_vanilla_agent[i, :], color='red')

        ax1.legend()
        ax1.axhline(y=optimal_avg_loss, color='g')
        ax1.set_ylim(0, 2500)
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Empirical average loss')
        plt.savefig(os.path.join(path, 'loss_all.png'), bbox_inches='tight')

    for a in system.subA:
        print(np.linalg.eigvals(a))

    return 0


if __name__ == "__main__":
    main()
