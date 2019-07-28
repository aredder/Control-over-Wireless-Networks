import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import csv
from scipy.linalg import block_diag

import system_models


def main():
    # System hyper-parameter
    linear_system_hp = {'system_dimension': 8,
                        'number_of_subsystems': 4,
                        'dependency': 0.1,
                        'ratio_of_stable_subsystems': 0.5,
                        'init_cov_scale': 1,
                        'noise_cov_scale': 0.1,
                        'state_cost_scale': 10,
                        'control_cost_scale': 5
                        }

    # Set system
    system = system_models.LinearSystem(dimension=linear_system_hp['system_dimension'],
                                        subsystems=linear_system_hp['number_of_subsystems'],
                                        init_cov=np.eye(linear_system_hp['system_dimension']) * linear_system_hp[
                                            'init_cov_scale'],
                                        noise_cov=np.eye(linear_system_hp['system_dimension']) * linear_system_hp[
                                            'noise_cov_scale'],
                                        dependency=linear_system_hp['dependency'],
                                        stability=linear_system_hp['ratio_of_stable_subsystems'])
    system.q_system = np.eye(np.shape(system.A)[0]) * linear_system_hp['state_cost_scale']
    system.r_system = np.eye(np.shape(system.B)[1]) * linear_system_hp['control_cost_scale']

    # Network hyper-parameter
    network_markov_chain_hp = {'model_quantities': [1, 2],
                               'model_0_dynamics': [0.1, 0.9, 0.95, 0.75],
                               'model_1_dynamics': [0.2, 0.8, 0.6, 0.4]}

    # Set network
    resources = sum(network_markov_chain_hp['model_quantities'])
    good_channel = system_models.GilbertElliot(network_markov_chain_hp['model_0_dynamics'][0],
                                               network_markov_chain_hp['model_0_dynamics'][1],
                                               network_markov_chain_hp['model_0_dynamics'][2],
                                               network_markov_chain_hp['model_0_dynamics'][3])
    bad_channel = system_models.GilbertElliot(network_markov_chain_hp['model_1_dynamics'][0],
                                              network_markov_chain_hp['model_1_dynamics'][1],
                                              network_markov_chain_hp['model_1_dynamics'][2],
                                              network_markov_chain_hp['model_1_dynamics'][3])
    channels = \
        [good_channel] * network_markov_chain_hp['model_quantities'][0] + [bad_channel] * \
        network_markov_chain_hp['model_quantities'][1]

    quality = [good_channel.error_stationary, bad_channel.error_stationary]

    delta_0 = \
        sum([a * (1 - b) for a, b in zip(network_markov_chain_hp['model_quantities'], quality)]) / \
        linear_system_hp['number_of_subsystems']

    # if 1 - 1 / np.max(np.abs(np.linalg.eigvals(system.A)) ** 2) < delta_0:
    #     for a in system.subA:
    #         print(np.linalg.eigvals(a))
    #     raise ValueError('System should be unstable under random policy')
    # print('System is unstable under random policy')

    # Compute initial controller based on success rate of intelligent random policy
    action_size = system.subsystems
    lambda_max = np.array([np.max(np.abs(np.linalg.eigvals(a))) for a in system.subA])
    int_rnd_policy = np.array([x / sum(lambda_max) for x in lambda_max])

    monte_carlo_average = 100000
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

    adaptive_controller = system_models.AdaptiveLinearController(system, p_success, 0.9)
    optimal_avg_loss = np.trace(sp.linalg.solve_discrete_are(system.A, system.B, system.q_system, system.r_system)
                                @system.noise_cov)
    # Agent hyper-parameter
    learning_hp = {'memory': 30000,
                   'gamma': 0.95,
                   'epsilon': 1,
                   'epsilon_min': 0.001,
                   'epsilon_decay': 0.999,
                   'learning_rate': np.exp(-3),
                   'learning_rate_decay': 0,
                   'hidden_layers': 1,
                   'neurons_per_layer': 1024,
                   'epochs': 40,
                   'batch_size': 64,
                   'target_update': 100,
                   'horizon': 500,
                   'combined_Q': True,          # Sutton: Deeper Look at Exp. Replay
                   'double_pick': True,         # Allow agent to give more than one resource per subsystem
                   'terminal_Q': True,          # Use terminal reward and Q-value as targets for every artificial state
                   'penalize': None}            # Penalize double pick, if double pick == False

    learning_rate_sweep = [np.exp(-4.5)]
    number_of_training_runs = 10
    save_path = 'C:/Users/adria/sciebo/9_Ma_Thesis/Training_results/'

    selection_1 = [0, 2, 3, 4, 5, 7, 9]
    selection_2 = [0, 1, 3, 4, 5, 6, 7, 8]

    for i in selection_1:
        with open(os.path.join('V6/with/run'+str(i) + '/', 'loss_data_iterative_agent.csv'), 'r') as f:
            reader = csv.reader(f)
            new_list = list(reader)
            if i == selection_1[0]:
                loss_data_with = np.array(new_list).astype(np.float).transpose()
            else:
                loss_data_with = np.vstack([loss_data_with, np.array(new_list).astype(np.float).transpose()])

    for i in selection_2:
        with open(os.path.join('V6/without/run'+str(i) + '/', 'loss_data_iterative_agent.csv'), 'r') as f:
            reader = csv.reader(f)
            new_list = list(reader)
            if i == selection_2[0]:
                loss_data_without = np.array(new_list).astype(np.float).transpose()
            else:
                loss_data_without = np.vstack([loss_data_without, np.array(new_list).astype(np.float).transpose()])

    loss_mean_with = np.mean(loss_data_with, axis=0)
    loss_var_with = np.var(loss_data_with, axis=0)
    loss_mean_without = np.mean(loss_data_without, axis=0)
    loss_var_without = np.var(loss_data_without, axis=0)


    fig1, ax1 = plt.subplots()
    ax1.plot(loss_mean_with, label='Iterative agent with adaptive control')
    ax1.plot(loss_mean_without, label='Iterative agent')

    lower_bound = [x if x > optimal_avg_loss else optimal_avg_loss for x in loss_mean_with - 2 * np.sqrt(loss_var_with)]
    ax1.fill_between(np.arange(learning_hp['epochs']), loss_mean_with
                     + 2 * np.sqrt(loss_var_with), lower_bound, color='blue', alpha=0.2)

    #lower_bound = [x if x > optimal_avg_loss else optimal_avg_loss for x in loss_mean_without - 2 * np.sqrt(loss_var_without)]
    #ax1.fill_between(np.arange(learning_hp['epochs']), loss_mean_without
    #                 + 2 * np.sqrt(loss_var_without), lower_bound, color='orange', alpha=0.2)
    ax1.legend()
    ax1.axhline(y=optimal_avg_loss, color='r')
    ax1.set_ylim(0, 200)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Empirical average loss')
    #plt.savefig(os.path.join(save_path, 'avg_loss_1.png'), bbox_inches='tight')
    plt.savefig('avg_loss_1.png', bbox_inches='tight')
    ax1.set_ylim(0, 200)
    #plt.savefig(os.path.join(save_path, 'avg_loss_2.png'), bbox_inches='tight')
    plt.savefig('avg_loss_2.png', bbox_inches='tight')

    print(optimal_avg_loss)
    print(np.min(loss_mean_with))
    print(np.min(loss_mean_without))
if __name__ == "__main__":
    main()
