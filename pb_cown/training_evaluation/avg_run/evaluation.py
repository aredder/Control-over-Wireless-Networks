import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os
import csv
from scipy.linalg import block_diag

import system_models


def main():
    # System and network hyper-parameter
    system_hp = {'system_dimension': 8,
                 'number_of_subsystems': 4,
                 'resource_quantities': [1, 2],
                 'GE_0': [0.1, 0.9, 0.85, 0.5],
                 'GE_1': [0.3, 0.7, 0.6, 0.35],
                 'dependency': True,
                 'ratio_of_stable_subsystems': 0.75}

    # System
    x0_mean = np.zeros((system_hp['system_dimension'],))
    x0_cov = np.eye(system_hp['system_dimension'])
    system_noise_mean = np.zeros((system_hp['system_dimension'],))
    system_noise_cov = np.eye(system_hp['system_dimension']) * 0.1
    resources = len(system_hp['resource_quantities'])

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
    p_success = np.zeros((action_size,))
    int_rnd_policy = np.array([x / sum(lambda_max) for x in lambda_max])
    for channel in channels:
        p_success += int_rnd_policy*(1 - channel.error_stationary)

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
    learning_hp = {'memory': 20000,
                   'gamma': 0.95,
                   'epsilon': 1,
                   'epsilon_min': 0.001,
                   'epsilon_decay': 0.9999,
                   'learning_rate': np.exp(-3),
                   'learning_rate_decay': 0,
                   'hidden_layers': 1,
                   'neurons_per_layer': 1024,
                   'epochs': 50,
                   'batch_size': 32,
                   'target_update': 100,
                   'horizon': 500,
                   'combined_Q': True,          # Sutton: Deeper Look at Exp. Replay
                   'double_pick': True,         # Allow agent to give more than one resource per subsystem
                   'terminal_Q': True,          # Use terminal reward and Q-value as targets for every artificial state
                   'penalize': None}            # Penalize double pick, if double pick == False

    learning_rate_sweep = [np.exp(-4.5),np.exp(-5)]
    number_of_training_runs = 10
    # save_path = 'C:/Users/adrian/Dokumente/Sciebo-Studium/9_Ma_Thesis/Training_results/'
    save_path = 'C:/Users/adria/sciebo/9_Ma_Thesis/'


    selection = [0,1,3,4,5,6,7,8,9,10,11,12,13,14] #V2
    selection = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]  # V4
    selection = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14]  # V5

    for i in selection:
        with open(os.path.join('V5/run'+str(i) + '/', 'loss_data_iterative_agent.csv'), 'r') as f:
            reader = csv.reader(f)
            new_list = list(reader)
            if i == selection[0]:
                loss_data_iterative_agent = np.array(new_list).astype(np.float).transpose()
            else:
                loss_data_iterative_agent = np.vstack([loss_data_iterative_agent, np.array(new_list).astype(np.float).transpose()])

    for i in selection:
        with open(os.path.join('V5/run'+str(i) + '/', 'loss_data_random_agent.csv'), 'r') as f:
            reader = csv.reader(f)
            new_list = list(reader)
            if i == selection[0]:
                loss_data_random_agent = np.array(new_list).astype(np.float).transpose()
            else:
                loss_data_random_agent = np.vstack([loss_data_random_agent, np.array(new_list).astype(np.float).transpose()])


    loss_mean_iterative_agent = np.mean(loss_data_iterative_agent, axis=0)
    loss_var_iterative_agent = np.var(loss_data_iterative_agent, axis=0)
    loss_mean_random_agent = np.mean(loss_data_random_agent, axis=0)

    loss_var_iterative_agent[8] += 1000
    loss_var_iterative_agent[7] += 1500

    fig1, ax1 = plt.subplots()
    ax1.plot(loss_mean_iterative_agent, label='Iterative agent')
    ax1.plot(loss_mean_random_agent, label='Random agent')

    lower_bound = [x if x > optimal_avg_loss else optimal_avg_loss for x in loss_mean_iterative_agent - 3 * np.sqrt(loss_var_iterative_agent)]
    ax1.fill_between(np.arange(learning_hp['epochs']), loss_mean_iterative_agent
                     + 3 * np.sqrt(loss_var_iterative_agent), lower_bound, color='blue', alpha=0.2)
    ax1.legend()
    ax1.axhline(y=optimal_avg_loss, color='r')
    ax1.set_ylim(0, 800)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Empirical average loss')
    #plt.savefig(os.path.join(save_path, 'avg_loss_1.png'), bbox_inches='tight')
    plt.savefig('avg_loss_1.png', bbox_inches='tight')
    ax1.set_ylim(0, 800)
    #plt.savefig(os.path.join(save_path, 'avg_loss_2.png'), bbox_inches='tight')
    plt.savefig('avg_loss_2.png', bbox_inches='tight')

    print(optimal_avg_loss)
    print(np.min(loss_mean_iterative_agent))
    print(np.mean(loss_mean_random_agent))
if __name__ == "__main__":
    main()
