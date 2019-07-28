import os
import csv
import sys
import datetime
import logging
import random

import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from system_env import system_models
from rl_agents import dqn_iterative_agent
from rl_agents import random_intelligent_agent


def main():
    # local or external computation
    if len(sys.argv) > 1:
        external_execution = sys.argv[1].lower() == 'true'
    else:
        external_execution = False

    # Create log files
    start = datetime.datetime.now()
    if external_execution:
        path = sys.argv[2]
    else:
        save_path = 'C:/training_results/'
        path = os.path.join(save_path, start.strftime("%I%M%p%B%d%Y") + '/')
    os.makedirs(path)

    log_path = path + 'training.log'
    logging.basicConfig(format='%(asctime)s %(name)-27s %(levelname)-8s %(processName)-15s %(message)s',
                        level=logging.DEBUG,
                        handlers=[logging.FileHandler(log_path, mode='w'),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__file__)

    # System hyper-parameter
    linear_system_hp = {'system_dimension': 16,
                        'number_of_subsystems': 8,
                        'dependency': 0.1,
                        'ratio_of_stable_subsystems': 0.5,
                        'init_cov_scale': 1,
                        'noise_cov_scale': 0.1,
                        'state_cost_scale': 1,
                        'control_cost_scale': 0.1
                        }

    # Agent hyper-parameter
    learning_hp = {'memory': 50000,
                   'gamma': 0.95,
                   'epsilon': 1,
                   'epsilon_min': 0.001,
                   'epsilon_decay': 0.9999,
                   'learning_rate': np.exp(-4),
                   'learning_rate_decay': 0,
                   'hidden_layers': 1,
                   'neurons_per_layer': 512,
                   'epochs': 50,
                   'sl_epochs': 5,
                   'batch_size': 32,
                   'tau_target': 0.005,
                   'horizon': 500,
                   'combined_Q': True,  # Sutton: Deeper Look at Exp. Replay https://arxiv.org/abs/1712.01275
                   'terminal_Q': True}  # Use terminal reward and Q-value as targets for every artificial state

    # Network hyper-parameter
    network_markov_chain_hp = {'model_quantities': [1, 2],
                               'model_0_dynamics': [0.1, 0.9, 1, 0.9],
                               'model_1_dynamics': [0.15, 0.85, 0.95, 0.85]}

    if external_execution:
        linear_system_hp['system_dimension'] = int(float(sys.argv[3]))*2
        linear_system_hp['number_of_subsystems'] = int(float(sys.argv[3]))
        learning_hp['neurons_per_layer'] = int(float(sys.argv[4]))
        learning_hp['learning_rate'] = float(sys.argv[5])
        learning_hp['epsilon_decay'] = float(sys.argv[6])
        learning_hp['epochs'] = int(float(sys.argv[7]))
        learning_hp['memory'] = int(float(sys.argv[8]))
        learning_hp['sl_epochs'] = int(float(sys.argv[9]))
        learning_hp['batch_size'] = int(float(sys.argv[10]))
        network_markov_chain_hp['model_quantities'] = \
            [int(z*int(float(sys.argv[3]))/4) for z in network_markov_chain_hp['model_quantities']]

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

    logger.debug('System- and network-setup done')

    # Compute initial controller based on success rate of intelligent random policy
    action_size = system.subsystems
    lambda_max = np.array([np.max(np.abs(np.linalg.eigvals(a))) for a in system.subA])
    int_rnd_policy = np.array([x / sum(lambda_max) for x in lambda_max])

    monte_carlo_average = 100000
    delta_samples_int_rnd = np.zeros((action_size, monte_carlo_average))
    delta_samples_rnd = np.zeros((action_size, monte_carlo_average))
    for i in range(monte_carlo_average):
        for channel in channels:
            action_int_rnd = np.random.choice(action_size, 1, p=int_rnd_policy)
            action_rnd = np.random.choice(action_size, 1)
            if np.random.random_sample() > channel.error_stationary:
                channel_outcome = 1
            else:
                channel_outcome = 0
            delta_samples_int_rnd[action_int_rnd[0], i] = \
                delta_samples_int_rnd[action_int_rnd[0], i] + (1 - delta_samples_int_rnd[action_int_rnd[0], i]) * \
                channel_outcome
            delta_samples_rnd[action_rnd[0], i] = \
                delta_samples_rnd[action_rnd[0], i] + (1 - delta_samples_rnd[action_rnd[0], i]) * channel_outcome

    delta_mean_int_rnd = np.mean(delta_samples_int_rnd, axis=1)
    delta_mean_rnd = np.mean(delta_samples_rnd, axis=1)

    n = np.int(linear_system_hp['system_dimension']/linear_system_hp['number_of_subsystems'])
    gamma_mat_int_rnd = sp.linalg.block_diag(*[np.eye(n)*np.sqrt(1-delta_i) for delta_i in delta_mean_int_rnd])
    gamma_mat_rnd = sp.linalg.block_diag(*[np.eye(n)*np.sqrt(1-delta_i) for delta_i in delta_mean_rnd])

    if np.max(np.abs(np.linalg.eigvals(gamma_mat_int_rnd@system.A))) >= 1:
        logger.debug('System is not stable under int random policy')
        # raise ValueError('System is not stable under int random policy')
    print(np.max(np.abs(np.linalg.eigvals(gamma_mat_rnd @ system.A))))
    if np.max(np.abs(np.linalg.eigvals(gamma_mat_rnd@system.A))) < 1:
        logger.debug('System is stable under random policy')
        logger.debug(str(np.max(np.abs(np.linalg.eigvals(gamma_mat_rnd@system.A)))))
        # raise ValueError('System is stable under random policy')

    adaptive_controller = system_models.AdaptiveLinearController(system, delta_mean_int_rnd, True)
    non_adaptive_controller = system_models.AdaptiveLinearController(system, delta_mean_int_rnd, False)
    optimal_avg_loss = np.trace(sp.linalg.solve_discrete_are(system.A, system.B, system.q_system, system.r_system)
                                @system.noise_cov)
    logger.debug('Controller initialized.')

    # Store hyper_parameters
    with open(os.path.join(path, 'system_parameter.csv'), "w") as file:
        writer = csv.writer(file)
        for key, val in linear_system_hp.items():
            writer.writerow([key, val])
    with open(os.path.join(path, 'network_parameter.csv'), "w") as file:
        writer = csv.writer(file)
        for key, val in network_markov_chain_hp.items():
            writer.writerow([key, val])
    with open(os.path.join(path, 'learning_parameter.csv'), "w") as file:
        writer = csv.writer(file)
        for key, val in learning_hp.items():
            writer.writerow([key, val])

    logger.debug('Training started')
    # Generate system noise and channel transitions
    np.random.seed()
    random.seed()
    system_noise = np.random.multivariate_normal(np.zeros((system.dim,)), system.noise_cov,
                                                 learning_hp['epochs']*learning_hp['horizon'])
    channel_init = np.random.sample((resources, learning_hp['epochs']))
    channel_transitions = np.random.sample((2*resources, learning_hp['epochs']*learning_hp['horizon']))
    # Train iterative agent with adaptive LQR
    loss, avg_act_with, agent = \
        dqn_iterative_agent.train_iterative_agent(learning_hp, network_markov_chain_hp,
                                                  system, adaptive_controller,
                                                  int_rnd_policy, channels, system_noise, channel_init,
                                                  channel_transitions, external_execution, logger)
    loss_data_iterative_agent_with = np.array(loss)
    pd.DataFrame(loss_data_iterative_agent_with).to_csv(os.path.join(path, 'loss_data_iterative_agent_with.csv'))

    if agent is not None:
        agent.save(os.path.join(path, 'model_with.h5'))
    logger.debug('Training iterative agent with adaptive LQR finished')

    # Train iterative agent
    loss, avg_act_without, agent = \
        dqn_iterative_agent.train_iterative_agent(learning_hp, network_markov_chain_hp,
                                                  system, non_adaptive_controller,
                                                  int_rnd_policy, channels, system_noise, channel_init,
                                                  channel_transitions, external_execution, logger)
    loss_data_iterative_agent = np.array(loss)
    pd.DataFrame(loss_data_iterative_agent).to_csv(os.path.join(path, 'loss_data_iterative_agent.csv'))
    if agent is not None:
        agent.save(os.path.join(path, 'model.h5'))
    logger.debug('Training iterative agent finished')

    # Random intelligent agent
    loss, agent \
        = random_intelligent_agent.random_intelligent_agent(network_markov_chain_hp,
                                                            system, non_adaptive_controller, channels,
                                                            int_rnd_policy, learning_hp['horizon'],
                                                            learning_hp['epochs'], system_noise,
                                                            channel_init, channel_transitions)

    loss_data_random_agent = np.array(loss)
    pd.DataFrame(loss_data_random_agent).to_csv(os.path.join(path, 'loss_data_random_agent.csv'))
    logger.debug('Random random agent finished')

    plt.figure()
    plt.plot(loss_data_iterative_agent_with, label='Iterative agent with adaptive control')
    plt.plot(loss_data_iterative_agent, label='Iterative agent')
    plt.plot(loss_data_random_agent, label='Random agent')
    plt.axhline(y=optimal_avg_loss, color='r')
    # plt.ylim(0, 250)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Empirical average loss')
    plt.savefig(os.path.join(path, 'loss_fig.png'), bbox_inches='tight')
    logger.debug('Run finished')

    # # Plot average
    # loss_mean_iterative_agent_with = np.mean(loss_data_iterative_agent_with, axis=0)
    # loss_var_iterative_agent_with = np.var(loss_data_iterative_agent_with, axis=0)
    # loss_mean_iterative_agent = np.mean(loss_data_iterative_agent, axis=0)
    # loss_var_iterative_agent = np.var(loss_data_iterative_agent, axis=0)
    # loss_mean_random_agent = np.mean(loss_data_random_agent, axis=0)
    #
    # fig1, ax1 = plt.subplots()
    # ax1.plot(loss_mean_iterative_agent_with, label='Iterative agent with adaptive control')
    # ax1.plot(loss_mean_iterative_agent_with, label='Iterative agent')
    # ax1.plot(loss_mean_random_agent, label='Random agent')
    #
    # lower_bound = [x if x > optimal_avg_loss else optimal_avg_loss for x in
    #                loss_mean_iterative_agent_with - 3 * np.sqrt(loss_var_iterative_agent_with)]
    # ax1.fill_between(np.arange(learning_hp['epochs']), loss_mean_iterative_agent_with
    #                  + 3 * np.sqrt(loss_var_iterative_agent_with), lower_bound, color='blue', alpha=0.2)
    #
    # lower_bound = [x if x > optimal_avg_loss else optimal_avg_loss for x in
    #                loss_mean_iterative_agent - 3 * np.sqrt(loss_var_iterative_agent)]
    # ax1.fill_between(np.arange(learning_hp['epochs']), loss_mean_iterative_agent
    #                  + 3 * np.sqrt(loss_var_iterative_agent), lower_bound, color='orange', alpha=0.2)
    #
    # ax1.axhline(y=optimal_avg_loss, color='r', label='Optimal LQR')
    # ax1.legend()
    # ax1.set_ylim(0, 200)
    # ax1.set_xlabel('Epochs')
    # ax1.set_ylabel('Empirical average loss')
    # plt.savefig(os.path.join(path, 'avg_loss.png'), bbox_inches='tight')

    return 0


if __name__ == "__main__":
    main()
