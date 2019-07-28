import os
import sys
import datetime
import logging

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from system_env import system_models
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

    log_path = path + 'training' + start.strftime("%I%M%p%B%d%Y") + '.log'
    logging.basicConfig(format='%(asctime)s %(name)-27s %(levelname)-8s %(processName)-15s %(message)s',
                        level=logging.DEBUG,
                        handlers=[logging.FileHandler(log_path, mode='w'),
                                  logging.StreamHandler()])
    logger = logging.getLogger(__file__)

    # System hyper-parameter
    linear_system_hp = {'system_dimension': 32,
                        'number_of_subsystems': 16,
                        'dependency': 0.1,
                        'ratio_of_stable_subsystems': 0.5,
                        'init_cov_scale': 1,
                        'noise_cov_scale': 0.1,
                        'state_cost_scale': 1,
                        'control_cost_scale': 0.1
                        }

    # Agent hyper-parameter
    learning_hp = {'memory': 30000,
                   'gamma': 0.95,
                   'epsilon': 1,
                   'epsilon_min': 0.001,
                   'epsilon_decay': 0.999,
                   'learning_rate': np.exp(-4),
                   'learning_rate_decay': 0,
                   'hidden_layers': 1,
                   'neurons_per_layer': 1024,
                   'epochs': 100,
                   'batch_size': 32,
                   'target_update': 100,
                   'horizon': 500,
                   'combined_Q': True,  # Sutton: Deeper Look at Exp. Replay
                   'double_pick': True,  # Allow agent to give more than one resource per subsystem
                   'terminal_Q': True,  # Use terminal reward and Q-value as targets for every artificial state
                   'penalize': None}  # Penalize double pick, if double pick == False

    # Network hyper-parameter
    network_markov_chain_hp = {'model_quantities': [4, 8],
                               'model_0_dynamics': [0.1, 0.9, 1, 0.9],
                               'model_1_dynamics': [0.15, 0.85, 0.95, 0.85]}

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
        print(np.max(np.abs(np.linalg.eigvals(gamma_mat_int_rnd@system.A))))
        logger.debug('System is not stable under int random policy')
        raise ValueError('System is not stable under int random policy')
    print(np.max(np.abs(np.linalg.eigvals(gamma_mat_rnd @ system.A))))
    if np.max(np.abs(np.linalg.eigvals(gamma_mat_rnd@system.A))) < 1:
        print(np.max(np.abs(np.linalg.eigvals(gamma_mat_rnd@system.A))))
        logger.debug('System is stable under random policy')
        raise ValueError('System is stable under random policy')

    adaptive_controller = system_models.AdaptiveLinearController(system, delta_mean_int_rnd, True)
    optimal_avg_loss = np.trace(sp.linalg.solve_discrete_are(system.A, system.B, system.q_system, system.r_system)
                                @system.noise_cov)

    print('test started')
    print('Avg_loss:' + str(optimal_avg_loss))
    number_of_training_runs = 10

    loss_data_random_agent = np.zeros((number_of_training_runs, learning_hp['epochs']))

    for i in range(number_of_training_runs):
        sub_path = os.path.join(path, 'run' + str(i) + '/')
        os.makedirs(sub_path)

        # Generate system noise and channel transitions
        system_noise = np.random.multivariate_normal(np.zeros((system.dim,)), system.noise_cov,
                                                     learning_hp['epochs'] * learning_hp['horizon'])
        channel_init = np.random.sample((resources, learning_hp['epochs']))
        channel_transitions = np.random.sample((2 * resources, learning_hp['epochs'] * learning_hp['horizon']))

        # Reset channels
        for channel in channels:
            channel.state = None
        # Random intelligent agent
        loss, agent \
            = random_intelligent_agent.random_intelligent_agent(network_markov_chain_hp,
                                                                system, adaptive_controller, channels,
                                                                int_rnd_policy, learning_hp['horizon'],
                                                                learning_hp['epochs'], system_noise,
                                                                channel_init, channel_transitions)
        # Reset channels
        for channel in channels:
            channel.state = None

        loss_data_random_agent[i, :] = np.array(loss)
        np.savetxt(os.path.join(sub_path, 'loss_data_random_agent.csv'), np.array(loss), delimiter=",")

        plt.figure()
        plt.plot(loss_data_random_agent[i, :], label='Random agent')
        plt.axhline(y=optimal_avg_loss, color='r')
        #plt.ylim(0, 20000)
        plt.legend()
        plt.xlabel('Epochs')
        plt.ylabel('Empirical average loss')
        plt.savefig(os.path.join(sub_path, 'loss_fig.png'), bbox_inches='tight')

    loss_mean_random_agent = np.mean(loss_data_random_agent, axis=0)

    fig1, ax1 = plt.subplots()
    ax1.plot(loss_mean_random_agent, label='Random agent')
    ax1.axhline(y=optimal_avg_loss, color='r', label='Optimal LQR')
    ax1.legend()
    #ax1.set_ylim(0, 20000)
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Empirical average loss')
    plt.savefig(os.path.join(path, 'avg_loss.png'), bbox_inches='tight')
    # fig2, ax2 = plt.subplots(figsize=(10, learning_hp['epochs']/4))
    # fig2.patch.set_visible(False)
    # ax2.axis('off')
    # ax2.axis('tight')
    # names = []
    # for k in range(system_hp['number_of_subsystems']):
    #     for n in range(resource_classes):
    #         names.append('R' + str(n) + ' to S' + str(k))
    # names.append('Sum')
    # col = np.array([np.sum(avg_act, axis=0)])
    # avg_act = np.concatenate((avg_act, col), axis=0)
    # df = pd.DataFrame(avg_act.transpose(), columns=names)
    # ax2.table(cellText=df.values, colLabels=df.columns, loc='center')

    for a in system.subA:
        print(np.linalg.eigvals(a))


if __name__ == "__main__":
    main()
