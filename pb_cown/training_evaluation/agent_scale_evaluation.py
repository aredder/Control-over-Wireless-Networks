import numpy as np
import scipy as sp
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import matplotlib
import system_models
import dqn_iterative_agent
from tqdm import tqdm

matplotlib.use('Qt5Agg')


def main():
    system_number = [4, 8, 12]
    model_number = [3, 1, 1]
    hidden_layers = [512, 1024, 1536]
    it_agent = []
    rnd_agent = []
    optimal_loss = []

    for idx, x in enumerate(system_number):
        system_hp = {'system_dimension': 2*x,
                     'number_of_subsystems': x,
                     'resource_quantities': [int(x/4), int(2*x/4)],
                     'GE_0': [0.1, 0.9, 0.95, 0.75],
                     'GE_1': [0.2, 0.8, 0.6, 0.4],
                     'dependency': True,
                     'ratio_of_stable_subsystems': 0.5,
                     'init_cov_scale': 1,
                     'noise_cov_scale': 0.1,
                     'state_cost_scale': 10,
                     'control_cost_scale': 5
                     }

        # System
        x0_mean = np.zeros((system_hp['system_dimension'],))
        x0_cov = np.eye(system_hp['system_dimension'])
        system_noise_mean = np.zeros((system_hp['system_dimension'],))
        system_noise_cov = np.eye(system_hp['system_dimension']) * 0.1
        resources = sum(system_hp['resource_quantities'])
        system = system_models.LinearSystem(dimension=system_hp['system_dimension'],
                                            subsystems=system_hp['number_of_subsystems'],
                                            init_cov=np.eye(system_hp['system_dimension']) *
                                                system_hp['init_cov_scale'],
                                            noise_cov=np.eye(system_hp['system_dimension']) *
                                                system_hp['noise_cov_scale'],
                                            dependency=system_hp['dependency'],
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

        quality = [good_channel.error_stationary, bad_channel.error_stationary]

        delta_0 = sum([a * (1 - b) for a, b in zip(system_hp['resource_quantities'], quality)]) / system_hp[
            'number_of_subsystems']
        print(1 - 1 / np.max(np.abs(np.linalg.eigvals(system.A)) ** 2))
        print(delta_0)
        if 1 - 1 / np.max(np.abs(np.linalg.eigvals(system.A)) ** 2) < delta_0:
            print('System is unstable under random policy')

        # Compute initial controller based on success rate of intelligent random policy
        action_size = system.subsystems
        lambda_max = np.array([np.max(np.abs(np.linalg.eigvals(a))) for a in system.subA])
        int_rnd_policy = np.array([x / sum(lambda_max) for x in lambda_max])

        # for channel in channels:
        #     p_success += (1 - channel.error_stationary)
        # delta_success = np.zeros((action_size,))
        # for m in range(resources):
        #     comb = combinations(np.arange(resources), m)
        #     np.sum([np.prod([1-channels[channel_ID].error_stationary for channel_ID in x]) for x in comb])
        # p_success = p_success*np.array([x*(1-x)**(resources-1) for x in int_rnd_policy])

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

        optimal_loss.append(np.trace(sp.linalg.solve_discrete_are(system.A, system.B, system.q_system, system.r_system)
                                     @system_noise_cov))
        # Monte carlo horizon
        learning_hp = {'epochs': 200,
                       'horizon': 500}

        # Iterative agent
        resources = sum(system_hp['resource_quantities'])
        state_size = system.dim + resources * system.subsystems
        action_size = system.subsystems
        agent = dqn_iterative_agent.DiraAgent(state_size, action_size, resources)
        agent.set_model(1, hidden_layers[idx])
        agent.load('models/model' + str(system_number[idx]) + '_' + str(model_number[idx]) + '.h5')
        agent.epsilon = 0
        network = system_models.BaseNetworkGE(channels)

        avg_it_loss = []
        avg_rnd_loss = []
        for epoch in tqdm(range(learning_hp['epochs'])):

            # Generate system noise and channel transitions
            system_noise = np.random.multivariate_normal(system_noise_mean, system_noise_cov,
                                                         learning_hp['epochs']*learning_hp['horizon'])
            channel_init = np.random.sample((resources, learning_hp['epochs']))
            channel_transitions = np.random.sample((2*resources, learning_hp['epochs']*learning_hp['horizon']))

            cum_loss = 0
            noise_idx = 0
            system.reset_state()
            system_state = system.state
            for c, channel in enumerate(network.channels):
                channel.initialize(channel_init[c, 0])

            for time in range(learning_hp['horizon']):
                # Scheduling action
                schedule, _, _ = agent.act(system_state, rnd_policy=None)
                transmission_outcomes = network.output(schedule,
                                                       np.reshape(channel_transitions[:, noise_idx], (resources, 2)))

                # Apply control
                inputs = controller @ system_state
                # For now single input system
                active_inputs = np.array([a * b for a, b in zip(transmission_outcomes, inputs)])
                next_system_state = system.state_update(active_inputs, np.array(system_noise[noise_idx, :]))
                # Compute loss
                system_loss = \
                    system_state.transpose() @ system.q_system @ system_state \
                    + active_inputs.transpose() @ system.r_system @ active_inputs

                cum_loss += system_loss/learning_hp['horizon']
                system_state = next_system_state
                noise_idx += 1
            if cum_loss < 10000:
                avg_it_loss.append(cum_loss)
                expt = True
            else:
                expt = False
                print('Error')

            cum_loss = 0
            noise_idx = 0
            system.reset_state()
            system_state = system.state
            for c, channel in enumerate(network.channels):
                channel.initialize(channel_init[c, 0])

            for time in range(learning_hp['horizon']):
                # Action
                schedule = np.zeros((resources, action_size))
                for j in range(resources):
                    action = np.random.choice(action_size, 1, p=int_rnd_policy)
                    schedule[j, action] = 1

                transmission_outcomes = network.output(schedule, np.reshape(channel_transitions[:, noise_idx],
                                                                            (resources, 2)))

                inputs = controller @ system_state
                # for know single input system
                active_inputs = np.array([a * b for a, b in zip(transmission_outcomes, inputs)])

                next_system_state = system.state_update(active_inputs, np.array(system_noise[noise_idx, :]))
                # Compute loss
                system_loss \
                    = system_state.transpose() @ system.q_system @ system_state \
                    + active_inputs.transpose() @ system.r_system @ active_inputs

                system_state = next_system_state
                noise_idx += 1
                # Cumulative loss
                cum_loss += system_loss / learning_hp['horizon']

            if expt:
                avg_rnd_loss.append(cum_loss)

        it_agent.append(sum(avg_it_loss)/len(avg_it_loss))
        rnd_agent.append(sum(avg_rnd_loss)/len(avg_rnd_loss))

        print(str(idx) + ' done')
        print('_______')

    data = np.vstack((np.array(optimal_loss), np.array(it_agent), np.array(rnd_agent)))


    data[2, :] = data[2, :] - data[1, :]
    data[1, :] = data[1, :] - data[0, :]

    columns = ('N = 4, M = 3', 'N = 8, M = 6', 'N = 12, M = 9')
    rows = ['Random Agent', 'Iterative Agent', 'Optimal loss']

    values = np.arange(0, 900, 100)
    value_increment = 1

    # Get some pastel shades for the colors
    #colors = plt.cm.Set1(np.linspace(0, 0.5, len(rows)))
    colors = ['indianred', 'royalblue', 'sandybrown']
    n_rows = len(data)

    index = np.arange(len(columns)) + 0.3
    bar_width = 0.4

    # Initialize the vertical-offset for the stacked bar chart.
    y_offset = np.zeros(len(columns))

    # Plot bars and create text labels for the table
    cell_text = []
    for row in range(n_rows):
        plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
        y_offset = y_offset + data[row]
        cell_text.append(['%.2f' % x for x in y_offset])
    # Reverse colors and text labels to display the last value at the top.
    colors = colors[::-1]
    cell_text.reverse()

    # Add a table at the bottom of the axes
    the_table = plt.table(cellText=cell_text,
                          rowLabels=rows,
                          rowColours=colors,
                          colLabels=columns,
                          loc='bottom',
                          cellLoc='center')
    the_table.scale(1, 1.5)

    # Adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)
    plt.ylabel('Average control loss per stage')
    plt.yticks(values * value_increment, ['%d' % val for val in values])
    plt.xticks([])
    plt.show()
    plt.savefig("scale_plot.pdf", bbox_inches='tight')


if __name__ == "__main__":
    main()
