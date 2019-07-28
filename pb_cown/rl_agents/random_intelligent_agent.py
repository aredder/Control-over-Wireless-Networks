import numpy as np

from system_env import system_models


def random_intelligent_agent(network_hp, system, controller, channels, policy, horizon, epochs, system_noise,
                             channel_init, channel_transitions):

    resources = sum(network_hp['model_quantities'])
    action_size = system.subsystems
    network = system_models.BaseNetworkGE(channels)

    avg_loss = []
    noise_idx = 0
    for e in range(epochs):
        for c, channel in enumerate(network.channels):
            channel.initialize(channel_init[c, e])
        system.reset_state()
        system_state = system.state
        cum_loss = 0

        for time in range(horizon):
            # Action
            schedule = np.zeros((resources, action_size))
            for j in range(resources):
                action = np.random.choice(action_size, 1, p=policy)
                schedule[j, action] = 1

            transmission_outcomes = network.output(schedule,
                                                   np.reshape(channel_transitions[:, noise_idx], (resources, 2)))

            # Apply control
            actor_decision = [1 if x != 0 else 0 for x in np.sum(schedule, axis=0)]
            inputs = controller.controller_one_step(actor_decision) @ system_state
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
            cum_loss += system_loss
        avg_loss.append(cum_loss / horizon)
    return avg_loss, None
