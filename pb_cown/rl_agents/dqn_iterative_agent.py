import random
from collections import deque

import numpy as np
import tensorflow as tf

from system_env import system_models


config = tf.ConfigProto(  # intra_op_parallelism_threads=1, inter_op_parallelism_threads=1,
        allow_soft_placement=False,
        device_count={'GPU': 1},
        log_device_placement=False)
session = tf.Session(config=config)
K = tf.keras
K.backend.set_session(session)


class DiraAgent:
    def __init__(self, state_size, action_size, resources):
        """
        Agent class for iterative resource scheduling.

        Parameters
        ----------
        state_size
        action_size
        resources
        """
        self.memory = None                                          # Replay memory
        self.gamma = None                                           # Discount factor
        self.epsilon = None                                         # Initial exploration rate
        self.epsilon_min = None                                     # Exploration clip
        self.epsilon_decay = None                                   # Exploration decay
        self.learning_rate = np.exp(-4.5)                           # Learning rate
        self.learning_rate_decay = 0.001                            # Learning rate decay
        self.model = None                                           # Main Q-Network
        self.target_model = None                                    # Target Q-Network
        self.tau_target = None                                      # Stationarity of targets update
        self.sl_epochs = None
        self.state_size = state_size                                # State space dimension
        self.action_size = action_size                              # Action space dimension
        self.resources = resources                                  # Number of resources to be scheduled
        self.initial_representation = np.zeros((self.resources, self.action_size))      # Scheduling representation

    def recompile(self):
        self.model.compile(loss='mse', optimizer=K.optimizers.Adam(lr=self.learning_rate,
                                                                   decay=self.learning_rate_decay))

    def set_hyper(self, rep_size, gamma, eps, eps_min, eps_decay, learning_rate, learning_rate_decay, tau_target,
                  sl_epochs):
        self.memory = deque(maxlen=rep_size)
        self.gamma = gamma
        self.epsilon = eps
        self.epsilon_decay = eps_decay
        self.epsilon_min = eps_min
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.tau_target = tau_target
        self.sl_epochs = sl_epochs

    def set_model(self, hidden_layers, hidden_layer_size):
        model = K.models.Sequential()
        model.add(K.layers.Dense(hidden_layer_size, input_dim=self.state_size, activation='relu'))
        for _ in range(hidden_layers-1):
            model.add(K.layers.Dense(hidden_layer_size, activation='relu'))
        model.add(K.layers.Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=K.optimizers.Adam(lr=self.learning_rate, decay=self.learning_rate_decay))
        self.model = model
        self.target_model = model

    def remember(self, state, action, reward, next_state, done, artificial):
        self.memory.append((state, action, reward, next_state, done, artificial))

    def act(self, state, rnd_policy):
        # Initialize schedule and agent_state
        schedule = self.initial_representation.copy()
        agent_state = np.hstack((state, schedule.flatten()))
        state_history, action_history, reward_history = [agent_state], [], []

        # Iterate over resources
        for j in range(self.resources):
            explore = np.random.rand() <= self.epsilon
            if explore:
                action = np.random.choice(self.action_size, 1, p=rnd_policy)
                schedule[j, action] = 1
            else:
                applied_state = np.reshape(agent_state, [1, self.state_size])
                action = np.argmax(self.model.predict(applied_state)[0])
                schedule[j, action] = 1

            agent_state = np.hstack((state, schedule.flatten()))
            state_history.append(agent_state)
            action_history.append(action)
        return schedule, state_history, action_history

    def predict(self, state):
        schedule = self.initial_representation.copy()
        agent_state = np.hstack((state, schedule.flatten()))
        predictions = None
        for j in range(self.resources):
            applied_state = np.reshape(agent_state, [1, self.state_size])
            predictions = self.target_model.predict(applied_state)[0]
            sub_action = np.argmax(predictions)
            schedule[j, sub_action] = 1
            agent_state = np.hstack((state, schedule.flatten()))
        return np.amax(predictions)

    def predict_simple(self, state):
        schedule = self.initial_representation.copy()
        agent_state = np.hstack((state, schedule.flatten()))
        applied_state = np.reshape(agent_state, [1, self.state_size])
        return np.amax(self.target_model.predict(applied_state)[0])

    def replay(self, batch_size, terminal_q, current_sample, epoch):
        mini_batch = random.sample(self.memory, batch_size)
        if current_sample is not None:
            mini_batch.append(current_sample)
        states, targets_f = [], []
        for agent_state, action, reward, next_agent_state, done, artificial in mini_batch:

            system_size = self.state_size - self.resources*self.action_size
            next_system_state = next_agent_state[0:system_size]

            agent_state = np.reshape(agent_state, [1, self.state_size])
            # Set targets
            target = reward
            if not done:
                if terminal_q:
                    target = reward + self.gamma * self.predict_simple(next_system_state)
                else:
                    if artificial:
                        target = reward + np.amax(self.target_model.predict(np.reshape(next_agent_state,
                                                                                       [1, self.state_size])))
                    else:
                        target = reward + self.gamma*np.amax(self.target_model.predict(np.reshape(next_agent_state,
                                                                                       [1, self.state_size])))
            target_f = self.model.predict(agent_state)
            target_f[0][action] = target
            # Filter targets of actions that were not used.
            states.append(agent_state[0])
            targets_f.append(target_f[0])

        if epoch >= self.sl_epochs:
            history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
            loss = history.history['loss'][0]
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            return loss
        else:
            history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
            loss = history.history['loss'][0]
            return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def train_iterative_agent(learning_hp, network_hp, system, controller, rnd_policy, channels, system_noise,
                          channel_init, channel_transitions, external_execution, logger):

    resources = sum(network_hp['model_quantities'])
    epochs = learning_hp['epochs']
    state_size = system.dim + resources * system.subsystems
    action_size = system.subsystems
    agent = DiraAgent(state_size, action_size, resources)
    agent.set_hyper(learning_hp['memory'], learning_hp['gamma'], learning_hp['epsilon'], learning_hp['epsilon_min'],
                    learning_hp['epsilon_decay'], learning_hp['learning_rate'], learning_hp['learning_rate_decay'],
                    learning_hp['tau_target'], learning_hp['sl_epochs'])
    agent.set_model(learning_hp['hidden_layers'], learning_hp['neurons_per_layer'])

    # Set network
    network = system_models.BaseNetworkGE(channels)

    avg_loss = []
    train_idx = 0
    train_loss = 0
    avg_schedule = agent.initial_representation.copy()
    noise_idx = 0

    epoch_accum_success = np.zeros((system.subsystems, epochs))
    accum_success = np.zeros((system.subsystems,))

    for epoch in range(epochs):
        # Initialize channels
        for c, channel in enumerate(network.channels):
            channel.initialize(channel_init[c, epoch])
        # Learning rate schedule
        drop = 1/3
        epochs_drop = 7
        if epoch % epochs_drop == 0:
            new_lr = learning_hp['learning_rate'] * np.power(drop, np.floor((1 + epoch) / epochs_drop))
            tf.keras.backend.set_value(agent.model.optimizer.lr, new_lr)
            if not external_execution:
                print("Learning rate: {}".format(new_lr))

        done = False
        system.reset_state()
        system_state = system.state
        cum_loss = 0

        for time in range(learning_hp['horizon']):

            # Scheduling action
            schedule, state_history, action_history = agent.act(system_state, rnd_policy)
            transmission_outcomes = network.output(schedule,
                                                   np.reshape(channel_transitions[:, noise_idx], (resources, 2)))
            accum_success += transmission_outcomes

            # Log final average schedule:
            if epoch == epochs - 1:
                avg_schedule += schedule

            # Apply control
            actor_decision = [1 if x != 0 else 0 for x in np.sum(schedule, axis=0)]
            inputs = controller.controller_one_step(actor_decision) @ system_state
            # For now: implemented for single input systems
            active_inputs = np.array([a * b for a, b in zip(transmission_outcomes, inputs)])
            next_system_state = system.state_update(active_inputs, np.array(system_noise[noise_idx, :]))
            # Compute loss
            system_loss = \
                system_state.transpose() @ system.q_system @ system_state \
                + active_inputs.transpose() @ system.r_system @ active_inputs

            # Keep track of final states
            if time == learning_hp['horizon'] - 1:
                done = True

            schedule = agent.initial_representation.copy()
            next_agent_state = np.hstack((next_system_state, schedule.flatten()))

            # Train agent (Start training when memory size = number of system time steps)
            # Set current_sample for combined_Q option
            if len(agent.memory) >= learning_hp['batch_size'] * resources:
                for i in range(resources):
                    if learning_hp['terminal_Q']:
                        current_sample = \
                            (state_history[i], action_history[i], -system_loss, next_agent_state, done, False)
                    else:
                        if i == resources - 1:
                            current_sample = \
                                (state_history[i], action_history[i], -system_loss, next_agent_state, done, False)
                        else:
                            current_sample = \
                                (state_history[i], action_history[i], 0, state_history[i + 1], done, True)

                    train_loss += agent.replay(learning_hp['batch_size'], learning_hp['terminal_Q'], current_sample,
                                               epoch)
                    train_idx += 1
                    if np.isnan(train_loss):
                        cum_loss += system_loss
                        avg_loss.append(cum_loss / time)
                        for _ in range(epochs - epoch - 1):
                            avg_loss.append(np.nan)
                        return avg_loss, None, None
                # Logging every 100 time-steps
                if time % 100 == 0:
                    if not external_execution:
                        print("episode: {}/{}, time: {}, loss: {:.4f}"
                              .format(epoch + 1, epochs, time, train_loss / train_idx))
                    logger.debug("episode: {}/{}, time: {}, loss: {:.4f}"
                                 .format(epoch + 1, epochs, time, train_loss / train_idx))
                    train_loss = 0
                    train_idx = 0

            # Append history
            for i in range(resources):
                if learning_hp['terminal_Q']:
                    agent.remember(state_history[i], action_history[i], -system_loss, next_agent_state, done, False)
                else:
                    if i == resources - 1:
                        agent.remember(state_history[i], action_history[i], -system_loss, next_agent_state, done, False)
                    else:
                        agent.remember(state_history[i], action_history[i], 0, state_history[i+1], done, True)

            # Update target network
            new_weights = [learning_hp['tau_target']*x + (1-learning_hp['tau_target'])*y for x, y in
                           zip(agent.model.get_weights(), agent.target_model.get_weights())]
            agent.target_model.set_weights(new_weights)

            # Accumulated loss
            cum_loss += system_loss
            system_state = next_system_state
            noise_idx += 1

        # Adaptive controller update
        epoch_accum_success[:, epoch] = accum_success
        if epoch >= learning_hp['sl_epochs'] + 5:
            new_success_rate = (accum_success - epoch_accum_success[:, epoch-5]) / (5*learning_hp['horizon'])
            controller.controller_update(new_success_rate)
            logger.debug('Success_rate: ' + str(new_success_rate))
            if not external_execution:
                print(new_success_rate)

        # Append avg epoch loss
        avg_loss.append(cum_loss / learning_hp['horizon'])

    avg_schedule = avg_schedule / learning_hp['horizon']
    return avg_loss, avg_schedule, agent
