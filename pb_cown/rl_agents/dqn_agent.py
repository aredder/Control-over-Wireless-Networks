import numpy as np
import random
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import system_models


class DqnAgent:
    def __init__(self, state_size, action_size):
        """
        Agent class for iterative resource scheduling.

        Parameters
        ----------
        state_size
        action_size
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
        self.target_update = None                                   # Number of updates until target update
        self.state_size = state_size                                # State space dimension
        self.action_size = action_size                              # Action space dimension

    def recompile(self):
        self.model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, decay=self.learning_rate_decay))

    def set_hyper(self, rep_size, gamma, eps, eps_min, eps_decay, learning_rate, learning_rate_decay, target_update):
        self.memory = deque(maxlen=rep_size)
        self.gamma = gamma
        self.epsilon = eps
        self.epsilon_decay = eps_decay
        self.epsilon_min = eps_min
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.target_update = target_update

    def set_model(self, hidden_layers, hidden_layer_size):
        model = Sequential()
        model.add(Dense(hidden_layer_size, input_dim=self.state_size, activation='relu'))
        for _ in range(hidden_layers-1):
            model.add(Dense(hidden_layer_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate, decay=self.learning_rate_decay))
        self.model = model
        self.target_model = model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        explore = np.random.rand() <= self.epsilon
        if explore:
            action = np.random.choice(self.action_size)
        else:
            action = np.argmax(self.model.predict(np.reshape(state, [1, self.state_size]))[0])
        return action

    def replay(self, batch_size, current_sample):
        mini_batch = random.sample(self.memory, batch_size)
        if current_sample is not None:
            mini_batch.append(current_sample)
        states, targets_f = [], []
        for agent_state, action, reward, next_agent_state, done in mini_batch:

            agent_state = np.reshape(agent_state, [1, self.state_size])
            # Set targets
            target = reward
            if not done:
                target = reward + self.gamma*np.amax(self.target_model.predict(np.reshape(next_agent_state,
                                                                               [1, self.state_size])))
            target_f = self.model.predict(agent_state)
            target_f[0][action] = target
            # Filter targets of actions that were not used.
            states.append(agent_state[0])
            targets_f.append(target_f[0])

        history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def train_agent(learning_hp, system_hp, system, controller, channels, system_noise, channel_init, channel_transitions):

    resources = sum(system_hp['resource_quantities'])
    epochs = learning_hp['epochs']
    state_size = system.dim
    action_size = system.subsystems**resources
    agent = DqnAgent(state_size, action_size)
    agent.set_hyper(learning_hp['memory'], learning_hp['gamma'], learning_hp['epsilon'], learning_hp['epsilon_min'],
                    learning_hp['epsilon_decay'], learning_hp['learning_rate'], learning_hp['learning_rate_decay'],
                    learning_hp['target_update'])
    agent.set_model(learning_hp['hidden_layers'], learning_hp['neurons_per_layer'])

    # Set network
    network = system_models.BaseNetworkGE(channels)
    # for 3 resources and 4 systems
    all_actions = [(x, y, z) for x in range(4) for y in range(4) for z in range(4)]
    scheduling_representation = np.zeros((resources, system.subsystems))
    avg_loss = []
    train_idx = 0
    train_loss = 0
    noise_idx = 0

    for epoch in range(epochs):
        # Initialize channels
        for c, channel in enumerate(network.channels):
            channel.initialize(channel_init[c, epoch])
        # Learning rate schedule
        # drop = 1/3
        # epochs_drop = 7
        # if epoch % epochs_drop == 0:
        #     new_lr = learning_hp['learning_rate'] * np.power(drop, np.floor((1 + epoch) / epochs_drop))
        #     tf.keras.backend.set_value(agent.model.optimizer.lr, new_lr)
        #     print("Learning rate: {}".format(new_lr))

        done = False
        system.reset_state()
        system_state = system.state
        cum_loss = 0
        for time in range(learning_hp['horizon']):

            # Scheduling action
            action = agent.act(system_state)
            schedule = scheduling_representation.copy()
            for j in range(resources):
                schedule[j, all_actions[action][j]] = 1

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

            # Keep track of final states
            if time == learning_hp['horizon'] - 1:
                done = True

            agent.remember(system_state, action, -system_loss, next_system_state, done)

            # Set current_sample for combined_Q option
            if len(agent.memory) >= learning_hp['batch_size']:
                current_sample = (system_state, action, -system_loss, next_system_state, done)
                train_loss += agent.replay(learning_hp['batch_size'], current_sample)
                train_idx += 1
                if np.isnan(train_loss):
                    cum_loss += system_loss
                    avg_loss.append(cum_loss / time)
                    for _ in range(epochs-epoch-1):
                        avg_loss.append(np.nan)
                    return avg_loss, None, None

                # Logging every 100 time-steps
                if time % 100 == 0:
                    print("episode: {}/{}, time: {}, loss: {:.4f}"
                          .format(epoch + 1, epochs, time, train_loss / train_idx))
                    train_loss = 0
                    train_idx = 0

            # Update target network
            if time % learning_hp['target_update'] == 0:
                agent.target_model.set_weights(agent.model.get_weights())
            # Cumulate loss
            cum_loss += system_loss

            system_state = next_system_state
            noise_idx += 1

        avg_loss.append(cum_loss / learning_hp['horizon'])
    return avg_loss, None, agent
