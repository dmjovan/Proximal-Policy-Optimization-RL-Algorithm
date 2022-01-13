import numpy as np
from numpy.lib.npyio import load
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import os
import shutil
import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal.filter_design import normalize

available_envs = {
                  'CartPole-v0': {'type': 'discrete', 'reward_info': 'Reward is 1 for every step taken, including the termination step'},
                  'MountainCar-v0': {'type': 'discrete', 'reward_info': 'Reward is -1.0 if not terminal state, 0.0 if agent has reached the flag'},
                  'Acrobot-v1': {'type': 'discrete', 'reward_info': 'Reward is -1.0 if not terminal state, 0.0 if terminal'},
                  'MountainCarContinuous-v0': {'type': 'continuous', 'reward_info': 'Reward is 100 if agent reached the flag, reward is decreased based on amount of energy consumed each step'},
                  'Pendulum-v0': {'type': 'continuous', 'reward_info': 'None'}
                 }

class PPOAgent(object):

    """ Implementation of Proximal Policy Optimization agent from Machine Learning field of Reinforcement Learning """    

    def __init__(self, args: object, _env_name: str=None, load_model: bool=False) -> None:

        """ 
            Constructor of the class.

            :params:
                - args: ArgumentParser object 
                - _env_name: name of default Environment or Environment for for-loop training inside IDE
                - load_model: indicator for loading models if exist

            :return:
                - None
        """

        np.random.seed(1)
        tf.random.set_seed(1)

        # defining hyper-parameters
        self.env_name = _env_name if args.env_name is None else args.env_name
        self.num_episodes = 100 if args.num_episodes is None else args.num_episodes
        self.max_iter = 5000 if args.max_iter is None else args.mx_iter
        self.eval_episodes = 20 if args.eval_episodes is None else args.eval_episodes
        self.gamma = 0.99 if args.gamma is None else args.gamma
        self.clip_ratio = 0.2 if args.clip_ratio is None else args.clip_ratio
        self.actor_lr = 3e-4 if args.actor_lr is None else args.actor_lr
        self.critic_lr = 1e-3 if args.critic_lr is None else args.critic_lr
        self.train_actor_iter = 60 if args.train_actor_iter is None else args.train_actor_iter
        self.train_critic_iter = 60 if args.train_critic_iter is None else args.train_critic_iter
        self.lambda_ = 0.97 if args.lambda_ is None else args.lambda_
        self.target_kl = 0.01 if args.target_kl is None else args.target_kl
        self.hidden_units = [64, 64] if args.hidden_units is None else args.hidden_units
        self.buffer_size = self.max_iter if args.buffer_size is None else args.buffer_size

        # checking if envirnoment is supported
        assert self.env_name in available_envs.keys(), 'Provided environment is not supported!'
    
        self.env_type = available_envs[self.env_name]['type']

        # creating OpenAi Gym Environment
        self.env = gym.make(self.env_name)

        # getting dimensions of state and action spaces
        self.state_space_dims = self.env.observation_space.shape[0]
        
        if self.env_type == 'discrete':
            self.action_space_dims = self.env.action_space.n

        elif self.env_type == 'continuous':
            pass
            # TODO add here

        # creating paths and folders for storing results
        self.models_path = self.env_name + '\\models'
        self.results_path = self.env_name + '\\results'

        if os.path.exists(self.env_name) and not load_model:
            shutil.rmtree(self.env_name) 
            os.makedirs(self.models_path)
            os.makedirs(self.results_path)

        # initialization of all buffers for storing trajectories
        self.state_buffer = np.zeros((self.buffer_size, self.state_space_dims), dtype=np.float32)
        self.action_buffer = np.zeros(self.buffer_size, dtype=np.int32)
        self.advantage_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.reward_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.return_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.value_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(self.buffer_size, dtype=np.float32)

        # initialization of volatle indices for tracking buffer
        self.buffer_counter, self.trajectory_start_index = 0, 0

        # creating Actor and Critic models
        self.create_actor(load_model)
        self.create_critic(load_model)

        # initialization of Actor and Critic optimizers
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=self.critic_lr)
        
        print(self.__repr__())
        
        # printing Actor model summary
        print('--------------------------------------------------------------------')
        self.actor.summary()
        print('--------------------------------------------------------------------')

        # printing Critic model summary
        print('--------------------------------------------------------------------')
        self.critic.summary()
        print('--------------------------------------------------------------------')


    def __repr__(self) -> str:

        """
            Function for representation of class.

            :params:
                - None

            :return:
                - None
        """

        rep = '--------------------------------------------------------------------\n'
        rep += '---------------- Proximal Policy Optimization Agent ----------------\n'
        rep += '********************\n'
        rep += '********************\n'
        rep += f'OpenAI Gym Environment used: {self.env_name}\n'
        rep += f'Environment state space size: {self.state_space_dims}\n'
        rep += f'Environment action space is: {self.env_type}\n'

        if self.env_type == 'discrete':
            rep += f'Environment number of actions is: {self.action_space_dims}\n'

        elif self.env_type == 'continuous':
            rep += ''
            # TODO add here 

        rep += f'Environment reward definition: {available_envs[self.env_name]["reward_info"]}\n'
        rep += '********************\n'
        rep += '********************\n'
        rep += f'Number of training episodes: {self.num_episodes}\n'
        rep += f'Maximum iterations over one episode: {self.max_iter}\n'
        rep += f'Discount factor - gamma: {self.gamma}\n'
        rep += f'Clip ratio: {self.clip_ratio}\n'
        rep += f'Actor model learning rate: {self.actor_lr}\n'
        rep += f'Critic model learning rate: {self.critic_lr}\n'
        rep += f'Number of iterations for updating Actor model: {self.train_actor_iter}\n'
        rep += f'Number of iterations for updating Critic model: {self.train_critic_iter}\n'
        rep += f'Lambda factor: {self.lambda_}\n'
        rep += f'Target KL value: {self.target_kl}\n'
        rep += f'Number of units for hidden layers of Actor/Critic models: {self.hidden_units}\n'
        rep += f'Buffer size: {self.buffer_size}\n'
        rep += '--------------------------------------------------------------------'

        return rep


    def create_actor(self, load_model: bool=False) -> None:

        """
            Function for creating Actor model.

            :params:
                - load_model: indicator for loading Actor model

            :return:
                - None
        """

        # creating linear FF neural network with only Fully-Connected layers
        state_input = keras.Input(shape=(self.state_space_dims,), dtype=tf.float32)

        x = state_input
        for num_units in self.hidden_units:
            x = layers.Dense(units=num_units, activation=tf.tanh)(x)

        logits_output = layers.Dense(units=self.action_space_dims, activation=None)(x)
        
        self.actor = keras.Model(inputs=state_input, outputs=logits_output, name='Actor_Model')

        if load_model and os.path.exists(self.models_path + '\\actor.h5'):
            self.actor.load_weights(self.models_path + '\\actor.h5')


    def create_critic(self, load_model: bool=False) -> None:

        """
            Function for creating Critic model.

            :params:
                - load_model: indicator for loading Critic model

            :return:
                - None
        """

        # creating linear FF neural network with only Fully-Connected layers
        state_input = keras.Input(shape=(self.state_space_dims,), dtype=tf.float32)

        x = state_input
        for num_units in self.hidden_units:
            x = layers.Dense(units=num_units, activation=tf.tanh)(x)
            
        value_output = layers.Dense(units=1, activation=None)(x)
        value = tf.squeeze(value_output, axis=1)

        self.critic = keras.Model(inputs=state_input, outputs=value, name='Critic_Model')

        if load_model and os.path.exists(self.models_path + '\\critic.h5'):
            self.critic.load_weights(self.models_path + '\\critic.h5')


    @tf.function
    def logprobabilities(self, logits, actions) -> float:

        """
            Function for computing the log-probabilities of taking actions by using the logits - the output of the Actor model.

            :params:
                - logits: logits from Actor model
                - actions: actions from buffer

            :return:
                - logprobability: log probabilities for actions
        """

        logprobabilities_all = tf.nn.log_softmax(logits)
        logprobability = tf.reduce_sum(tf.one_hot(actions, self.action_space_dims) * logprobabilities_all, axis=1)

        return logprobability


    @tf.function
    def sample_action(self, state) -> tuple:

        """ 
            Sample action from Actor model for provided state.

            :params:
                - state: current state

            :return:
                - logits: log softmax probabilities from actor
                - action: random sampled actions, sampled regarding the log softmax probabilities from actor (logits)
        """

        logits = self.actor(state)
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)

        return logits, action


    @tf.function
    def update_actor(self, normalize: bool=False) -> float:

        """ 
            Function for updating Actor model, acording to buffer values, maxizing the PPO-Clip objective.

            :params:
                - normalize: inicator for normalizing advantage buffer

            :return:
                - kl: KL value 
        """

        if normalize:
            # getting statistics for advantage buffer
            advantage_mean, advantage_std = np.mean(self.advantage_buffer), np.std(self.advantage_buffer)

            # normalizing advantage_buffer
            self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std

        with tf.GradientTape() as tape:
            ratio = tf.exp(self.logprobabilities(self.actor(self.state_buffer), self.action_buffer) - self.logprobability_buffer)

            min_advantage = tf.where(self.advantage_buffer > 0, (1 + self.clip_ratio) * self.advantage_buffer, (1 - self.clip_ratio) * self.advantage_buffer)

            policy_loss = -tf.reduce_mean(tf.minimum(ratio * self.advantage_buffer, min_advantage))

        policy_grads = tape.gradient(policy_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(policy_grads, self.actor.trainable_variables))

        kl = tf.reduce_mean(self.logprobability_buffer - self.logprobabilities(self.actor(self.state_buffer), self.action_buffer))
        kl = tf.reduce_sum(kl)

        return kl


    @tf.function
    def update_critic(self):

        """ 
            Function for updating Critic model, acording to buffer values, by regression on mean-squared error.

            :params:
                - None

            :return:
                - None 
        """

        with tf.GradientTape() as tape: 
            value_loss = tf.reduce_mean((self.return_buffer - self.critic(self.state_buffer)) ** 2)
        
        value_grads = tape.gradient(value_loss, self.critic.trainable_variables)

        self.critic_optimizer.apply_gradients(zip(value_grads, self.critic.trainable_variables))


    def store(self, state, action, reward, value, logprobability) -> None:

        """ 
            Function for storing one agent-environment interaction data in buffer.

            :params:
                - state: current agent's state
                - action: sampled action in state
                - reward: env reward for that state
                - value: value from value function (Critic model) for action
                - logprobability: log probability for action

            :return:
                - None
        """

        # storing one agent-env interaction
        self.state_buffer[self.buffer_counter] = state
        self.action_buffer[self.buffer_counter] = action
        self.reward_buffer[self.buffer_counter] = reward
        self.value_buffer[self.buffer_counter] = value
        self.logprobability_buffer[self.buffer_counter] = logprobability

        # incrementing memory counter
        self.buffer_counter += 1


    def discounted_cumulative_sums(self, x: np.ndarray, discount: float) -> float:

        """ 
            Function for computing discounted cumulative sums over provided vector and provided discount factor.

            :params:
                - x: provide vector
                - discount: discount factor

            :return:
                - calculated sums

        """

        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


    def finish_trajectory(self, last_value: float=0.0) -> None:

        """ 
            Function for finishing trajectories in terminal state or when max iteration
            exceeded, by computing advantage estimates and rewards-to-go, using current buffers.

            :params:
                - last_value: last value from training

            :return:
                - None

        """

        path_slice = slice(self.trajectory_start_index, self.buffer_counter)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)

        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]

        self.advantage_buffer[path_slice] = self.discounted_cumulative_sums(deltas, self.gamma * self.lambda_)
        self.return_buffer[path_slice] = self.discounted_cumulative_sums(rewards, self.gamma)[:-1]

        self.trajectory_start_index = self.buffer_counter


    def train(self, render: bool=False) -> None:

        """
            Function for training agent over epsodes, storing interactions and updating Actor and Critic models.

            :params:
                - render: indicator for rendering OpenAI Gym Environment

            :return:
                - None:
        """

        self.train_episodic_rewards, self.train_episodic_lengths = [], []

        print('--------------------------------------------------------------------')
        print('-------------------------- TRAINING --------------------------------')

        # iterating over episodes
        for episode in range(self.num_episodes):

            # reseting environment on the start of episode
            state = self.env.reset()

            # initialize episode reward and episode length
            episode_reward, episode_length = 0, 0

            # iterating over the steps for each episode
            for t in range(self.max_iter):

                print(f'Executing Episode {episode+1}/{self.num_episodes} ---> Executing Iteration {t+1}', end='\r')

                # rendering OpenAI Gym Environment
                if render:
                    self.env.render()

                # getting logits and action 
                state = state.reshape(1, -1)
                logits, action = self.sample_action(state)

                # taking step in environment and getting reward and new state
                new_state, reward, done, _ = self.env.step(action[0].numpy())

                episode_reward += reward
                episode_length += 1

                # get the value and log-probability of the action
                value_t = self.critic(state)
                logprobability_t = self.logprobabilities(logits, action)

                # storing in buffer tuple: current_state, action, rewrd, value, log_probability
                self.store(state, action, reward, value_t, logprobability_t)

                # state transition
                state = new_state

                # finish trajectory if reached to a terminal state
                if done or (t == self.max_iter - 1):
                    last_value = 0 if done else self.critic(state.reshape(1, -1))
                    self.finish_trajectory(last_value)
                    break

            self.train_episodic_rewards.append(episode_reward)
            self.train_episodic_lengths.append(episode_length)

            # reseting buffer indices
            self.buffer_counter, self.trajectory_start_index = 0, 0

            # training Actor model
            for i in range(self.train_actor_iter):

                normalize = False
                if i == 0:
                    normalize = True

                # iterative Actor model updating
                kl = self.update_actor(normalize)

                # early stopping if KL value is diverging (1.5 is expermintal value)
                if kl > 1.5 * self.target_kl:
                    break

            # training Critic model
            for _ in range(self.train_critic_iter):

                # iterative Critic model updating
                self.update_critic()

            # print episodic reward and episode duration
            print(f'Episode: {episode + 1}/{self.num_episodes} --> Episodic Reward: {episode_reward} || Episode Duration [in iterations]: {episode_length}/{self.max_iter}')

        # saving model weigths
        self.actor.save_weights(self.models_path + '\\actor.h5')
        self.critic.save_weights(self.models_path + '\\critic.h5')

        # visulazing results
        self.visualize()


    def evalute(self, render: bool=False) -> None:

        """
            Function for evaluating agent.

            :params:
                - render: indicator for rendering OpenAI Gym Environment

            :return:
                - None:
        """

        self.eval_episodic_rewards, self.eval_episodic_lengths = [], []

        print('--------------------------------------------------------------------')
        print('-------------------------- EVALUATION ------------------------------')

        # iterating over episodes
        for episode in range(self.eval_episodes):

            # reseting environment on the start of episode
            state = self.env.reset()

            # initialize episode reward and episode length
            episode_reward, episode_length = 0, 0

            # iterating over the steps for each episode
            for t in range(self.max_iter):

                print(f'Executing Episode {episode+1}/{self.num_episodes} ---> Executing Iteration {t+1}', end='\r')

                # rendering OpenAI Gym Environment
                if render:
                    self.env.render()

                # getting logits and action 
                state = state.reshape(1, -1)
                _, action = self.sample_action(state)

                # taking step in environment and getting reward and new state
                new_state, reward, done, _ = self.env.step(action[0].numpy())

                episode_reward += reward
                episode_length += 1

                # state transition
                state = new_state

                # finish trajectory if reached to a terminal state
                if done:
                    break

            self.eval_episodic_rewards.append(episode_reward)
            self.eval_episodic_lengths.append(episode_length)

            # print episodic reward and episode duration
            print(f'Episode: {episode + 1}/{self.eval_episodes} --> Episodic Reward: {episode_reward} || Episode Duration [in iterations]: {episode_length}/{self.max_iter}')

        # visulazing results
        self.visualize(train=False)

    
    def visualize(self, train: bool=False) -> None:
        
        """
            Function for visualizing agents performance.

            :params:
                - train: indicator for visualizing training (if True) or evaluation (if False) results

            :return:
                - None
        """

        if train:
            rewards = self.train_episodic_rewards
            lenghts = self.train_episodic_lengths
            suptitle = 'TRAINING PROCESS'
            filename = '\\training.png'

        else:
            rewards = self.eval_episodic_rewards
            lenghts = self.eval_episodic_lengths
            suptitle = 'EVALUATION PROCESS'
            filename = '\\evaluation.png'

        fig, axes = plt.subplots(ncols=2, figsize=(20,14))
        ax = axes.ravel()

        ax[0].plot(np.arange(1, len(rewards)+1), rewards)
        ax[0].set_title('Total episodic reward')
        ax[0].set_xlabel('#No. Episode')
        ax[0].set_ylabel('Total reward')
        ax[0].grid()

        ax[1].plot(np.arange(1, len(lenghts)+1), lenghts)
        ax[1].set_title('Episode duration')
        ax[1].set_xlabel('#No. Episode')
        ax[1].set_ylabel('Duration [in iterations]')
        ax[1].grid()

        plt.suptitle(suptitle)
        plt.tight_layout()
        plt.show()

        fig.savefig(self.results_path + filename, facecolor = 'white', bbox_inches='tight')


    def dump_info(self) -> None:

        """
            Function for dumping agent's info into txt file.

            :params:
                - None

            :return:
                - None
        """

        with open(self.env_name + 'info.txt', 'w') as f:
            f.write(self.__repr__())
            f.write('\n--------------------------------------------------------------------\n')
            f.write(self.actor.summary())
            f.write('\n--------------------------------------------------------------------\n')
            f.write(self.critic.summary())
            f.write('\n--------------------------------------------------------------------\n')