import os
import gym
import glob
import torch
import shutil
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt

from PIL import Image
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal

available_envs = {
                  'CartPole-v1': {'type': 'discrete', 'reward_info': 'Reward is 1 for every step taken, including the termination step'},
                  'MountainCar-v0': {'type': 'discrete', 'reward_info': 'Reward is -1.0 if not terminal state, 0.0 if agent has reached the flag'},
                  'Acrobot-v1': {'type': 'discrete', 'reward_info': 'Reward is -1.0 if not terminal state, 0.0 if terminal'},
                  'MountainCarContinuous-v0': {'type': 'continuous', 'reward_info': 'Reward is 100 if agent reached the flag, reward is decreased based on amount of energy consumed each step'},
                  'Pendulum-v1': {'type': 'continuous', 'reward_info': 'None'}
                 }

# setting device
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print(f'Device set to : {torch.cuda.get_device_name(device)}')
else:
    print('Device set to : CPU')


class RolloutBuffer:

    """ Implementation of Rollout Buffer for PPO algorithm """

    def __init__(self):

        """ Constructor of the class. """

        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):

        """ Function for deleting all data from buffer. """

        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class CriticModel(nn.Module):

    """ Implementation of Critic Model class. """

    def __init__(self, state_dim) -> None:

        """ 
            Constructor of class.
        
            :params:
                - state_dim: dimensions of environment states

            :return:
                - None
         """

        super().__init__()

        self.state_dim = state_dim

        self.model = nn.Sequential(nn.Linear(self.state_dim, 64),
                                   nn.Tanh(),
                                   nn.Linear(64, 64),
                                   nn.Tanh(),
                                   nn.Linear(64, 1))

    def evaluate(self, state: np.ndarray) -> float:

        """ 
            Function for evaluation of current state via Critic model.
        
            :params:
                - state: current state

            :return:
                - state_value: value for current state, estimated with critic model
         """

        state_value = self.model(state)
        
        return state_value


class ActorModel(nn.Module):

    """ Implementation of Actor Model class. """

    def __init__(self, state_dim: int, action_dim: int, actions_type: str, action_std_init: float=None) -> None:

        """ 
            Constructor of class.
        
            :params:
                - state_dim: dimensions of environment states
                - action_dim: dimensions of environment actions 
                - actions_type: type of actions - either continuous or discrete
                - action_std_init: initial standard deviation for actions, if they are continuous

            :return:
                - None
         """

        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.actions_type = actions_type
        self.action_var = None

        if self.actions_type == 'continuous':

            self.action_var = torch.full((self.action_dim,), action_std_init**2).to(device)
            self.actor = nn.Sequential(nn.Linear(self.state_dim, 64),
                                       nn.Tanh(),
                                       nn.Linear(64, 64),
                                       nn.Tanh(),
                                       nn.Linear(64, self.action_dim),
                                       nn.Tanh())

        elif self.actions_type == 'discrete':

            self.actor = nn.Sequential(nn.Linear(self.state_dim, 64),
                                       nn.Tanh(),
                                       nn.Linear(64, 64),
                                       nn.Tanh(),
                                       nn.Linear(64, self.action_dim),
                                       nn.Softmax(dim=-1))

        
    def set_action_std(self, new_action_std: float) -> None:

        """ 
            Function for setting new actor standard deviation/variance.
        
            :params:
                - new_action_std: new standard deviation for actor

            :return:
                - None
         """

        if self.actions_type == 'continuous':
            self.action_var = torch.full((self.action_dim,), new_action_std**2).to(device)
        else:
            raise RuntimeError('Cannot set standard deviation! Actor model is instantiated for continuous actions, not for discrete!')


    def feed_forward(self, state: np.ndarray) -> tuple:

        """ 
            Function for feeding Actor model with state and getting action and log probabilities from model.
        
            :params:
                - state: current agents state

            :return:
                - action: action proposed by Actor model for provided state
                - action_logprob: log-probability for proposed action
         """

        if self.actions_type == 'continuous':

            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)

        elif self.actions_type == 'discrete':

            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        # sampling actions from Multivariable Normal or from Categorical Distribution 
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state: np.ndarray, action: float) -> tuple:

        """ 
            Function for evaluation of current state and action via Actor model.
        
            :params:
                - state: current state
                - action: proposed action in current state

            :return:
                - action_logprobs: log-probabilites fro proposed action
                - dist_entropy: output distribution entropy
         """

        if self.actions_type == 'continuous':

            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        elif self.actions_type == 'discrete':

            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, dist_entropy


class PPOAgent:

    """ Implementation of Proximal Policy Optimization agent from Machine Learning field of Reinforcement Learning """    

    def __init__(self, 
                 args: object,
                 env_name: str=None, 
                 num_episodes: int=3000,
                 max_iter: int=1000,
                 eval_episodes: int=10,
                 gamma: float=0.99,
                 clip_ratio: float=0.2,
                 actor_lr: float=0.0003,
                 critic_lr: float=0.001,
                 num_update_episodes: int=80,
                 action_std_init: float=0.6,
                 hidden_units: tuple=(64,64),
                 load_model: bool=False,
                 random_seed: int=0) -> None:

        """ 
            Constructor of the class.

            :params:
                - args: ArgumentParser object

                ********** constructor arguments for IDE calling **********

                - env_name: name of OpenAI Gym Environments
                - num_episodes: number of episodes for training
                - max_iter: maximum of iterations in one episode
                - eval_episodes: number of evaluation episodes
                - gamma: discount factor
                - clip_ratio: clipping ratio
                - actor_lr: Actor model learning rate
                - critic_lr: Critic model learning rate
                - num_update_episodes: nnumber of episodes/epochs for updating Actor and Critic models
                - action_std_init: standard deviation for Actor model
                - hidden_units: tuple containing all number of units/neurons for each layer
                - load_model: indicator for loading models if exist
                - random_seed: random seed number

            :return:
                - None
        """

        # defining hyper-parameters
        self.env_name = env_name if args.env_name is None else args.env_name
        self.num_episodes = num_episodes if args.num_episodes is None else args.num_episodes
        self.max_iter = max_iter if args.max_iter is None else args.mx_iter
        self.eval_episodes = eval_episodes if args.eval_episodes is None else args.eval_episodes
        self.gamma = gamma if args.gamma is None else args.gamma
        self.clip_ratio = clip_ratio if args.clip_ratio is None else args.clip_ratio
        self.actor_lr = actor_lr if args.actor_lr is None else args.actor_lr
        self.critic_lr = critic_lr if args.critic_lr is None else args.critic_lr
        self.num_update_episodes = num_update_episodes if args.num_update_episodes is None else args.num_update_episodes
        self.action_std_init = action_std_init if args.action_std_init is None else args.action_std_init
        self.hidden_units = hidden_units if args.hidden_units is None else args.hidden_units

        # check if env_name is supported
        assert self.env_name in available_envs.keys(), 'Environment is not supported!'
    
        self.env_type = available_envs[self.env_name]['type']

        # creating OpenAi Gym Environment
        self.env = gym.make(self.env_name)

        # getting dimensions of state and action spaces
        self.state_dim = self.env.observation_space.shape[0]

        if self.env_type == 'continuous':
            self.action_dim  = self.env.action_space.shape[0]
        
        elif self.env_type == 'discrete':
            self.action_dim = self.env.action_space.n

        # creating paths and folders for storing results
        self.models_path = self.env_name + '\\models\\'
        self.plots_path = self.env_name + '\\plots\\'
        self.gif_images_path = self.env_name + '\\gif_images\\'
        self.gif_path  = self.env_name + '\\gif\\'
        self.log_path = self.env_name + '\\log\\'

        if os.path.exists(self.env_name) and not load_model:
            shutil.rmtree(self.env_name) 
        
        if not os.path.exists(self.models_path):
            os.makedirs(self.models_path)
        
        if not os.path.exists(self.plots_path):
            os.makedirs(self.plots_path)

        if not os.path.exists(self.gif_images_path):
            os.makedirs(self.gif_images_path)
        
        if not os.path.exists(self.gif_path):
            os.makedirs(self.gif_path)
          
        if not os.path.exists(self.log_path):  
            os.makedirs(self.log_path)
    
        # initialization of all buffers for storing trajectories
        self.buffer = RolloutBuffer()

        self.action_std = self.action_std_init

        # creating Actor and Critic models - CURRENT POLICY
        self.actor = ActorModel(self.state_dim, self.action_dim, self.env_type, self.action_std_init).to(device)
        self.critic = CriticModel(self.state_dim).to(device)

        # initialization of Actor and Critic optimizers
        self.optimizer = torch.optim.Adam([{'params': self.actor.parameters(), 'lr': self.actor_lr},
                                           {'params': self.critic.parameters(), 'lr': self.critic_lr}])

        # creating Actor and Critic models - OLD POLICY
        self.actor_old = ActorModel(self.state_dim, self.action_dim, self.env_type, self.action_std_init).to(device)
        self.critic_old = CriticModel(self.state_dim).to(device)

        # load last state dict for old policy
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        # loading from stored models
        if load_model:
            self.load_models()

        # initialization of Mean-Squared Error Loss function from PyTorch
        self.MseLoss = nn.MSELoss()
        
        # setting random seeds
        if random_seed:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed) 
            self.env.seed(random_seed)

        # initialization of training and evaluation records
        self.train_episodic_rewards = []
        self.train_episodic_lengths = []
        self.eval_episodic_rewards = []
        self.eval_episodic_lengths = []

        # initialization of train and eval texts
        self.train_text = ''
        self.eval_text = ''


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
        rep += f'Environment state space size: {self.state_dim}\n'
        rep += f'Environment action space is: {self.env_type}\n'
        rep += f'Environment number of actions is: {self.action_dim}\n'
        rep += f'Environment reward definition: {available_envs[self.env_name]["reward_info"]}\n'
        rep += '********************\n'
        rep += '********************\n'
        rep += f'Number of training episodes: {self.num_episodes}\n'
        rep += f'Maximum iterations over one episode: {self.max_iter}\n'
        rep += f'Discount factor - gamma: {self.gamma}\n'
        rep += f'Clip ratio: {self.clip_ratio}\n'
        rep += f'Actor model learning rate: {self.actor_lr}\n'
        rep += f'Critic model learning rate: {self.critic_lr}\n'
        rep += f'Number of episodes/epochs for updating Actor & Critic model: {self.num_update_episodes}\n'
        rep += f'Number of units for hidden layers of Actor/Critic models: {self.hidden_units}\n'
        rep += f'Initial standard deviation for Actor model (used for cont. action spaces): {self.action_std_init}\n'
        rep += f'Standard deviation for Actor model (used for cont. action spaces): {self.action_std}\n'
        rep += '--------------------------------------------------------------------'

        return rep


    def set_action_std(self, new_action_std: float) -> None:

        """
            Function for setting Actors standard deviation
            for environments with continuous action spaces.

            :params:
                - new_action_std: new Actor standard deviation

            :return:
                - None
        """
        
        if self.env_type == 'continuous':

            self.action_std = new_action_std
            self.actor.set_action_std(new_action_std)
            self.actor_old.set_action_std(new_action_std)

        elif self.env_type == 'discrete':
            raise RuntimeError('Cannot set new standard deviation for Actor model, since actions are discrete.')


    def decay_action_std(self, action_std_decay_rate: float, min_action_std: float) -> None:

        """
            Function for decaying Actors standard deviation
            for environments with continuous action spaces.

            :params:
                - action_std_decay_rate: decay rate
                - min_action_std: minimum std. deviation

            :return:
                - None
        """

        if self.env_type == 'continuous':

            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std

            self.set_action_std(self.action_std)

        elif self.env_type == 'discrete':
            raise RuntimeError('Cannot decay standard deviation for Actor model, since actions are discrete.')

    
    def select_action(self, state: np.ndarray) -> np.ndarray or float:

        """
            Function for selecting action from Actor model.

            :params:
                - state: current state

            :return:
                - action: proposed action(s)
        """

        # feeding Actor network with state
        if self.env_type == 'continuous':

            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.actor_old.feed_forward(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()

        elif self.env_type == 'discrete':         

            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.actor_old.feed_forward(state)
            
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()


    def update_models(self) -> None:

        """
            Function for computing and applying gradients for Actor and Critic models.

            :params:
                - None

            :return:
                - None
        """

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        # Actor & Critic models updating for num_update_episodes
        for _ in range(self.num_update_episodes):

            # evaluation of old actions and states
            logprobs, dist_entropy = self.actor.evaluate(old_states, old_actions)
            state_values = self.critic.evaluate(old_states)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # finding the ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # finding "Surrogate Loss"
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # copy new weights into old policy / Actor & Critic models
        self.actor_old.load_state_dict(self.actor.state_dict())
        self.critic_old.load_state_dict(self.critic.state_dict())

        # clear buffer
        self.buffer.clear()

    
    def save_models(self):

        """
            Function for saving models.

            :params:
                - None

            :return:
                - None
        """

        torch.save(self.actor_old.state_dict(), self.models_path + 'actor.pth')
        torch.save(self.critic_old.state_dict(), self.models_path + 'critic.pth')

    def load_models(self):

        """
            Function for loading Actor & Critic models.

            :params:
                - None

            :return:
                - None
        """
        
        # loading actor models
        self.actor.load_state_dict(torch.load(self.models_path + 'actor.pth', map_location=lambda storage, loc: storage))
        self.actor_old.load_state_dict(torch.load(self.models_path + 'actor.pth', map_location=lambda storage, loc: storage))

        # loading critic models
        self.critic.load_state_dict(torch.load(self.models_path + 'critic.pth', map_location=lambda storage, loc: storage))
        self.critic_old.load_state_dict(torch.load(self.models_path + 'critic.pth', map_location=lambda storage, loc: storage))
        

    def train(self, update_timestep: int=4000, decay_std_timestep: int=int(2e6), action_std_decay_rate: float=0.05, min_action_std: float=0.1) -> None:

        """
            Function for training agent over epsodes, storing interactions and updating Actor and Critic models.

            :params:
                - update_timestep: number of iterations after which update of models is executed
                - decay_std_timestep: number of iterations after which std. deviation of Actor model decay is executed
                - action_std_decay_rate: float number for decay rate 
                - min_action_std: minimum Actor action std. deviation

            :return:
                - None
        """

        self.train_episodic_rewards, self.train_episodic_lengths = [], []

        print(f'---------------- ENVIRONMENT: {self.env_name} ---------------------')
        print('-------------------------- TRAINING --------------------------------')

        time_step = 0

        # iterating over episodes
        for episode in range(1, self.num_episodes+1):

            # reseting environment on the start of episode
            state = self.env.reset()

            # initialize episode reward and episode length
            episode_reward, episode_length = 0, 0

            # iterating over the steps for each episode
            for t in range(1, self.max_iter+1):

                print(f'Executing Episode {episode}/{self.num_episodes} ---> Executing Iteration {t}', end='\r')

                # getting action 
                action = self.select_action(state)

                # taking step in environment and getting reward and new state
                state, reward, done, _ = self.env.step(action)

                # saving reward and is_terminals
                self.buffer.rewards.append(reward)
                self.buffer.is_terminals.append(done)

                episode_reward += reward
                episode_length += 1
                time_step += 1

                if time_step % update_timestep == 0:
                    self.update_models()

                if self.env_type == 'continuous' and time_step % decay_std_timestep == 0:
                    self.decay_action_std(action_std_decay_rate, min_action_std)

                if done:
                    break
            
            # storing episodic rewards and lenghts
            self.train_episodic_rewards.append(episode_reward)
            self.train_episodic_lengths.append(episode_length)

            # print episodic reward and episode duration
            self.train_text += f'Episode: {episode}/{self.num_episodes} --> Episodic Reward: {episode_reward} || Episode Duration [in iterations]: {episode_length}/{self.max_iter}\n'

        # saving models
        self.save_models()

        # closing environment
        self.env.close()

        # visulazing results
        self.visualize()


    def evaluate(self, render: bool=True) -> None:

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

        time_step = 0

        # iterating over episodes
        for episode in range(1, self.eval_episodes + 1):

            # reseting environment on the start of episode
            state = self.env.reset()

            # initialize episode reward and episode length
            episode_reward, episode_length = 0, 0

            # iterating over the steps for each episode
            for t in range(1, self.max_iter + 1):

                print(f'Executing Episode {episode}/{self.eval_episodes} ---> Executing Iteration {t}', end='\r')

                # rendering OpenAI Gym Environment
                if render:
                    img = self.env.render(mode = 'rgb_array')
                    img = Image.fromarray(img)
                    img.save(self.gif_images_path + str(t).zfill(6) + '.jpg')

                # getting action 
                action = self.select_action(state)

                # taking step in environment and getting reward and new state
                state, reward, done, _ = self.env.step(action)

                episode_reward += reward
                episode_length += 1
                time_step += 1

                if done:
                    break

            # clearing buffer
            self.buffer.clear()

            # storing episodic rewards
            self.eval_episodic_rewards.append(episode_reward)
            self.eval_episodic_lengths.append(episode_length)

            # print episodic reward and episode duration
            self.eval_text += f'Episode: {episode}/{self.eval_episodes} --> Episodic Reward: {episode_reward} || Episode Duration [in iterations]: {episode_length}/{self.max_iter}\n'

        # closing environment
        self.env.close()

        # visulazing results
        self.visualize(train=False)

        # making gif
        self.make_gif()

    
    def visualize(self, train: bool=True) -> None:
        
        """
            Function for visualizing agents performance.

            :params:
                - train: indicator for visualizing training (if True) or evaluation (if False) results

            :return:
                - None
        """

        fig, axes = plt.subplots(ncols=2, figsize=(28,14))
        ax = axes.ravel()

        if train:
            ax[0].plot(np.arange(1, len(self.train_episodic_rewards)+1), self.train_episodic_rewards)
            ax[1].plot(np.arange(1, len(self.train_episodic_lengths)+1), self.train_episodic_lengths)
            plt.suptitle('TRAINING PROCESS')
            filename = 'training.png'

        else:
            ax[0].plot(np.arange(1, len(self.eval_episodic_rewards)+1), self.eval_episodic_rewards)
            ax[1].plot(np.arange(1, len(self.eval_episodic_lengths)+1), self.eval_episodic_lengths)
            plt.suptitle('EVALUATION PROCESS')
            filename = 'evaluation.png'
        
        ax[0].set_title('Total episodic reward')
        ax[0].set_xlabel('#No. Episode')
        ax[0].set_ylabel('Total reward')
        ax[0].grid()
        
        ax[1].set_title('Episode duration')
        ax[1].set_xlabel('#No. Episode')
        ax[1].set_ylabel('Duration [in iterations]')
        ax[1].grid()

        plt.tight_layout()
        # plt.show()

        fig.savefig(self.plots_path + filename, facecolor = 'white', bbox_inches='tight')


    def make_gif(self, total_timesteps: int=400, step: int=10, frame_duration: int=200) -> None:

        """
            Function for gif image from stored evaluation images.

            :params:
                - total_timesteps: number of total iterations
                - step: number of iterations to skip
                - frame_duration:  number of iterations for one frame

            :return:
                - None
        """

        img_paths = sorted(glob.glob(self.gif_images_path + r'\\*.jpg'))
        img_paths = img_paths[:total_timesteps]
        img_paths = img_paths[::step]

        # saving gif image
        img, *imgs = [Image.open(f) for f in img_paths]
        img.save(fp=self.gif_path + 'result.gif', format='GIF', append_images=imgs, save_all=True, optimize=True, duration=frame_duration, loop=0)


    def dump_log(self) -> None:

        """
            Function for dumping agent's info into txt file.

            :params:
                - None

            :return:
                - None
        """

        with open(self.log_path + 'log.txt', 'w') as f:
            f.write(self.__repr__())
            f.write('\n--------------------------------------------------------------------\n')
            f.write('-------------------------- TRAINING --------------------------------\n')
            f.write(self.train_text)
            f.write('--------------------------------------------------------------------\n')
            f.write('-------------------------- EVALUATION ------------------------------\n')
            f.write(self.eval_text)