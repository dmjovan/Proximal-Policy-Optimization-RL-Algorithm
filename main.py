import json
import sys
import argparse
from ppo import PPOAgent
import warnings
warnings.filterwarnings("ignore")

# definition of parser for cmd line arguments input
parser = argparse.ArgumentParser(description='Parsing program arguments - algorithm hyper-parameters ...') 
parser.add_argument('--env_name', type=str)
parser.add_argument('--num_episodes', type=int)
parser.add_argument('--max_iter', type=int)
parser.add_argument('--eval_episodes', type=int)
parser.add_argument('--gamma', type=float)
parser.add_argument('--clip_ratio', type=float)
parser.add_argument('--actor_lr', type=float)
parser.add_argument('--critic_lr', type=float)
parser.add_argument('--num_update_episodes', type=int)
parser.add_argument('--action_std_init', type=float)

# main part of program
if __name__ == "__main__":

    # discrete environments
    with open('envs.json') as json_file:
        envs_dict = json.load(json_file)

    # parsing arguments
    args = parser.parse_args()

    # iterating ove envs
    for env_name in envs_dict.keys():

        # creating agent for Proximal Policy Optimization algorithm
        ppo_agent = PPOAgent(args, 
                             env_name = env_name,
                             num_episodes = envs_dict[env_name]['num_episodes'],
                             max_iter = envs_dict[env_name]['max_iter'],
                             num_update_episodes = envs_dict[env_name]['num_update_episodes'])

        # training agent
        ppo_agent.train(update_timestep = envs_dict[env_name]['update_timestep'])

        # evaluation 
        ppo_agent.evaluate()

        # dumping all informations
        ppo_agent.dump_log()

        # deleting agent
        del ppo_agent

        # breaking loop because arguments are passed 
        if len(sys.argv) > 1:
            break
