import argparse
from ppo import PPOAgent

# definition of parser for cmd line arguments input
parser = argparse.ArgumentParser(description='Parsing program arguments - algorithm hyper-parameters ...') 
parser.add_argument('--env_name', type=str)
parser.add_argument('--num_episodes', type=int)
parser.add_argument('--max_iter', type=int)
parser.add_argument('--gamma', type=float)
parser.add_argument('--clip_ratio', type=float)
parser.add_argument('--actor_lr', type=float)
parser.add_argument('--critic_lr', type=float)
parser.add_argument('--train_actor_iter', type=int)
parser.add_argument('--train_critic_iter', type=int)
parser.add_argument('--lambda_', type=float)
parser.add_argument('--target_kl', type=float)
parser.add_argument('--hidden_units', type=list)
parser.add_argument('--buffer_size', type=int)

# main part of program
if __name__ == "__main__":

    # parsing arguments
    args = parser.parse_args()

    # creating agent for Proximal Policy Optimization algorithm
    ppo = PPOAgent(args, 'Cartpole-v0')
    print(ppo)

    ppo.train()
