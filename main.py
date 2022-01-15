import argparse
from ppo import PPOAgent, available_envs

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
parser.add_argument('--train_actor_iter', type=int)
parser.add_argument('--train_critic_iter', type=int)
parser.add_argument('--lambda_', type=float)
parser.add_argument('--target_kl', type=float)
parser.add_argument('--hidden_units', type=list)
parser.add_argument('--buffer_size', type=int)

# main part of program
if __name__ == "__main__":

    # discrete environments
    envs = [env for env in available_envs if available_envs[env]['type'] == 'discrete']

    # parsing arguments
    args = parser.parse_args()

    env_name = 'CartPole-v1'

    assert env_name in envs, 'Environment is not supported!'

    # creating agent for Proximal Policy Optimization algorithm
    ppo_agent = PPOAgent(args, env_name, num_episodes=30, max_iter=4000, train_actor_iter=80, train_critic_iter=80)

    # training agent
    ppo_agent.train()

    # evaluation 
    ppo_agent.evaluate(render=True)

    # dumping all informations
    ppo_agent.dump_agent_info()

    # deleting agent
    del ppo_agent
