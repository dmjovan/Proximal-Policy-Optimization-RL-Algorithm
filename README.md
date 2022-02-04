# Proximal-Policy-Optimization-RL-Algorithm
 Implementation and Analysis of Proximal Policy Optimization (PPO) Algorithm from Machine Learning field of Reinforcement Learning
 
 Used OpenAI Gym Environments:
 - **Acrobot-v1**
 - **CartPole-v0**
 - **MountainCarContinuous-v0**
 - **Pendulum-v1**

The script **_main.py_** can be run using Command Line like `python main.py` with following arguments: \
_--env_name CartPole-v1 \
--num_episodes 100 \
--max_iter 1000 \
--eval_episodes 10 \
--gamma 0.99 \
--clip_ratio 0.2 \
--actor_lr 0.0003 \
--critic_lr 0.001 \
--num_update_episodes 80 \
--action_std_init 0.1_ 

**Results**

| Acrobot-v1 | CartPole-v0 | MountainCarContinuous-v0 | Pendulum-v1 |
|------------|:-----------:|:------------------------:|:------------:|
|![alt text](https://github.com/dmjovan/Proximal-Policy-Optimization-RL-Algorithm/blob/main/Acrobot-v1/Acrobot-v1.gif)|![alt text](https://github.com/dmjovan/Proximal-Policy-Optimization-RL-Algorithm/blob/main/CartPole-v0/CartPole-v0.gif)|![alt text](https://github.com/dmjovan/Proximal-Policy-Optimization-RL-Algorithm/blob/main/MountainCarContinuous-v0/MountainCarContinuous-v0.gif)|![alt text](https://github.com/dmjovan/Proximal-Policy-Optimization-RL-Algorithm/blob/main/Pendulum-v1/Pendulum-v1.gif)|
