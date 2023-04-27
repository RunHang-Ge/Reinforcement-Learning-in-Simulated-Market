# Reinforcement-Learning-in-Simulated-Market
The core code of one of my thesis, including a simulated high-frequency Limit Order Book(LOB) financial market based on openai.gym and four kinds of Reinforcement Learning algorithms based on Pytorch.

Here I separated my work into four parts:
## LOB_Env.py
The environment created based on openai.gym. Its aim is to simulate a LOB market with high-frequency trading. It includes several functions:
### Initialization
Initialize the environment containing three kinds of agents: norm agents, FCN agents and Reinforcement Learning (RL) agents.
### Taking steps
Each agent needs to take its actions as:
- **Norm agents**: make a random bid or buy action;
- **FCN agents**: make basic prediction of the future price based on moving average, and take buy or bid based on their own prediction
- **RL agents**: take action based on RL
### Calculation
After all agents taking actions, the current LOB, including bid side and buy side, will update in each step. The market's information will also update in:
- **Current Trading Price**: the average of the highest buy price and lowest bid price.
- **Current Buy and Bid Volume**: the sum of volume for both sides.
- **Successful Transaction**: all transactions which are made during this step.
## LOB_Execute.py
This is the core script which connects RL algorithms with the output of LOB market. In this script I achieved an online critic learning with two kinds of advantage calculation:
- **Direct Update**: Advantage = reward - V_T
- **Sarsa**: Advantage = reward + gamma*V_T+1 - V_T
## LOB_Execute_DQN.py
Different from LOB_Execute.py, I achieved DQN with a network structure with three kinds of input:
- **Market Price Series**: a time-series data containing market short time-trend information.
- **LOB Data**: LOB buy and bid order data containing current market information.
- **Agent Feature**: current status of RL agents itself.
## LOB_Exxcute_PPO.py
Different from LOB_Execute_DQN.py, I added the policy network, changing it to an actor-critic algorithm, and applied Proximal Policy Optimization algorithm by changing update procedure and objective functions.
![image](https://github.com/RunHang-Ge/Reinforcement-Learning-in-Simulated-Market/blob/main/Network%20structure.png)
