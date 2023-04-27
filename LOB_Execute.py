import gym
import numpy as np
import pandas as pd
import random

#volumn={0,1,5};spread={0,2,4,6,8,10};
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    # 创建显卡
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 是否调用GPU
    num_updates = 1000000
    Out_time_step = 0
    RL_action_time = 400  # 规定RL必须从200步开始行动
    RL_action_step = 0
    gamma = 0.2 #时间折扣因子
    train_sign = False
    norm_action_mu = 50

    env = gym.make('LOB-v0')
    env.reset()

    # 用于Tensorboard可视化
    writer = SummaryWriter("runs/LOB_test")

    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):  # 初始化网络每一层的参数和bias
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def LSTM_init(layer, std=np.sqrt(2), bias_const=0.0):  # 初始化网络每一层的参数和bias
        torch.nn.init.orthogonal_(layer.weight_ih_l0, std)
        torch.nn.init.orthogonal_(layer.weight_hh_l0, std)
        torch.nn.init.constant_(layer.bias_ih_l0, bias_const)
        torch.nn.init.constant_(layer.bias_hh_l0, bias_const)
        return layer

    class Agent(nn.Module):  # 定义agent
        def __init__(self): #Agent 有两个网络，一个网络输出actor+是否买卖，一个网络输出critic
            super().__init__()
            #LSTM层，原论文有25个hidden units，这里设置为4个
            # Observation为4*20的向量，20为seq长度，4是feature数量
            self.lstm = LSTM_init(nn.LSTM(input_size = 4, hidden_size = 4))

            #卷积+max pool
            #卷积输入的是维度为20*2的orderbook，最终输出长度为8的向量
            self.OrderBook1 = layer_init(nn.Conv2d(in_channels = 1,out_channels = 16,kernel_size = (4,2),stride = 1))
            self.OrderBook2 = nn.Sequential(
                layer_init(nn.Conv1d(in_channels = 16,out_channels = 16,kernel_size = 4,stride = 1)),
                nn.MaxPool1d(kernel_size = 2,stride = 1),
                layer_init(nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1)),
                layer_init(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1)),
                nn.MaxPool1d(kernel_size = 2,stride = 1),
                layer_init(nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1, stride=1)),
            )

            #两个全连接
            self.AgentFeature = nn.Sequential(
                layer_init(nn.Linear(in_features=2, out_features=16)),
                nn.ReLU(),
                layer_init(nn.Linear(in_features=16, out_features=8)),
            )

            #总网络，一层merge+两层全连接
            #从前面特征提取得到的输入为16，输出8
            self.Main = nn.Sequential(
                layer_init(nn.Linear(in_features=96, out_features=256)),
                nn.ReLU(),
                layer_init(nn.Linear(in_features=256, out_features=128))
            )

            #Critic网络，输出对应的value
            self.critic = nn.Sequential(
                layer_init(nn.Linear(in_features=128, out_features=64)),
                nn.ReLU(),
                layer_init(nn.Linear(in_features=64, out_features=32)),
                nn.ReLU(),
                layer_init(nn.Linear(in_features=32, out_features=1))
            )

            # actor网络，输出未规范的prob
            self.actor = nn.Sequential(
                layer_init(nn.Linear(in_features=128, out_features=128)),
                nn.ReLU(),
                layer_init(nn.Linear(in_features=128, out_features=64)),
                nn.ReLU(),
                layer_init(nn.Linear(in_features=64, out_features=45))
            )

        def order_trans(self, x):
            x = self.OrderBook1(x)
            x = x.reshape(x.shape[0],x.shape[1])
            return self.OrderBook2(x)

        def get_main(self, x):
            #这里包含两步，第一步就是拆解obs；第二步是三种observation通过三个网络；第三步是将结果merge；第四步是通过main网络
            x_series, x_LOB, x_agent = x
            #print(x_series)
            x_one = torch.tensor(x_series).t().float().to(device)
            x_two = torch.tensor(x_LOB).float().to(device)
            x_three = torch.tensor(x_agent).float().to(device)
            output_series = self.lstm(x_one)[0]#长度为20×4的向量
            output_series = output_series.reshape(-1)
            output_LOB = self.order_trans(x_two).reshape(-1) #长度为8的向量
            output_agent = self.AgentFeature(x_three) #长度为4的向量
            merge_output =  torch.cat((output_series, output_LOB, output_agent), 0)#这一步是三个tensor连成一个长96的tensor向量
            return self.Main(merge_output)

        def get_value(self, x):
            x_main = self.get_main(x)
            return self.critic(x_main)

        def get_actor(self, x): #根据actor网路输出log动作概率
            return self.actor(x)

        def get_action_and_value(self, x):
            # x就是observation，observation需要包含所有输入
            x_main = self.get_main(x)
            action_logits = self.get_actor(x_main)
            action_probs = Categorical(logits=action_logits)  # 指category distribution,这里假设actor网络输出的是log distribution
            action = action_probs.sample()  # 按概率选择动作
            return action, action_probs.log_prob(action), action_probs.entropy(), self.critic(x_main)

    #新建RL agent实例和优化器
    agent = Agent().to(device)
    agent.load_state_dict(torch.load("Vanilla_AC.pth"))
    optimizer = optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

    # 这里我们只考虑在一个eps里面的事情
    # 先考虑在线学习
    action = logprob = _ = value = observation = RL_amount = 0
    for update in range(1, num_updates + 1):
        if False:#env.time_step == RL_action_time:  # 如果到了RL行动的时间；

            RL_action_time += round(np.random.normal(loc = norm_action_mu,scale = 10))  # 更新下一次的行动时间
            # Observation为4*20的向量，20为seq长度，4是feature数量

            # 如果是online算法的actor-crtic，直接更新即可
            """
            # --------第一种，advantage项 = reward - V(now)
            action, logprob, _, value = agent.get_action_and_value(observation)
            observation, reward, done, info = env.step(action)
            pg_loss = -(reward - value)*logprob
            entropy_loss = _
            v_loss = 0.5 * ((value - reward) ** 2)
            loss = pg_loss - entropy_loss + v_loss
            """

            # --------第二种，advantage项 = reward+gamma*V(next)
            # 判断此点之前是否训练了
            if train_sign:

                next_value = agent.get_value(observation)
                reward = env.RL_Agent['Total_amount'] - RL_amount

                pg_loss = -(reward + gamma * next_value - value) * logprob
                entropy_loss = _
                v_loss = 0.5 * ((reward + gamma * next_value - value) ** 2)
                loss = pg_loss - entropy_loss + v_loss

                optimizer.zero_grad()
                # loss.backward() 本质就是计算loss梯度并传递给每一个网络
                loss.backward()
                # 如果对梯度范数的最大值有要求(默认2范数)，那么会对所有梯度进行修改使其<max_grad_norm
                # https://blog.csdn.net/Mikeyboi/article/details/119522689
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

                print("已完成一次训练,目前是第%i步" % env.time_step)

                writer.add_scalar("losses/value_loss", v_loss.item(), RL_action_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), RL_action_step)
                writer.add_scalar("losses/entropy", entropy_loss.item(), RL_action_step)

                RL_amount = env.RL_Agent['Total_amount']
                action, logprob, _, value = agent.get_action_and_value(observation)
                observation, reward, done, info = env.step(action)

                if done:
                    RL_action_time = 1000
                    train_sign = False
                else:
                    train_sign = True
            else:
                action, logprob, _, value = agent.get_action_and_value(observation)
                observation, reward, done, info = env.step(action)

                RL_amount = env.RL_Agent['Total_amount']

                train_sign = True

            Out_time_step += 1
            RL_action_step += 1
            print("已完成一次训练,目前是第%i步" % env.time_step)

            if done:
                writer.add_text('GBM_mu', str(env.GBM_mu), global_step=0)
                RL_action_time = 400

        if env.time_step == RL_action_time:
            RL_action_time += round(np.random.normal(loc=norm_action_mu, scale=10))  # 更新下一次的行动时间
            action, logprob, _, value = agent.get_action_and_value(observation)
            action = [action,random.randint(0,44)]
            observation, reward, done, info = env.step(action)

            print("已行动一次，目前是%i步" % env.time_step)

            writer.add_scalar("losses/RL_amount", env.RL_Agent['Total_amount'], RL_action_step)
            writer.add_scalar("losses/RL_inventory", env.RL_Agent['Inventory'], RL_action_step)
            writer.add_scalar("losses/RL_Cash", env.RL_Agent['Cash'], RL_action_step)
            writer.add_scalar("losses/Test_amount", env.RL_Agent_two['Total_amount'], RL_action_step)
            writer.add_scalar("losses/Test_inventory", env.RL_Agent_two['Inventory'], RL_action_step)
            writer.add_scalar("losses/Test_Cash", env.RL_Agent_two['Cash'], RL_action_step)

            writer.add_scalar("env/mid_price", env.history_midprice[env.time_step], Out_time_step)
            writer.add_scalar("env/trade_price", env.history_tradePrice[env.time_step], Out_time_step)
            writer.add_scalar("env/Sell_price", env.history_bestSell[env.time_step], Out_time_step)
            writer.add_scalar("env/Buy_price", env.history_bestBuy[env.time_step], Out_time_step)
            writer.add_scalar("env/Buy_LOB_shape", env.Buy_LOB.shape[0], Out_time_step)
            writer.add_scalar("env/Sell_LOB_shape", env.Sell_LOB.shape[0], Out_time_step)

            Out_time_step += 1
            RL_action_step += 1
            if done:
                writer.add_text('GBM_mu', str(env.GBM_mu), global_step=0)
                train_sign = False
                RL_action_time = 400
        else:
            action = 0
            observation, reward, done, info = env.step(action)

            writer.add_scalar("env/mid_price", env.history_midprice[env.time_step], Out_time_step)
            writer.add_scalar("env/trade_price", env.history_tradePrice[env.time_step], Out_time_step)
            writer.add_scalar("env/Sell_price", env.history_bestSell[env.time_step], Out_time_step)
            writer.add_scalar("env/Buy_price", env.history_bestBuy[env.time_step], Out_time_step)
            writer.add_scalar("env/Buy_LOB_shape", env.Buy_LOB.shape[0], Out_time_step)
            writer.add_scalar("env/Sell_LOB_shape", env.Sell_LOB.shape[0], Out_time_step)

            Out_time_step += 1
            if done:
                writer.add_text('GBM_mu', str(env.GBM_mu), global_step=0)
                RL_action_time = 400

    #一整个流程结束，才会记录当局的收益


#所有关于sign的都需要改，如果RL_agent想买，那么volume就是正的；如果想卖，那么volume就是负的









