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
import os
from torch.utils.data import DataLoader
import copy



if __name__ == "__main__":
    # 创建显卡
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 是否调用GPU
    num_updates = 1000000  # 环境总simulat步数
    gamma = 0.2  # 折扣因子
    update_times = 4  # 每次更新的次数
    norm_action_mu = 50
    real_time_step = 0 #总的时间步
    RL_action_step = 0

    env = gym.make('LOB-v0')
    env.reset()
    Out_time_step = 0  # 这个用于计算RL的行动次数，同时确定Replay Buffer的位置
    RL_action_time = 400  # 规定RL必须从200步开始行动

    # 用于Tensorboard可视化-----------------------还需要改
    writer = SummaryWriter(f"runs/LOB_DQN_test")

    # Replay Buffer的最大容量
    max_RB = 2000

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
        def __init__(self):  # Agent 有两个网络，一个网络输出actor+是否买卖，一个网络输出critic
            super().__init__()
            # LSTM层，原论文有25个hidden units，这里设置为4个
            # Observation为4*20的向量，20为seq长度，4是feature数量
            self.lstm = LSTM_init(nn.LSTM(input_size=4, hidden_size=4))

            # 卷积+max pool
            # 卷积输入的是维度为20*2的orderbook，最终输出长度为8的向量
            self.OrderBook1 = layer_init(nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 2), stride=1))
            self.OrderBook2 = nn.Sequential(
                layer_init(nn.Conv1d(in_channels=16, out_channels=16, kernel_size=4, stride=1)),
                nn.MaxPool1d(kernel_size=2, stride=1),
                layer_init(nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1)),
                layer_init(nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1)),
                nn.MaxPool1d(kernel_size=2, stride=1),
                layer_init(nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1, stride=1)),
            )

            # 两个全连接
            self.AgentFeature = nn.Sequential(
                layer_init(nn.Linear(in_features=2, out_features=16)),
                nn.ReLU(),
                layer_init(nn.Linear(in_features=16, out_features=8)),
            )

            # 总网络，一层merge+两层全连接
            # 从前面特征提取得到的输入为16，输出8
            self.Main = nn.Sequential(
                layer_init(nn.Linear(in_features=96, out_features=256)),
                nn.ReLU(),
                layer_init(nn.Linear(in_features=256, out_features=128))
            )

            # Critic网络，输出对应的每一个action-state的value
            self.critic = nn.Sequential(
                layer_init(nn.Linear(in_features=128, out_features=128)),
                nn.ReLU(),
                layer_init(nn.Linear(in_features=128, out_features=64)),
                nn.ReLU(),
                layer_init(nn.Linear(in_features=64, out_features=21))
            )

        def order_trans(self, x):
            x = self.OrderBook1(x)
            x = x.reshape(x.shape[0], x.shape[1])
            return self.OrderBook2(x)

        def get_main(self, x):
            # 这里包含两步，第一步就是拆解obs；第二步是三种observation通过三个网络；第三步是将结果merge；第四步是通过main网络
            x_series, x_LOB, x_agent = x
            # print(x_series)
            if torch.is_tensor(x_series):
                x_one = x_series.t().float()
                x_two = x_LOB.float()
                x_three = x_agent.float()
            else:
                x_one = torch.tensor(x_series).t().float().to(device)
                x_two = torch.tensor(x_LOB).float().to(device)
                x_three = torch.tensor(x_agent).float().to(device)
            output_series = self.lstm(x_one)[0]  # 长度为20×4的向量
            output_series = output_series.reshape(-1)
            output_LOB = self.order_trans(x_two)  # 长度为8的向量
            output_LOB = output_LOB.reshape(-1)
            output_agent = self.AgentFeature(x_three)  # 长度为8的向量
            merge_output = torch.cat((output_series, output_LOB, output_agent), 0)  # 这一步是三个tensor连成一个长96的tensor向量
            return self.Main(merge_output)

        def get_value(self, x):
            x_main = self.get_main(x)
            return self.critic(x_main)


    # 新建RL agent实例和优化器
    agent = Agent().to(device)
    #torch.save(agent.state_dict(), "agent_para.pth")
    #target_agent = Agent().to(device)
    #target_agent.load_state_dict(torch.load("agent_para.pth"))
    agent.load_state_dict(torch.load("DQN.pth"))
    optimizer = optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

    # 引入动作列表长度
    action_len = env.RL_action_list.shape[0]

    # 这里是Replay Buffer
    record_obs_one = torch.zeros((max_RB, 4, 20)).to(device)  # 记录当前状态s_t的价格序列,obs1的shape为（4，20）
    record_obs_two = torch.zeros((max_RB, 1, 20, 2)).to(device)  # 记录当前状态s_t的orderbook，obs2的shape为（20，2）
    record_obs_three = torch.zeros((max_RB, 2)).to(device)  # 记录当前状态s_t的agent属性
    record_data = torch.zeros((max_RB, 4)).to(device)  # 记录value,RL_amount,done

    observation = done = 0
    for update in range(1, num_updates + 1):

        if False:  # 如果到了RL行动的时间
            RL_interval = round(np.random.normal(loc=norm_action_mu, scale=10))
            RL_action_time += RL_interval  # 更新下一次的行动时间

            # 此时处于s_t，是决定a_t的状态，因此需要先储存一下
            location = Out_time_step % max_RB
            with torch.no_grad():
                record_data[location, 0] = target_agent.get_value(observation).max()  # 记录s_t的target_return
                record_obs_one[location] = torch.tensor(observation[0]).to(device)  # 记录当前状态s_t的价格序列,obs1的shape为（4，20）
                record_obs_two[location] = torch.tensor(observation[1]).to(device)  # 记录当前状态s_t的orderbook，obs2的shape为（20，2）
                record_obs_three[location] = torch.tensor(observation[2]).to(device)  # 记录当前状态s_t的agent属性
                record_data[location, 1] = env.RL_Agent['Total_amount']  # 记录s_t的RL_agent amount
                record_data[location, 2] = done  # 记录是否是最后一步

                # ------------执行动作----------------------------
                if done:
                    action = np.random.randint(0, action_len)
                else:
                    if random.uniform(0, 1) * (1 - (Out_time_step) / (num_updates / norm_action_mu)) < 0.2:
                        action = int(agent.get_value(observation).argmax())
                    else:
                        action = np.random.randint(0, action_len)
                record_data[location, 3] = action  # 记录是否是最后一步

            observation, reward, done, info = env.step(action)

            # ------------------从第300步开始，每走一步，会训练一次；且每隔200步才更新一次参数---------------------------------#
            if Out_time_step > 50:
                if Out_time_step < max_RB:  # 随机抽取训练样本
                    RB_content = Out_time_step  # 如果RB没满，我们需要得到当前RB内的个数
                else:
                    RB_content = max_RB

                train_choice = random.sample(range(0, RB_content), update_times)
                for i in train_choice:
                    # 计算batch中每一个list对应的loss，最后计算平均并更新
                    if record_data[(i+1)%max_RB, 0] == 0:
                        train_reward = 0
                        train_value_ = 0
                    else:
                        train_reward = record_data[(i+1)%max_RB, 1] - record_data[i, 1]
                        if env.RL_Agent['Inventory'] < 0:
                            train_reward += env.RL_Agent["Inventory"] * 2 * env.history_midprice[env.time_step]
                        else:
                            train_reward -= env.RL_Agent["Inventory"] * env.history_midprice[env.time_step] / 5
                        if env.RL_Agent['Cash'] < 0:
                            train_reward += env.RL_Agent["Cash"]
                        train_value_ = agent.get_value([record_obs_one[i],record_obs_two[i],record_obs_three[i]])[record_data[i, 3].long()]

                    # Value loss
                    if True:
                        v_loss_unclipped = (train_value_ - (train_reward + gamma * record_data[(i+1)%max_RB, 0])) ** 2
                        v_clipped = record_data[i, 0] + torch.clamp(  # b_values是V_old计算出的value
                            train_value_ - record_data[i, 0], -0.2, 0.2, )  # 控制V_new相比V_old的变化幅度
                        v_loss_clipped = (v_clipped - (train_reward + gamma * record_data[(i+1)%max_RB, 0])) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max
                    else:
                        v_loss = (train_value_ - (train_reward + gamma * record_data[(i+1)%max_RB, 0])) ** 2

                    # 总loss =value loss
                    loss = v_loss

                    optimizer.zero_grad()
                    # loss.backward() 本质就是计算loss梯度并传递给每一个网络
                    loss.backward()
                    # 如果对梯度范数的最大值有要求(默认2范数)，那么会对所有梯度进行修改使其<max_grad_norm
                    # https://blog.csdn.net/Mikeyboi/article/details/119522689
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()
                #print("已更新一次")

                writer.add_scalar("losses/value_loss", v_loss.item(), Out_time_step)

            if Out_time_step % 50 == 0:
                torch.save(agent.state_dict(), "agent_para.pth")
                target_agent.load_state_dict(torch.load("agent_para.pth"))

            Out_time_step += 1
            real_time_step += 1

            print("RL已行动%i次,目前是第%i步" % (Out_time_step, env.time_step))

            # 一整个训练流程结束，画出这一轮的训练结果
            writer.add_scalar("losses/RL_amount", env.RL_Agent['Total_amount'], Out_time_step)
            writer.add_scalar("losses/RL_inventory", env.RL_Agent['Inventory'], Out_time_step)
            writer.add_scalar("losses/RL_Cash", env.RL_Agent['Cash'], Out_time_step)

            writer.add_scalar("env/mid_price", env.history_midprice[env.time_step], real_time_step)
            writer.add_scalar("env/trade_price", env.history_tradePrice[env.time_step], real_time_step)
            writer.add_scalar("env/Sell_price", env.history_bestSell[env.time_step], real_time_step)
            writer.add_scalar("env/Buy_price", env.history_bestBuy[env.time_step], real_time_step)
            writer.add_scalar("env/Buy_LOB_shape", env.Buy_LOB.shape[0], real_time_step)
            writer.add_scalar("env/Sell_LOB_shape", env.Sell_LOB.shape[0], real_time_step)
            writer.add_text('GBM_mu', str(env.GBM_mu), global_step=0)

            if done:
                RL_action_time = 400

        if env.time_step == RL_action_time:
            RL_action_time += round(np.random.normal(loc=norm_action_mu, scale=10))  # 更新下一次的行动时间
            action= agent.get_value(observation).argmax()
            action = [action, random.randint(0, 20)]
            observation, reward, done, info = env.step(action)

            print("已行动一次，目前是%i步" % env.time_step)

            writer.add_scalar("losses/RL_amount", env.RL_Agent['Total_amount'], RL_action_step)
            writer.add_scalar("losses/RL_inventory", env.RL_Agent['Inventory'], RL_action_step)
            writer.add_scalar("losses/RL_Cash", env.RL_Agent['Cash'], RL_action_step)
            writer.add_scalar("losses/Test_amount", env.RL_Agent_two['Total_amount'], RL_action_step)
            writer.add_scalar("losses/Test_inventory", env.RL_Agent_two['Inventory'], RL_action_step)
            writer.add_scalar("losses/Test_Cash", env.RL_Agent_two['Cash'], RL_action_step)

            writer.add_scalar("env/mid_price", env.history_midprice[env.time_step], real_time_step)
            writer.add_scalar("env/trade_price", env.history_tradePrice[env.time_step], real_time_step)
            writer.add_scalar("env/Sell_price", env.history_bestSell[env.time_step], real_time_step)
            writer.add_scalar("env/Buy_price", env.history_bestBuy[env.time_step], real_time_step)
            writer.add_scalar("env/Buy_LOB_shape", env.Buy_LOB.shape[0], real_time_step)
            writer.add_scalar("env/Sell_LOB_shape", env.Sell_LOB.shape[0], real_time_step)

            real_time_step += 1
            RL_action_step += 1
            if done:
                writer.add_text('GBM_mu', str(env.GBM_mu), global_step=0)
                train_sign = False
                RL_action_time = 400
        else:
            action = 0
            observation, reward, done, info = env.step(action)

            writer.add_scalar("env/mid_price", env.history_midprice[env.time_step], real_time_step)
            writer.add_scalar("env/trade_price", env.history_tradePrice[env.time_step], real_time_step)
            writer.add_scalar("env/Sell_price", env.history_bestSell[env.time_step], real_time_step)
            writer.add_scalar("env/Buy_price", env.history_bestBuy[env.time_step], real_time_step)
            writer.add_scalar("env/Buy_LOB_shape", env.Buy_LOB.shape[0], real_time_step)
            writer.add_scalar("env/Sell_LOB_shape", env.Sell_LOB.shape[0], real_time_step)
            writer.add_text('GBM_mu', str(env.GBM_mu), global_step=0)

            real_time_step += 1

            if done:
                RL_action_time = 400




















