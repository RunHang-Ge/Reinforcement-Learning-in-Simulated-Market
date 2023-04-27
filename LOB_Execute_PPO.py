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
from torch.utils.data import DataLoader


if __name__ == "__main__":
    # 创建显卡
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 是否调用GPU
    num_updates = 1000000 #环境总simulat步数
    gamma = 0.2 #折扣因子
    max_update = 0 #每次更新时epoch的次数
    update_times = 5 #每次更新epoch中更新的次数
    batch_size = 128 #每次更新时采用的样本数
    norm_action_mu = 50

    env = gym.make('LOB-v0')
    env.reset()
    Out_time_step = 0 #这个用于计算整个环境的行动
    RL_action_step = 0 #用于计算RL agent行动的次数
    RL_action_time = 400 #规定RL必须从400步开始行动
    RL_train_time = 200 #规定每100步RL会训练一次

    # 用于Tensorboard可视化-----------------------还需要改
    writer = SummaryWriter(f"runs/LOB_PPO_test")

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
            if torch.is_tensor(x_series):
                x_one = x_series.t().float()
                x_two = x_LOB.float()
                x_three = x_agent.float()
            else:
                x_one = torch.tensor(x_series).t().float().to(device)
                x_two = torch.tensor(x_LOB).float().to(device)
                x_three = torch.tensor(x_agent).float().to(device)
            output_series = self.lstm(x_one)[0]#长度为20×4的向量
            output_series = output_series.reshape(-1)
            output_LOB = self.order_trans(x_two) #长度为8的向量
            output_LOB = output_LOB.reshape(-1)
            output_agent = self.AgentFeature(x_three) #长度为4的向量
            merge_output =  torch.cat((output_series, output_LOB, output_agent), 0)#这一步是三个tensor连成一个长96的tensor向量
            return self.Main(merge_output)

        def get_value(self, x):
            return self.critic(x)

        def get_actor(self, x): #做两件事：1、根据actor网路输出log动作概率；2、根据sign网络输出买卖方向
            return self.actor(x)

        def get_action_and_value(self, x, action = None):
            # x就是observation，observation需要包含所有输入
            # sign表示这次执行买，卖，不动，是一个{0，1，2}中的整数；logits是一个长度为22的动作列表
            x_main = self.get_main(x)
            action_logits = self.get_actor(x_main)
            action_probs = Categorical(logits=action_logits)  # 指category distribution,这里假设actor网络输出的是log distribution
            if action is None:
                action = action_probs.sample()  # 按概率选择动作
            return action, action_probs.log_prob(action), action_probs.entropy(), self.critic(x_main)

    #新建RL agent实例和优化器
    agent = Agent().to(device)
    agent.load_state_dict(torch.load("PPO.pth"))
    optimizer = optim.Adam(agent.parameters(), lr=2.5e-4, eps=1e-5)

    # 这里我们只考虑在一个eps里面的事情；考虑离线算法PPO

    # 这里是Replay Buffer
    record_obs_one = torch.zeros((max_RB, 4, 20)).to(device)  # 记录当前状态s_t的价格序列,obs1的shape为（4，20）
    record_obs_two = torch.zeros((max_RB, 1, 20, 2)).to(device)  # 记录当前状态s_t的orderbook，obs2的shape为（20，2）
    record_obs_three = torch.zeros((max_RB, 2)).to(device)  # 记录当前状态s_t的agent属性
    record_RL_amount = torch.zeros(max_RB).to(device)   # 记录当前amount
    record_action = torch.zeros(max_RB).to(device)   # 记录当时选择的动作
    record_logprob = torch.zeros(max_RB).to(device)  # 记录当时选择动作的logprob
    record_advantage = torch.zeros(max_RB).to(device)   # 记录reward-value
    record_value = torch.zeros(max_RB).to(device)  # 记录当时的critic网络估计
    record_RL_cash = torch.zeros(max_RB).to(device)
    record_RL_inventory = torch.zeros(max_RB).to(device)

    for update in range(1, num_updates + 1):
        if False:  # 如果到了RL行动的时间
            RL_action_time += round(np.random.normal(loc=norm_action_mu, scale=10))  # 更新下一次的行动时间

            location = RL_action_step % max_RB
            # ----------------每次RL行动后，会记录这次行动结果--------------------- #
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(observation)
                record_obs_one[location] = torch.tensor(observation[0]).to(device)
                record_obs_two[location] = torch.tensor(observation[1]).to(device)
                record_obs_three[location] = torch.tensor(observation[2]).to(device)
                record_value[location] = value
                record_action[location] = action
                record_logprob[location] = logprob
                record_RL_amount[location] = env.RL_Agent['Total_amount']
                record_RL_cash[location] = env.RL_Agent['Cash']
                record_RL_inventory[location] = env.RL_Agent['Inventory']

            observation, reward, done, info = env.step(action)

            writer.add_scalar("env/RL_amount", env.RL_Agent['Total_amount'], Out_time_step)
            writer.add_scalar("env/RL_inventory", env.RL_Agent['Inventory'], Out_time_step)
            writer.add_scalar("env/RL_Cash", env.RL_Agent['Cash'], Out_time_step)

            writer.add_scalar("env/mid_price", env.history_midprice[env.time_step], Out_time_step)
            writer.add_scalar("env/trade_price", env.history_tradePrice[env.time_step], Out_time_step)
            writer.add_scalar("env/Sell_price", env.history_bestSell[env.time_step], Out_time_step)
            writer.add_scalar("env/Buy_price", env.history_bestBuy[env.time_step], Out_time_step)
            writer.add_scalar("env/Buy_LOB_shape", env.Buy_LOB.shape[0], Out_time_step)
            writer.add_scalar("env/Sell_LOB_shape", env.Sell_LOB.shape[0], Out_time_step)

            Out_time_step += 1
            RL_action_step += 1
            print("RL已行动%i次,目前是第%i步" % (RL_action_step, env.time_step))

            if done:
                writer.add_text('GBM_mu', str(env.GBM_mu), global_step=0)
                RL_action_time = 400

        if env.time_step == RL_action_time:
            RL_action_time += round(np.random.normal(loc=norm_action_mu, scale=10))  # 更新下一次的行动时间
            action, logprob, _, value = agent.get_action_and_value(observation)
            action = [action, random.randint(0, 44)]
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


        if False:#RL_action_step == RL_train_time : ## 每隔一段时间再开始更新;
            RL_train_time += 50
            with torch.no_grad():
                """
                #-----方法1：advantage = reward - V(now)
                record_advantage = record_reward - record_value
                """

                if RL_action_step < max_RB:  # 随机抽取训练样本
                    # -----方法2：advantage = reward + gamma*V(next) - V(now)
                    record_value_ = torch.cat((record_value[1:], torch.tensor([0]).to(device)))
                    record_reward = torch.cat((record_RL_amount[1:] - record_RL_amount[:-1],torch.tensor([0]).to(device)))
                    reward += torch.clamp(record_RL_inventory,max=0) * env.history_midprice[env.time_step]
                    reward += torch.clamp(record_RL_cash,max=0)
                    record_advantage = record_reward + gamma * record_value_ - record_value
                    RB_content = RL_action_step - 1  # 如果RB没满，我们需要得到当前RB内的个数
                else:
                    record_value_ = torch.cat((record_value[1:], record_value[0].unsqueeze(0)))
                    record_reward = torch.cat((record_RL_amount[1:] - record_RL_amount[:-1],record_RL_amount[0].unsqueeze(0)))
                    reward += torch.clamp(record_RL_inventory, max=0) * env.history_midprice[env.time_step]
                    reward += torch.clamp(record_RL_cash, max=0)
                    record_advantage = record_reward + gamma * record_value_ - record_value
                    RB_content = max_RB

                # ------数据处理部分，把train data转化为(RB_content,5),feature分别为value,next_value,advantage,reward,logprob
                train_data = torch.vstack(
                    (record_value[:RB_content], record_value_[:RB_content], record_advantage[:RB_content],
                        record_reward[:RB_content], record_logprob[:RB_content])).T

            epoch = 0
            while epoch <= max_update:
                epoch += 1
                clipfracs = []

                train_choice = random.sample(range(0,(RB_content - batch_size - 1)),update_times)

                # 计算batch中每一个list对应的loss，最后计算平均并更新
                for i in train_choice:

                    batch_data = train_data[i:(i+batch_size)]
                    batch_obs1 = record_obs_one[i:(i+batch_size)]
                    batch_obs2 = record_obs_two[i:(i+batch_size)]
                    batch_obs3 = record_obs_three[i:(i+batch_size)]

                    train_logratio = torch.tensor([]).to(device)
                    train_newvalue = torch.tensor([]).to(device)
                    train_entropy = torch.tensor([]).to(device)
                    for index in range(0,batch_size):
                        _, new_logprob, entropy, new_value = agent.get_action_and_value(
                            [batch_obs1[index], batch_obs2[index], batch_obs3[index]])  # 输出logΠ_new和V_new
                        logratio = new_logprob - batch_data[index,4]  # logratio = logΠ_new - logΠ_old
                        logratio = logratio.unsqueeze(0)
                        entropy = entropy.unsqueeze(0)
                        train_logratio = torch.cat((train_logratio, logratio))
                        train_newvalue = torch.cat((train_newvalue, new_value))
                        train_entropy = torch.cat((train_entropy, entropy))

                    ratio = train_logratio.exp()  # ratio = Π_new/Π_old

                    """
                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-train_logratio).mean()
                        approx_kl = ((ratio - 1) - train_logratio).mean()
                        # float() 将T/F转化为0/1；mean()计算超过clip_coef的比例，输出tensor；item()转化为float
                        clipfracs += [((ratio - 1.0).abs() > 0.2).float().mean().item()]
                    """

                    # Policy loss, PPO中的目标函数，用截断控制KL散度变化
                    pg_loss1 = -batch_data[:,2] * ratio
                    # clamp的作用类似于截断：https://pytorch.org/docs/stable/generated/torch.clamp.html#torch.clamp
                    pg_loss2 = -batch_data[:,2] * torch.clamp(ratio, 1 - 0.2, 1 + 0.2)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    # 这里引入了一个和PPO的cliped policy gradient类似的思路，用cliped policy gradient,保证value 更新的幅度
                    if True:
                        v_loss_unclipped = (train_newvalue - (batch_data[:,0]+gamma*batch_data[:,1])) ** 2  # Return就是reward的weighted sum，等价于实际value V_sample
                        v_clipped = batch_data[:,0] + torch.clamp(  # b_values是V_old计算出的value
                            train_newvalue - batch_data[:,0], -0.2, 0.2,)  # 控制V_new相比V_old的变化幅度
                        v_loss_clipped = (v_clipped - (batch_data[:,0]+gamma*batch_data[:,1])) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - train_reward[mini_index]) ** 2).mean()

                    #Entropy loss
                    entropy_loss = train_entropy.mean()

                    # 总loss = policy loss+entropy平均+value loss
                    loss = pg_loss - entropy_loss + v_loss

                    optimizer.zero_grad()
                    # loss.backward() 本质就是计算loss梯度并传递给每一个网络
                    loss.backward()
                    # 如果对梯度范数的最大值有要求(默认2范数)，那么会对所有梯度进行修改使其<max_grad_norm
                    # https://blog.csdn.net/Mikeyboi/article/details/119522689
                    nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                    optimizer.step()

            print("已完成一次训练")
            writer.add_scalar("losses/value_loss", v_loss.item(), RL_action_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), RL_action_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), RL_action_step)












#所有关于sign的都需要改，如果RL_agent想买，那么volume就是正的；如果想卖，那么volume就是负的









