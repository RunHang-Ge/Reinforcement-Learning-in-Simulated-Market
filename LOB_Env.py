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
from gym import spaces

class LOB_Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, Norm_Agent_num = 700, FCN_Agent_num = 300, RL_Agent_num = 1, mu = 0, dev = 0.0001,
                 action_norm = 2, action_FCN = 1, max_step = 100000, start_price_mu = 500, start_price_std = 20, norm_average = 10):
        # 定义环境变量
        self.Norm_Agent_num = Norm_Agent_num #norm agent数量
        self.FCN_Agent_num = FCN_Agent_num #FCN agent数量
        self.RL_Agent_num = RL_Agent_num #RL agent数量
        self.time_step = 0 #当前时间步
        self.RL_action_time = 0
        self.GBM_mu = mu #GBM 均值
        self.GBM_dev = dev #GBM 方差
        self.action_norm = action_norm  # GBM 均值
        self.action_FCN = action_FCN  # GBM 方差
        self.max_step = max_step #最大步数
        self.eps = 0
        self.norm_ave = norm_average #norm agent移动平均的时间长度
        #self.RL_action_list = pd.DataFrame({ #假设RL_action总共有45种情况
        #    'Spread': [0] + list(range(-10,11,2)) * 4,
        #   'Volume': [0] + ([1]*11) + ([5]*11) + ([-1]*11) + ([-5]*11)
        #})
        self.RL_action_list = pd.DataFrame({  # 假设RL_action总共有45种情况
            'Spread': [0] + list(range(-4, 5, 2)) * 4,
            'Volume': [0] + ([1] * 5) + ([5] * 5) + ([-1] * 5) + ([-5] * 5)
        })
        self.start_price_mu = start_price_mu
        self.start_price_std = start_price_std

        print('成功初始化')

    def step(self, action):
        self.time_step += 1

        #所有agent需要更新自己的信息
        # Norm_Agent 更新
        if self.time_step < self.norm_ave:
            if self.time_step == 1:
                self.Norm_Agent['Prediction'] = self.history_midprice[0] + np.random.normal(loc=0, scale=5,size=self.Norm_Agent_num)
            else:
                self.Norm_Agent['Prediction'] = (self.history_midprice[self.time_step - 1] - np.mean(
                    self.history_midprice[0:self.time_step])) / (round(self.time_step / 2)) + self.history_midprice[
                                                    self.time_step - 1] + np.random.normal(loc=0, scale=5,
                                                                                           size=self.Norm_Agent_num)
        else:
            self.Norm_Agent['Prediction'] = (self.history_midprice[self.time_step - 1] - np.mean(
                    self.history_midprice[(self.time_step - self.norm_ave):self.time_step])) / (self.norm_ave / 2) + self.history_midprice[
                                                    self.time_step - 1] + np.random.normal(loc = 0, scale = 5,size = self.Norm_Agent_num)

        # FCN_Agent 更新
        self.FCN_Agent['Fundam_price'] = self.FCN_Agent['Fundam_price'] * np.exp(
            (self.GBM_mu - 1 / 2 * (self.GBM_dev ** 2)) + self.GBM_dev * np.random.normal(size=self.FCN_Agent_num))
        F = np.log(self.FCN_Agent['Fundam_price']/self.history_midprice[self.time_step - 1]) #获得和expect price的对比
        C = np.zeros(shape = self.FCN_Agent_num)
        for i in range(self.FCN_Agent_num): #获取过去一个时间段的估计
            if self.time_step < self.FCN_Agent.loc[i, "Time window"]:
                if self.time_step == 1:
                    C[i] = np.log(self.history_midprice[self.time_step - 1] / self.history_midprice[0])
                else:
                    C[i] = np.log(self.history_midprice[self.time_step - 1] / self.history_midprice[0])/(self.time_step - 1)
            else:
                C[i] = np.log(self.history_midprice[self.time_step - 1] / self.history_midprice[
                        self.time_step - self.FCN_Agent.loc[i, "Time window"]]) / (
                    self.FCN_Agent.loc[i, "Time window"])

        N = np.random.normal(loc = 0,scale = 0.0001,size = self.FCN_Agent_num)
        self.FCN_Agent['r'] = (self.FCN_Agent['Weights1'] * F + self.FCN_Agent['Weights2'] * C + self.FCN_Agent[
            'Weights3'] * N) / (self.FCN_Agent[['Weights1', 'Weights2', 'Weights3']].sum(axis=1))
        self.FCN_Agent['Prediction'] = np.exp(self.FCN_Agent['r']) * self.history_midprice[self.time_step - 1]

        # 每次随机选5个normal agent发出quote；每次选1个FCN_agent发出quote；每隔一段随机时间RL agent发出quote
        # 同时生成两个临时的LOB，用于后续match
        self.norm_choice = random.sample(range(0,self.Norm_Agent_num),k = self.action_norm)
        self.FCN_choice = random.sample(range(0,self.FCN_Agent_num),k = self.action_FCN)
        self.temp_Buy_LOB = self.Buy_LOB
        self.temp_Sell_LOB = self.Sell_LOB

        #Norm_Agent执行行动
        for num in self.norm_choice:
            if self.Norm_Agent.loc[num,'Prediction'] > self.history_midprice[self.time_step - 1]: #如果估计价格上涨，他会买
                buy_quote = pd.DataFrame({
                    'Name': 'Norm' + str(num),
                    'Price': self.Norm_Agent.loc[num, 'Prediction'] * (1 - self.Norm_Agent.loc[num,'k']),
                    'Volume': random.sample([1, 5], k=1),
                    'Time': self.time_step
                })
                # 先判断他是否有足够的钱；如果没有足够钱，就换一个agent直至有钱
                temp_num = num
                iter = 0
                while int(buy_quote['Price']*buy_quote['Volume']) > self.Norm_Agent.loc[temp_num,'Cash']: #df['']返回的是df；df.loc返回的具体一个元素
                    iter += 1
                    #print(self.Norm_Agent.loc[num,'Cash'])
                    #int(buy_quote['Price'] * buy_quote['Volume'])
                    temp_num = random.randint(0,self.Norm_Agent_num - 1)
                    buy_quote = pd.DataFrame({
                        'Name': 'Norm' + str(temp_num),
                        'Price': self.Norm_Agent.loc[temp_num, 'Prediction'] * (1 - self.Norm_Agent.loc[temp_num,'k']),
                        'Volume': random.sample([1, 5], k=1),
                        'Time': self.time_step
                    })
                    if iter > 1000:
                        break
                if random.uniform(0,1) < 0.5 and iter <= 1000: #每次确定后，它有0.5的概率不行动
                    self.temp_Buy_LOB = pd.concat([self.temp_Buy_LOB, buy_quote])
            else: #如果价格下跌，他会卖
                sell_quote = pd.DataFrame({
                    'Name': 'Norm' + str(num),
                    'Price': self.Norm_Agent.loc[num, 'Prediction'] * (1 + self.Norm_Agent.loc[num,'k']),
                    'Volume': random.sample([1, 5], k=1),
                    'Time': self.time_step
                })
                # 先判断他是否有足够的inventory
                temp_num = num
                iter = 0
                while int(sell_quote['Volume']) > self.Norm_Agent.loc[temp_num, 'Inventory']: #df['']返回的是df；df.loc返回的具体一个元素
                    iter += 1
                    #print(self.Norm_Agent.loc[num, 'Inventory'])
                    #print(int(sell_quote['Volume']))
                    temp_num = random.randint(0,self.Norm_Agent_num - 1)
                    sell_quote = pd.DataFrame({
                        'Name': 'Norm' + str(temp_num),
                        'Price': self.Norm_Agent.loc[temp_num, 'Prediction'] * (1 + self.Norm_Agent.loc[temp_num,'k']),
                        'Volume': random.sample([1, 5], k=1),
                        'Time': self.time_step
                    })
                    if iter > 1000:
                        break
                if random.uniform(0, 1) < 0.5 and iter <= 1000:  # 每次确定后，它有0.5的概率不行动
                    self.temp_Sell_LOB = pd.concat([self.temp_Sell_LOB, sell_quote])
        #print("Norm_agent 行动成功")

        #FCN_Agent 行动
        for num in self.FCN_choice:
            if self.FCN_Agent.loc[num,'Prediction'] > self.history_midprice[self.time_step - 1]: #如果估计价格上涨，他会买
                buy_quote = pd.DataFrame({
                    'Name': 'FCN' + str(num),
                    'Price': self.FCN_Agent.loc[num,'Prediction']*(1 - self.FCN_Agent.loc[num,'k']),
                    'Volume': random.sample([1,5],k=1),
                    'Time': self.time_step
                })
                # 先判断他是否有足够的钱
                temp_num = num
                iter = 0
                while int(buy_quote['Price']*buy_quote['Volume']) > self.FCN_Agent.loc[temp_num,'Cash']: #df['']返回的是df；df.loc返回的具体一个元素
                    iter += 1
                    #print(int(buy_quote['Price']*buy_quote['Volume']))
                    #print(self.FCN_Agent.loc[num,'Cash'])
                    temp_num = random.randint(0,self.FCN_Agent_num - 1)
                    buy_quote = pd.DataFrame({
                        'Name': 'FCN' + str(temp_num),
                        'Price': self.FCN_Agent.loc[temp_num,'Prediction']*(1 - self.FCN_Agent.loc[temp_num,'k']),
                        'Volume': random.sample([1, 5], k=1),
                        'Time': self.time_step
                    })
                    if iter > 1000:
                        break
                if random.uniform(0, 1) < 0.5 and iter <= 1000:  # 每次确定后，它有0.5的概率不行动
                    self.temp_Buy_LOB = pd.concat([self.temp_Buy_LOB, buy_quote])
            else: #如果价格下跌，他会卖
                sell_quote = pd.DataFrame({
                    'Name': 'FCN' + str(num),
                    'Price': self.FCN_Agent.loc[num, 'Prediction'] * (1 + self.FCN_Agent.loc[num, 'k']),
                    'Volume': random.sample([1, 5], k=1),
                    'Time': self.time_step
                })
                # 先判断他是否有足够的inventory
                temp_num = num
                iter = 0
                while int(sell_quote['Volume']) > self.FCN_Agent.loc[temp_num, 'Inventory']: #df['']返回的是df；df.loc返回的具体一个元素
                    iter += 1
                    #print(int(sell_quote['Volume']))
                    #print(self.FCN_Agent.loc[num, 'Inventory'])
                    temp_num = random.randint(0,self.FCN_Agent_num - 1)
                    sell_quote = pd.DataFrame({
                        'Name': 'FCN' + str(temp_num),
                        'Price': self.FCN_Agent.loc[temp_num, 'Prediction'] * (1 + self.FCN_Agent.loc[temp_num, 'k']),
                        'Volume': random.sample([1, 5], k=1),
                        'Time': self.time_step
                    })
                    if iter > 1000:
                        break
                if random.uniform(0, 1) < 0.5 and iter <= 1000:  # 每次确定后，它有0.5的概率不行动
                    self.temp_Sell_LOB = pd.concat([self.temp_Sell_LOB, sell_quote])
        #print("FCN agent行动成功")

        #RL agent行动；根据输入的action而定
        if isinstance(action,int):
            action = int(action)
            action = self.RL_action_list.iloc[action]
            # action的形式为[spread,volume]
            if action[1] == 0:
                # print("Stay")
                pass
            elif action[1] > 0:
                buy_quote = pd.DataFrame({
                    'Name': 'RL',
                    'Price': [action[0] + self.history_midprice[self.time_step - 1]],
                    'Volume': [action[1]],
                    'Time': [self.time_step]
                })
                # 不用判断他是否有足够的钱
                self.temp_Buy_LOB = pd.concat([self.temp_Buy_LOB, buy_quote])
            elif action[1] < 0:
                sell_quote = pd.DataFrame({
                    'Name': 'RL',
                    'Price': [action[0] + self.history_midprice[self.time_step - 1]],
                    'Volume': [- action[1]],
                    'Time': [self.time_step]
                })
                # 不用判断他是否有足够的钱
                self.temp_Sell_LOB = pd.concat([self.temp_Sell_LOB, sell_quote])

        else:
            action_one = self.RL_action_list.iloc[int(action[0])]
            action_two = self.RL_action_list.iloc[int(action[1])]
            # action的形式为[spread,volume]
            if action_one[1] == 0:
                # print("Stay")
                pass
            elif action_one[1] > 0:
                buy_quote = pd.DataFrame({
                    'Name': 'RL',
                    'Price': [action_one[0] + self.history_midprice[self.time_step - 1]],
                    'Volume': [action_one[1]],
                    'Time': [self.time_step]
                })
                # 不用判断他是否有足够的钱
                self.temp_Buy_LOB = pd.concat([self.temp_Buy_LOB, buy_quote])
            elif action_one[1] < 0:
                sell_quote = pd.DataFrame({
                    'Name': 'RL',
                    'Price': [action_one[0] + self.history_midprice[self.time_step - 1]],
                    'Volume': [- action_one[1]],
                    'Time': [self.time_step]
                })
                # 不用判断他是否有足够的钱
                self.temp_Sell_LOB = pd.concat([self.temp_Sell_LOB, sell_quote])

            if action_two[1] == 0:
                # print("Stay")
                pass
            elif action_two[1] > 0:
                buy_quote = pd.DataFrame({
                    'Name': 'two',
                    'Price': [action_two[0] + self.history_midprice[self.time_step - 1]],
                    'Volume': [action_two[1]],
                    'Time': [self.time_step]
                })
                # 不用判断他是否有足够的钱
                self.temp_Buy_LOB = pd.concat([self.temp_Buy_LOB, buy_quote])
            elif action_two[1] < 0:
                sell_quote = pd.DataFrame({
                    'Name': 'two',
                    'Price': [action_two[0] + self.history_midprice[self.time_step - 1]],
                    'Volume': [- action_two[1]],
                    'Time': [self.time_step]
                })
                # 不用判断他是否有足够的钱
                self.temp_Sell_LOB = pd.concat([self.temp_Sell_LOB, sell_quote])


        #LOB match交易，产生match order并更新temp LOB的volume
        self.temp_Buy_LOB = self.temp_Buy_LOB.sort_values(by = 'Price',ascending = False) #Buy的价格从高到低
        self.temp_Sell_LOB = self.temp_Sell_LOB.sort_values(by='Price') #Sell的价格从底到高
        #因为np.concat 导致所有行的index均为0;需要修改index
        self.temp_Sell_LOB.index = range(self.temp_Sell_LOB.shape[0])
        self.temp_Buy_LOB.index = range(self.temp_Buy_LOB.shape[0])
        temp_Buy_volume = 0
        temp_Sell_volume = 0
        Buy_LOB_index = 0
        Sell_LOB_index = 0
        self.Matched_Sell = self.Matched_Buy = None
        #都是调用行，df.loc[a:b] 是左闭右闭;df[c:d] 是左闭右开
        while Buy_LOB_index < self.temp_Buy_LOB.shape[0] and Sell_LOB_index < self.temp_Sell_LOB.shape[0]:
            if self.temp_Buy_LOB.loc[Buy_LOB_index,'Price'] >= self.temp_Sell_LOB.loc[Sell_LOB_index,'Price']:
                if (temp_Buy_volume + self.temp_Buy_LOB.loc[Buy_LOB_index,'Volume']) > (temp_Sell_volume + self.temp_Sell_LOB.loc[Sell_LOB_index, 'Volume']):
                    temp_Sell_volume += self.temp_Sell_LOB.loc[Sell_LOB_index, 'Volume']
                    Sell_LOB_index += 1
                    last_incre = 'Sell'
                elif (temp_Buy_volume + self.temp_Buy_LOB.loc[Buy_LOB_index,'Volume']) < (temp_Sell_volume + self.temp_Sell_LOB.loc[Sell_LOB_index, 'Volume']):
                    temp_Buy_volume += self.temp_Buy_LOB.loc[Buy_LOB_index, 'Volume']
                    Buy_LOB_index += 1
                    last_incre = 'Buy'
                else:
                    temp_Sell_volume += self.temp_Sell_LOB.loc[Sell_LOB_index, 'Volume']
                    temp_Buy_volume += self.temp_Buy_LOB.loc[Buy_LOB_index, 'Volume']
                    Buy_LOB_index += 1
                    Sell_LOB_index += 1
                    last_incre = 'Same'
            else: #此时两个temp volume分别代表了可交易的volume
                if Buy_LOB_index == 0 and Sell_LOB_index == 0: #当两个LOB都不为空，且没有达成任何交易
                    break
                if last_incre == 'Sell':
                    if temp_Buy_volume == 0:
                        self.Matched_Sell = self.temp_Sell_LOB[:Sell_LOB_index] # 因为:是左闭右开
                        self.Matched_Buy = self.temp_Buy_LOB.loc[Buy_LOB_index].to_frame().T #因为直接取会得到series，需要转换为df
                        self.Matched_Buy['Volume'] = temp_Sell_volume
                        self.temp_Buy_LOB.loc[Buy_LOB_index, 'Volume'] -= temp_Sell_volume
                        break
                    else:
                        self.Matched_Sell = self.temp_Sell_LOB[:Sell_LOB_index]  # 因为:是左闭右开
                        self.Matched_Buy = self.temp_Buy_LOB[:Buy_LOB_index]
                        part_match_Buy = self.temp_Buy_LOB.loc[Buy_LOB_index].to_frame().T
                        part_match_Buy['Volume'] =  temp_Sell_volume - temp_Buy_volume
                        self.Matched_Buy = pd.concat([self.Matched_Buy, part_match_Buy])
                        self.temp_Buy_LOB.loc[Buy_LOB_index, 'Volume'] -= int(part_match_Buy['Volume'])
                        break
                elif last_incre == 'Buy':
                    if temp_Sell_volume == 0:
                        self.Matched_Buy = self.temp_Buy_LOB[:Buy_LOB_index]
                        self.Matched_Sell = self.temp_Sell_LOB.loc[Sell_LOB_index].to_frame().T
                        self.Matched_Sell['Volume'] = temp_Buy_volume
                        self.temp_Sell_LOB.loc[Sell_LOB_index, 'Volume'] -= temp_Buy_volume
                        break
                    else:
                        self.Matched_Buy = self.temp_Buy_LOB[:Buy_LOB_index]  # 因为:是左闭右开
                        self.Matched_Sell = self.temp_Sell_LOB[:Sell_LOB_index]
                        part_match_Sell = self.temp_Sell_LOB.loc[Sell_LOB_index].to_frame().T  # 此处part_match 是series
                        part_match_Sell['Volume'] = temp_Buy_volume - temp_Sell_volume
                        self.Matched_Sell = pd.concat([self.Matched_Sell, part_match_Sell])
                        self.temp_Sell_LOB.loc[Sell_LOB_index, 'Volume'] -= int(part_match_Sell['Volume'])
                        break
                elif last_incre == 'Same':
                    self.Matched_Buy = self.temp_Buy_LOB[:Buy_LOB_index]  # 因为:是左闭右开
                    self.Matched_Sell = self.temp_Sell_LOB[:Sell_LOB_index]
                    break

        # 如果两个index都超过了temp LOB，此时会跳出循环；证明有一方LOB可以全部match
        if Buy_LOB_index == self.temp_Buy_LOB.shape[0] and Sell_LOB_index == self.temp_Sell_LOB.shape[0]:
            if self.temp_Buy_LOB.empty or self.temp_Buy_LOB.empty:
                pass
            else:
                if temp_Buy_volume == temp_Sell_volume:
                    self.Matched_Sell = self.temp_Sell_LOB
                    self.Matched_Buy = self.temp_Buy_LOB
                elif temp_Buy_volume > temp_Sell_volume:
                    self.Matched_Sell = self.temp_Sell_LOB
                    self.Matched_Buy = self.temp_Buy_LOB.loc[:(Buy_LOB_index - 2)] #不包括最后一行
                    part_match_Buy = self.temp_Buy_LOB.loc[Buy_LOB_index - 1].to_frame().T
                    part_match_Buy['Volume'] = temp_Buy_volume - temp_Sell_volume
                    self.Matched_Buy = pd.concat([self.Matched_Buy, part_match_Buy])
                    self.temp_Buy_LOB.loc[(Buy_LOB_index - 1), 'Volume'] -= int(part_match_Buy['Volume'])
                elif temp_Buy_volume < temp_Sell_volume:
                    self.Matched_Buy = self.temp_Buy_LOB
                    self.Matched_Sell = self.temp_Sell_LOB.loc[:(Sell_LOB_index - 2)]  # 不包括最后一行
                    part_match_Sell = self.temp_Sell_LOB.loc[Sell_LOB_index - 1].to_frame().T
                    part_match_Sell['Volume'] = temp_Sell_volume - temp_Buy_volume
                    self.Matched_Sell = pd.concat([self.Matched_Sell, part_match_Sell])
                    self.temp_Sell_LOB.loc[(Sell_LOB_index - 1), 'Volume'] -= int(part_match_Sell['Volume'])
        elif Buy_LOB_index == self.temp_Buy_LOB.shape[0]:  # 如果是Buy 全部match
            if not self.temp_Buy_LOB.empty:
                self.Matched_Sell = self.temp_Sell_LOB[:Sell_LOB_index]
                self.Matched_Buy = self.temp_Buy_LOB
                part_match_Sell = self.temp_Sell_LOB.loc[Sell_LOB_index].to_frame().T
                part_match_Sell['Volume'] = temp_Buy_volume - temp_Sell_volume
                self.Matched_Sell = pd.concat([self.Matched_Sell, part_match_Sell])
                self.temp_Sell_LOB.loc[Sell_LOB_index, 'Volume'] -= int(part_match_Sell['Volume'])
            else:
                pass
        elif Sell_LOB_index == self.temp_Sell_LOB.shape[0]:
            if not self.temp_Sell_LOB.empty:
                self.Matched_Buy = self.temp_Buy_LOB[:Buy_LOB_index]
                self.Matched_Sell = self.temp_Sell_LOB
                part_match_Buy = self.temp_Buy_LOB.loc[Buy_LOB_index].to_frame().T
                part_match_Buy['Volume'] = temp_Sell_volume - temp_Buy_volume
                self.Matched_Buy = pd.concat([self.Matched_Buy, part_match_Buy])
                self.temp_Buy_LOB.loc[Buy_LOB_index, 'Volume'] -= int(part_match_Buy['Volume'])
            else:
                pass
        #print("已产生所有match order")
        #print("已sort所有temp LOB")

        # 如果某一个orderbook完全匹配，我们会另外结算
        # 每回合只有一个人行动，
        if self.Matched_Sell is not None:
            if Buy_LOB_index == self.temp_Buy_LOB.shape[0] and Sell_LOB_index != self.temp_Sell_LOB.shape[0]:
                self.history_midprice.append((self.temp_Buy_LOB.loc[(Buy_LOB_index - 1), 'Price'] +
                                              self.temp_Sell_LOB.loc[Sell_LOB_index, 'Price']) / 2)
                self.history_bestSell.append(self.temp_Sell_LOB.loc[Sell_LOB_index, 'Price'])
                # 因为Buy都被消耗完，因此buy的报价不能反应市场变化，所以用sell的价格来确定buy的价格
                self.history_bestBuy.append(self.temp_Sell_LOB.loc[Sell_LOB_index, 'Price'] - 1)
                self.history_tradePrice.append(self.history_midprice[self.time_step])
            elif Buy_LOB_index != self.temp_Buy_LOB.shape[0] and Sell_LOB_index == self.temp_Sell_LOB.shape[0]:
                self.history_midprice.append((self.temp_Sell_LOB.loc[(Sell_LOB_index - 1), 'Price'] +
                                              self.temp_Buy_LOB.loc[Buy_LOB_index, 'Price']) / 2)
                self.history_bestSell.append(self.temp_Buy_LOB.loc[Buy_LOB_index, 'Price'] + 1)
                # 因为Buy都被消耗完，因此buy的报价不能反应市场变化，所以用sell的价格来确定buy的价格
                self.history_bestBuy.append(self.temp_Buy_LOB.loc[Buy_LOB_index, 'Price'])
                self.history_tradePrice.append(self.history_midprice[self.time_step])
            elif Buy_LOB_index == self.temp_Buy_LOB.shape[0] and Sell_LOB_index == self.temp_Sell_LOB.shape[0]:
                self.history_midprice.append((self.temp_Sell_LOB.loc[(Sell_LOB_index - 1), 'Price'] +
                                              self.temp_Buy_LOB.loc[(Buy_LOB_index - 1), 'Price']) / 2)
                self.history_bestSell.append(self.history_midprice[self.time_step])
                # 因为Buy都被消耗完，因此buy的报价不能反应市场变化，所以用sell的价格来确定buy的价格
                self.history_bestBuy.append(self.history_midprice[self.time_step])
                self.history_tradePrice.append(self.history_midprice[self.time_step])
            else:
                self.history_midprice.append((self.temp_Sell_LOB.loc[Sell_LOB_index, 'Price'] +
                                              self.temp_Buy_LOB.loc[Buy_LOB_index, 'Price']) / 2)
                self.history_bestSell.append(self.temp_Sell_LOB.loc[Sell_LOB_index, 'Price'])
                self.history_bestBuy.append(self.temp_Buy_LOB.loc[Buy_LOB_index, 'Price'])
                self.history_tradePrice.append(self.history_midprice[self.time_step])
        else:
            if self.temp_Buy_LOB.empty and not self.temp_Sell_LOB.empty:
                self.history_midprice.append(self.history_midprice[self.time_step - 1] * 0.999)
                self.history_bestSell.append(self.temp_Sell_LOB.loc[0, 'Price'])
                self.history_bestBuy.append(0)
                self.history_tradePrice.append(0)
            elif self.temp_Sell_LOB.empty and not self.temp_Buy_LOB.empty:
                self.history_midprice.append(self.history_midprice[self.time_step - 1] * 1.001)
                self.history_bestBuy.append(self.temp_Buy_LOB.loc[0, 'Price'])
                self.history_bestSell.append(0)
                self.history_tradePrice.append(0)
            elif self.temp_Buy_LOB.empty and self.temp_Sell_LOB.empty:
                self.history_midprice.append(self.history_midprice[self.time_step - 1])
                self.history_bestBuy.append(0)
                self.history_bestSell.append(0)
                self.history_tradePrice.append(0)
            else:
                self.history_midprice.append((self.temp_Buy_LOB.loc[Buy_LOB_index, 'Price'] +
                                              self.temp_Sell_LOB.loc[Sell_LOB_index, 'Price']) / 2)
                self.history_bestSell.append(self.temp_Sell_LOB.loc[Sell_LOB_index, 'Price'])
                self.history_bestBuy.append(self.temp_Buy_LOB.loc[Buy_LOB_index, 'Price'])
                self.history_tradePrice.append(0)
        #print("已更新所有history价格")

        #更新match order的agent状态；Name分别是FCN，Norm和RL
        if self.Matched_Sell is not None:
            for i in range(self.Matched_Sell.shape[0]):  # 针对Sell LOB
                if 'Norm' in self.Matched_Sell.loc[i, 'Name']:
                    match_index = int(self.Matched_Sell.loc[i, 'Name'].split('Norm')[1])
                    # 只需修改Agent的Inventory,cash
                    # Quote包含属性：Name,Price,Volume,Time
                    self.Norm_Agent.loc[match_index, 'Inventory'] -= self.Matched_Sell.loc[i, 'Volume']
                    self.Norm_Agent.loc[match_index, 'Cash'] += self.Matched_Sell.loc[i, 'Volume'] * \
                                                                self.history_midprice[
                                                                    self.time_step]
                elif 'FCN' in self.Matched_Sell.loc[i, 'Name']:
                    match_index = int(self.Matched_Sell.loc[i, 'Name'].split('FCN')[1])
                    self.FCN_Agent.loc[match_index, 'Inventory'] -= self.Matched_Sell.loc[i, 'Volume']
                    self.FCN_Agent.loc[match_index, 'Cash'] += self.Matched_Sell.loc[i, 'Volume'] * \
                                                               self.history_midprice[
                                                                   self.time_step]
                elif 'RL' in self.Matched_Sell.loc[i, 'Name']:
                    self.RL_Agent['Inventory'] -= self.Matched_Sell.loc[i, 'Volume']
                    self.RL_Agent['Cash'] += self.Matched_Sell.loc[i, 'Volume'] * self.history_midprice[
                        self.time_step]
                elif 'two' in self.Matched_Sell.loc[i, 'Name']:
                    self.RL_Agent_two['Inventory'] -= self.Matched_Sell.loc[i, 'Volume']
                    self.RL_Agent_two['Cash'] += self.Matched_Sell.loc[i, 'Volume'] * self.history_midprice[
                        self.time_step]
            for i in range(self.Matched_Buy.shape[0]):  # 针对Buy LOB
                if 'Norm' in self.Matched_Buy.loc[i, 'Name']:
                    match_index = int(self.Matched_Buy.loc[i, 'Name'].split('Norm')[1])
                    # Norm_Agent属性：Inventory,Prediction,cash,k
                    # Quote属性：Name,Price,Volume,Time
                    self.Norm_Agent.loc[match_index, 'Inventory'] += self.Matched_Buy.loc[i, 'Volume']
                    self.Norm_Agent.loc[match_index, 'Cash'] -= self.Matched_Buy.loc[i, 'Volume'] * \
                                                                self.history_midprice[
                                                                    self.time_step]
                elif 'FCN' in self.Matched_Buy.loc[i, 'Name']:
                    match_index = int(self.Matched_Buy.loc[i, 'Name'].split('FCN')[1])
                    # FCN_Agent属性：Inventory,cash
                    # Quote属性：Name,Price,Volume,Time
                    self.FCN_Agent.loc[match_index, 'Inventory'] += self.Matched_Buy.loc[i, 'Volume']
                    self.FCN_Agent.loc[match_index, 'Cash'] -= self.Matched_Buy.loc[i, 'Volume'] * \
                                                               self.history_midprice[
                                                                   self.time_step]
                elif 'RL' in self.Matched_Buy.loc[i, 'Name']:
                    self.RL_Agent['Inventory'] += self.Matched_Buy.loc[i, 'Volume']
                    self.RL_Agent['Cash'] -= self.Matched_Buy.loc[i, 'Volume'] * self.history_midprice[self.time_step]
                elif 'two' in self.Matched_Buy.loc[i, 'Name']:
                    self.RL_Agent_two['Inventory'] += self.Matched_Buy.loc[i, 'Volume']
                    self.RL_Agent_two['Cash'] -= self.Matched_Buy.loc[i, 'Volume'] * self.history_midprice[self.time_step]
        else:
            pass

        # 更新RL的总资产，并赋值给history
        self.RL_Agent['Total_amount'] = self.RL_Agent['Cash'] + self.RL_Agent['Inventory'] * \
                                            self.history_midprice[self.time_step]
        self.RL_Agent_two['Total_amount'] = self.RL_Agent_two['Cash'] + self.RL_Agent_two['Inventory'] * \
                                        self.history_midprice[self.time_step]
        self.history_RLamount.append(self.RL_Agent['Total_amount'])
        #print("已更新所有agent属性")

        #删除temp LOB中已match的order
        if self.Matched_Sell is None:
            pass
        elif temp_Sell_volume == 0:
            self.temp_Buy_LOB = self.temp_Buy_LOB.drop(range(Buy_LOB_index), axis=0)
        elif temp_Buy_volume == 0:
            self.temp_Sell_LOB = self.temp_Sell_LOB.drop(range(Sell_LOB_index), axis=0)
        else:
            self.temp_Buy_LOB = self.temp_Buy_LOB.drop(range(Buy_LOB_index), axis=0)
            self.temp_Sell_LOB = self.temp_Sell_LOB.drop(range(Sell_LOB_index), axis=0)

        # 更新index
        self.temp_Sell_LOB.index = range(self.temp_Sell_LOB.shape[0])
        self.temp_Buy_LOB.index = range(self.temp_Buy_LOB.shape[0])

        #如果等待时间过长，order会被取消
        for i in range(self.temp_Buy_LOB.shape[0]):
            if self.time_step - self.temp_Buy_LOB.loc[i, 'Time'] > 750:
                self.temp_Buy_LOB = self.temp_Buy_LOB.drop(i,axis = 0)
            else:
                pass
        for i in range(self.temp_Sell_LOB.shape[0]):
            if self.time_step - self.temp_Sell_LOB.loc[i, 'Time'] > 750:
                self.temp_Sell_LOB = self.temp_Sell_LOB.drop(i,axis = 0)
            else:
                pass
        self.Buy_LOB = self.temp_Buy_LOB
        self.Sell_LOB = self.temp_Sell_LOB
        #print("已产生新的正式LOB")

        #判断是否到达最大time step
        if self.time_step == self.max_step:
            observation = self._get_obs()
            info = self._get_info()
            """
            reward = self.history_RLamount[self.time_step] - self.history_RLamount[self.time_step - 1]
            if self.RL_Agent['Inventory'] < 0:
                reward += self.RL_Agent["Inventory"] * 2*self.history_midprice[self.time_step]
            else:
                reward -= self.RL_Agent["Inventory"] * self.history_midprice[self.time_step]/2
            if self.RL_Agent['Cash'] < 0:
                reward += self.RL_Agent["Cash"]
            reward_scale = reward / np.mean([num * num for num in self.history_reward] + [(reward ** 2)] + [1e-08])
            self.history_reward.append(reward_scale)
            """
            reward_scale = 0
            done = True
            print("%r th episode Done" % self.eps)
            self.reset()
            return observation, reward_scale, done, info
        else:
            observation = self._get_obs()
            info = self._get_info()
            """
            reward = self.history_RLamount[self.time_step] - self.history_RLamount[self.time_step - 1]
            if self.RL_Agent['Inventory'] < 0:
                reward += self.RL_Agent["Inventory"] * 2*self.history_midprice[self.time_step]
            else:
                reward -= self.RL_Agent["Inventory"] * self.history_midprice[self.time_step]/2
            if self.RL_Agent['Cash'] < 0:
                reward += self.RL_Agent["Cash"]
            reward_scale = reward / np.mean([num * num for num in self.history_reward] + [(reward ** 2)] + [1e-08])
            self.history_reward.append(reward_scale)
            """
            reward_scale = 0
            done = False
            #print('一个时间步结束')
            return observation, reward_scale, done, info


    def reset(self, return_info=False):
        self.eps += 1

        self.start_price = np.random.normal(loc = self.start_price_mu, scale = self.start_price_std)

        # 生成LOB，Buy和Sell分开；包含agent名，价格，volume，时间；长度20
        self.Buy_LOB = self.Sell_LOB = pd.DataFrame({
            'Name': [],
            'Price': [],
            'Volume': [],
            'Time': []
        })

        # 生成普通agent；不同agent之间的区分只是参数；因此用一个df来储存agent参数；Inventory，Prediction，Cash，参数
        self.Norm_Agent = pd.DataFrame({
            "Inventory": np.random.randint(low=0, high=10, size=self.Norm_Agent_num),
            "Prediction": [0] * self.Norm_Agent_num,  # 对当前价格的预测
            "Cash": np.random.normal(loc=30000, scale=100, size=self.Norm_Agent_num),
            'k': np.random.uniform(low=0, high=0.05, size=self.Norm_Agent_num),
        })
        self.sigma = np.random.normal(loc=0.3, scale=0.03, size=3)
        self.FCN_Agent = pd.DataFrame({
            "Inventory": np.random.randint(low=0, high=10, size=self.FCN_Agent_num),
            "Cash": np.random.normal(loc=30000, scale=100, size=self.FCN_Agent_num),
            "Fundam_price": [self.start_price] * self.FCN_Agent_num,
            "Weights1": np.random.exponential(scale=self.sigma[0], size=self.FCN_Agent_num),
            "Weights2": np.random.exponential(scale=self.sigma[1], size=self.FCN_Agent_num),
            "Weights3": np.random.exponential(scale=self.sigma[2], size=self.FCN_Agent_num),
            'k': np.random.uniform(low=0, high=0.05, size=self.FCN_Agent_num),
            'r': [0] * self.FCN_Agent_num,
            "Prediction": [0] * self.FCN_Agent_num,
            "Time window": np.random.randint(500,1000,size = self.FCN_Agent_num),
        })

        # 生成强化学习agent信息
        self.RL_Agent = pd.Series({
            "Inventory": np.random.randint(low=0, high=10),
            "Cash": 35000,
            "Total_amount": []
        })

        # 生成测试agent
        self.RL_Agent_two = pd.Series({
            "Inventory": np.random.randint(low=0, high=10),
            "Cash": 35000,
            "Total_amount": []
        })
        self.RL_Agent['Total_amount'] = self.RL_Agent['Inventory'] * self.start_price + self.RL_Agent['Cash']
        self.RL_Agent_two['Total_amount'] = self.RL_Agent_two['Inventory'] * self.start_price + self.RL_Agent_two['Cash']

        # 记录price Series四个特征：tradePrice, middle_price, bestAsk, bestSell
        # 每一次trade都是按照当时的mid_price决定的
        self.history_midprice = [self.start_price]
        self.history_tradePrice = [self.start_price]
        self.history_bestSell = [self.start_price]
        self.history_bestBuy = [self.start_price]
        self.history_RLamount = [self.RL_Agent['Total_amount']]
        self.history_reward = [0]

        self.time_step = 0
        self.GBM_mu = np.random.uniform(-0.00001,0.00001)  # GBM 均值
        ###################################### 这个功能后续要改到主函数中 ###############################################
        print("%r th episode begin" % self.eps)

    def close(self):
        pass

    def _get_info(self):
        return {"Buy_LOB": self.Buy_LOB, "Sell_LOB": self.Sell_LOB}

    def _get_obs(self):
        # 每次的obs包含三个值：每次的mid，trade，bs和bb；orderbook，agent信息
        # 每隔10步，取一次features；总共取20次；RL在200步开始才会有行动
        obs_mid = self.history_midprice[(self.time_step - 380):(self.time_step + 1):20]
        obs_trade = self.history_tradePrice[(self.time_step - 380):(self.time_step + 1):20]
        obs_bestS = self.history_bestSell[(self.time_step - 380):(self.time_step + 1):20]
        obs_bestB = self.history_bestBuy[(self.time_step - 380):(self.time_step + 1):20]
        # 输出一个shape为（4，seq）的array
        obs_series = [obs_mid, obs_trade, obs_bestB, obs_bestS]
        # orderbook的长度没有限制，但每次输入RL的只有前20个order;卖出的order的volume是负值。
        # LOB的特征：Name，Price，Volume，Time
        adj_Sell_LOB = self.Sell_LOB.copy()
        adj_Sell_LOB['Volume'] = -adj_Sell_LOB['Volume']
        merge_LOB = pd.concat([adj_Sell_LOB,self.Buy_LOB])[['Price', 'Volume']]
        if merge_LOB.shape[0] < 20:
            adj_orderbook = merge_LOB.values.tolist()
            while len(adj_orderbook) != 20:
                adj_orderbook.append([0, 0])
        else:
            if adj_Sell_LOB.shape[0] < 10 and self.Buy_LOB.shape[0] > 10:
                adj_orderbook = pd.concat([adj_Sell_LOB, self.Buy_LOB])[['Price', 'Volume']][0:20].values.tolist()
            elif adj_Sell_LOB.shape[0] > 10 and self.Buy_LOB.shape[0] < 10:
                adj_orderbook = pd.concat([adj_Sell_LOB, self.Buy_LOB])[['Price', 'Volume']][
                                (adj_Sell_LOB.shape[0] - 20 + self.Buy_LOB.shape[0]):].values.tolist()
            else:
                adj_orderbook = pd.concat([adj_Sell_LOB, self.Buy_LOB])[['Price', 'Volume']][
                                (adj_Sell_LOB.shape[0] - 10):(adj_Sell_LOB.shape[0] + 10)].values.tolist()
        obs_orderbook = [adj_orderbook]
        obs_agent = self.RL_Agent[['Inventory', 'Cash']].tolist()
        return [obs_series, obs_orderbook, obs_agent]









