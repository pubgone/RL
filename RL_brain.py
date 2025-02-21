# import torch
# from torch import nn
# import torch.nn.functional as F
# import numpy as np
# import collections
# import random
#
# # --------------------------------------- #
# # 经验回放池
# # --------------------------------------- #
#
# class ReplayBuffer():
#     def __init__(self, capacity):
#         # 创建一个先进先出的队列，最大长度为capacity，保证经验池的样本量不变
#         self.buffer = collections.deque(maxlen=capacity)
#     # 将数据以元组形式添加进经验池
#     def add(self, state, action, reward, next_state, done):
#         self.buffer.append((state, action, reward, next_state, done))
#     # 随机采样batch_size行数据
#     def sample(self, batch_size):
#         transitions = random.sample(self.buffer, batch_size)  # list, len=32
#         # *transitions代表取出列表中的值，即32项
#         state, action, reward, next_state, done = zip(*transitions)
#         return np.array(state), action, reward, np.array(next_state), done
#     # 目前队列长度
#     def size(self):
#         return len(self.buffer)
#
# # -------------------------------------- #
# # 构造深度学习网络，输入状态s，得到各个动作的reward
# # -------------------------------------- #
#
# class Net(nn.Module):
#     # 构造只有一个隐含层的网络
#     def __init__(self, n_states, n_hidden, n_actions):
#         super(Net, self).__init__()
#         # [b,n_states]-->[b,n_hidden]
#         self.fc1 = nn.Linear(n_states, n_hidden)
#         # [b,n_hidden]-->[b,n_actions]
#         self.fc2 = nn.Linear(n_hidden, n_actions)
#     # 前传
#     def forward(self, x):  # [b,n_states]
#         x = self.fc1(x)
#         x = self.fc2(x)
#         return x
#
# # -------------------------------------- #
# # 构造深度强化学习模型
# # -------------------------------------- #
#
# class DQN:
#     #（1）初始化
#     def __init__(self, n_states, n_hidden, n_actions,
#                  learning_rate, gamma, epsilon,
#                  target_update, device):
#         # 属性分配
#         self.n_states = n_states  # 状态的特征数
#         self.n_hidden = n_hidden  # 隐含层个数
#         self.n_actions = n_actions  # 动作数
#         self.learning_rate = learning_rate  # 训练时的学习率
#         self.gamma = gamma  # 折扣因子，对下一状态的回报的缩放
#         self.epsilon = epsilon  # 贪婪策略，有1-epsilon的概率探索
#         self.target_update = target_update  # 目标网络的参数的更新频率
#         self.device = device  # 在GPU计算
#         # 计数器，记录迭代次数
#         self.count = 0
#
#         # 构建2个神经网络，相同的结构，不同的参数
#         # 实例化训练网络  [b,4]-->[b,2]  输出动作对应的奖励
#         self.q_net = Net(self.n_states, self.n_hidden, self.n_actions)
#         # 实例化目标网络
#         self.target_q_net = Net(self.n_states, self.n_hidden, self.n_actions)
#
#         # 优化器，更新训练网络的参数
#         self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.learning_rate)
#
#     #（2）动作选择
#     def take_action(self, state):
#         # 维度扩充，给行增加一个维度，并转换为张量shape=[1,4]
#         state = torch.Tensor(state[np.newaxis, :])
#         # 如果小于该值就取最大的值对应的索引
#         if np.random.random() < self.epsilon:  # 0-1
#             # 前向传播获取该状态对应的动作的reward
#             actions_value = self.q_net(state)
#             # 获取reward最大值对应的动作索引
#             action = actions_value.argmax().item()  # int
#         # 如果大于该值就随机探索
#         else:
#             # 随机选择一个动作
#             action = np.random.randint(self.n_actions)
#         return action
#
#     #（3）网络训练
#     def update(self, transition_dict):  # 传入经验池中的batch个样本
#         # 获取当前时刻的状态 array_shape=[b,4]
#         states = torch.tensor(transition_dict['states'], dtype=torch.float)
#         # 获取当前时刻采取的动作 tuple_shape=[b]，维度扩充 [b,1]
#         actions = torch.tensor(transition_dict['actions']).view(-1,1)
#         # 当前状态下采取动作后得到的奖励 tuple=[b]，维度扩充 [b,1]
#         rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1)
#         # 下一时刻的状态 array_shape=[b,4]
#         next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float)
#         # 是否到达目标 tuple_shape=[b]，维度变换[b,1]
#         dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1)
#
#         # 输入当前状态，得到采取各运动得到的奖励 [b,4]==>[b,2]==>[b,1]
#         # 根据actions索引在训练网络的输出的第1维度上获取对应索引的q值（state_value）
#         q_values = self.q_net(states).gather(1, actions)  # [b,1]
#         # 下一时刻的状态[b,4]-->目标网络输出下一时刻对应的动作q值[b,2]-->
#         # 选出下个状态采取的动作中最大的q值[b]-->维度调整[b,1]
#         max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1,1)
#         # 目标网络输出的当前状态的q(state_value)：即时奖励+折扣因子*下个时刻的最大回报
#         q_targets = rewards + self.gamma * max_next_q_values * (1-dones)
#
#         # 目标网络和训练网络之间的均方误差损失
#         dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
#         # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
#         self.optimizer.zero_grad()
#         # 反向传播参数更新
#         dqn_loss.backward()
#         # 对训练网络更新
#         self.optimizer.step()
#
#         # 在一段时间后更新目标网络的参数
#         if self.count % self.target_update == 0:
#             # 将目标网络的参数替换成训练网络的参数
#             self.target_q_net.load_state_dict(
#                 self.q_net.state_dict())
#
#         self.count += 1
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(2)


class Network(nn.Module):
    """
    Network Structure
    """
    def __init__(self,
                 n_features,
                 n_actions,
                 n_neuron=10
                 ):
        super(Network, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=n_features, out_features=n_neuron, bias=True),
            nn.Linear(in_features=n_neuron, out_features=n_actions, bias=True),
            nn.ReLU()
        )

    def forward(self, s):
        """

        :param s: s
        :return: q
        """
        q = self.net(s)
        return q


class DeepQNetwork(nn.Module):
    """
    Q Learning Algorithm
    """
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate=0.01,
                 reward_decay=0.9,
                 e_greedy=0.9,
                 replace_target_iter=300,
                 memory_size=500,
                 batch_size=32,
                 e_greedy_increment=None):
        super(DeepQNetwork, self).__init__()

        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        # 这里用pd.DataFrame创建的表格作为memory
        # 表格的行数是memory的大小，也就是transition的个数
        # 表格的列数是transition的长度，一个transition包含[s, a, r, s_]，其中a和r分别是一个数字，s和s_的长度分别是n_features
        self.memory = pd.DataFrame(np.zeros((self.memory_size, self.n_features*2+2)))

        # build two network: eval_net and target_net
        self.eval_net = Network(n_features=self.n_features, n_actions=self.n_actions)
        self.target_net = Network(n_features=self.n_features, n_actions=self.n_actions)
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=self.lr)

        # 记录每一步的误差
        self.cost_his = []


    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            # hasattr用于判断对象是否包含对应的属性。
            self.memory_counter = 0

        transition = np.hstack((s, [a,r], s_))

        # replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory.iloc[index, :] = transition

        self.memory_counter += 1

    def choose_action(self, observation):
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            s = torch.FloatTensor(observation)
            actions_value = self.eval_net(s)
            action = [np.argmax(actions_value.detach().numpy())][0]
        else:
            action = np.random.randint(0, self.n_actions)
        return action

    def _replace_target_params(self):
        # 复制网络参数
        self.target_net.load_state_dict(self.eval_net.state_dict())

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            print('\ntarget params replaced\n')

        # sample batch memory from all memory
        batch_memory = self.memory.sample(self.batch_size) \
            if self.memory_counter > self.memory_size \
            else self.memory.iloc[:self.memory_counter].sample(self.batch_size, replace=True)

        # run the nextwork
        s = torch.FloatTensor(batch_memory.iloc[:, :self.n_features].values)
        s_ = torch.FloatTensor(batch_memory.iloc[:, -self.n_features:].values)
        q_eval = self.eval_net(s)
        q_next = self.target_net(s_)

        # change q_target w.r.t q_eval's action
        q_target = q_eval.clone()

        # 更新值
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory.iloc[:, self.n_features].values.astype(int)
        reward = batch_memory.iloc[:, self.n_features + 1].values

        q_target[batch_index, eval_act_index] = torch.FloatTensor(reward) + self.gamma * q_next.max(dim=1).values

        # train eval network
        loss = self.loss_function(q_target, q_eval)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.cost_his.append(loss.detach().numpy())

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.figure()
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.show()
