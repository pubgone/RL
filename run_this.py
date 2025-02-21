# import gym
# from RL_brain import DQN, ReplayBuffer
# import torch
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import numpy
#
# # GPU运算
# device = torch.device("cuda") if torch.cuda.is_available() \
#     else torch.device("cpu")
#
# # ------------------------------- #
# # 全局变量
# # ------------------------------- #
#
# capacity = 500  # 经验池容量
# lr = 2e-3  # 学习率
# gamma = 0.9  # 折扣因子
# epsilon = 0.9  # 贪心系数
# target_update = 200  # 目标网络的参数的更新频率
# batch_size = 32
# n_hidden = 128  # 隐含层神经元个数
# min_size = 200  # 经验池超过200后再训练
# return_list = []  # 记录每个回合的回报
#
# # 加载环境
# env = gym.make("CartPole-v1", render_mode="human")  # 使用 render_mode="human" 渲染环境
# n_states = env.observation_space.shape[0]  # 4
# n_actions = env.action_space.n  # 2
#
# # 实例化经验池
# replay_buffer = ReplayBuffer(capacity)
# # 实例化DQN
# agent = DQN(n_states=n_states,
#             n_hidden=n_hidden,
#             n_actions=n_actions,
#             learning_rate=lr,
#             gamma=gamma,
#             epsilon=epsilon,
#             target_update=target_update,
#             device=device,
#             )
# # 训练模型
# # 将进度条的逻辑放在外层循环中
# i=0
# with tqdm(total=100, desc='Training Progress') as pbar:
#     for i in range(100):  # 100回合
#         # 每个回合开始前重置环境
#         state, info = env.reset()
#         # 记录每个回合的回报
#         episode_return = 0
#         done = False
#
#         while not done:
#             # 获取当前状态下需要采取的动作
#             action = agent.take_action(state)
#             # 更新环境
#             next_state, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated  # 修改为使用 terminated 和 truncated
#             # 添加经验池
#             replay_buffer.add(state, action, reward, next_state, done)
#             # 更新当前状态
#             state = next_state
#             # 更新回合回报
#             episode_return += reward
#
#             # 当经验池超过一定数量后，训练网络
#             if replay_buffer.size() > min_size:
#                 # 从经验池中随机抽样作为训练集
#                 s, a, r, ns, d = replay_buffer.sample(batch_size)
#                 # 构造训练集
#                 transition_dict = {
#                     'states': s,
#                     'actions': a,
#                     'next_states': ns,
#                     'rewards': r,
#                     'dones': d,
#                 }
#                 # 网络更新
#                 agent.update(transition_dict)
#             # 找到目标就结束
#             if done: break
#
#         # 记录每个回合的回报
#         return_list.append(episode_return)
#
#         # 更新进度条信息
#         pbar.set_postfix({
#             'return': '%.3f' % return_list[-1]
#         })
#         pbar.update(1)  # 每个回合结束后更新进度条
# # # 训练模型
# # for i in range(100):  # 100回合
# #     # 每个回合开始前重置环境
# #     state, info = env.reset()  # 修改为接收两个返回值
# #     # 记录每个回合的回报
# #     episode_return = 0
# #     done = False
# #
# #     # 打印训练进度，一共10回合
# #     with tqdm(total=10, desc='Iteration %d' % i) as pbar:
# #
# #         while True:
# #             # 获取当前状态下需要采取的动作
# #             action = agent.take_action(state)
# #             # 更新环境
# #             next_state, reward, terminated, truncated, info = env.step(action)
# #             done = terminated or truncated  # 修改为使用 terminated 和 truncated
# #             # 添加经验池
# #             replay_buffer.add(state, action, reward, next_state, done)
# #             # 更新当前状态
# #             state = next_state
# #             # 更新回合回报
# #             episode_return += reward
# #
# #             # 当经验池超过一定数量后，训练网络
# #             if replay_buffer.size() > min_size:
# #                 # 从经验池中随机抽样作为训练集
# #                 s, a, r, ns, d = replay_buffer.sample(batch_size)
# #                 # 构造训练集
# #                 transition_dict = {
# #                     'states': s,
# #                     'actions': a,
# #                     'next_states': ns,
# #                     'rewards': r,
# #                     'dones': d,
# #                 }
# #                 # 网络更新
# #                 agent.update(transition_dict)
# #             # 找到目标就结束
# #             if done: break
# #
# #         # 记录每个回合的回报
# #         return_list.append(episode_return)
# #
# #         # 更新进度条信息
# #         pbar.set_postfix({
# #             'return': '%.3f' % return_list[-1]
# #         })
# #         pbar.update(1)
#
# # 绘图
# episodes_list = list(range(len(return_list)))
# plt.plot(episodes_list, return_list)
# plt.xlabel('Episodes')
# plt.ylabel('Returns')
# plt.title('DQN Returns')
# plt.show()
from maze_env import Maze
from RL_brain import DeepQNetwork

def run_maze():
    step = 0  # 为了记录走到第几步，记忆录中积累经验（也就是积累一些transition）之后再开始学习
    for episode in range(200):
        # initial observation
        observation = env.reset()

        while True:
            # refresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # !! restore transition
            RL.store_transition(observation, action, reward, observation_)

            # 超过200条transition之后每隔5步学习一次
            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print("game over")
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000)
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()