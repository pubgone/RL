import gym
import numpy as np
import matplotlib.pyplot as plt


# 从观测到状态
def ob2state(observation):
    return (observation[0], observation[1], int(observation[2]))


# 回合更新策略评估
def evaluate_action_monte_carlo(env, policy, episode_num=50000):
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        # 玩一回合
        state_actions = []
        observation, _ = env.reset()
        while True:
            state = ob2state(observation)
            action = np.random.choice(env.action_space.n, p=policy[state])
            state_actions.append((state, action))
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done:
                break
        g = reward
        for state, action in state_actions:
            c[state][action] += 1.
            q[state][action] += (g - q[state][action]) / c[state][action]
    return q


# 绘制最后一维的指标为0或1的3维数组
def plot(data):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    titles = ['without ace', 'with ace']
    have_aces = [0, 1]
    extent = [12, 22, 1, 11]
    for title, have_ace, axis in zip(titles, have_aces, axes):
        dat = data[extent[0]:extent[1], extent[2]:extent[3], have_ace].T
        axis.imshow(dat, extent=extent, origin='lower')
        axis.set_xlabel('player sum')
        axis.set_ylabel('dealer showing')
        axis.set_title(title)
    plt.show()


# 带起始探索的回合更新
def monte_carlo_with_exploring_start(env, episode_num=500000):
    policy = np.zeros((22, 11, 2, 2))
    policy[:, :, :, 1] = 1.
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        # 随机选择起始状态和起始动作
        #state[0]:玩家手牌总点数
        # state[1]:庄家展示手牌
        # state[2]:手牌包含A代表True
        state = (np.random.randint(12, 22),
                 np.random.randint(1, 11),
                 np.random.randint(2))
        action = np.random.randint(2)
        # 玩一回合
        env.reset()
        if state[2]:  # 有A
            env.player = [1, state[0] - 11]
        else:  # 没有A
            if state[0] == 21:
                env.player = [10, 9, 2]
            else:
                env.player = [10, state[0] - 10]
        env.dealer[0] = state[1]#将庄家明牌点数设置为当前游戏环境中庄家亮出的实际点数
        state_actions = []
        while True:
            state_actions.append((state, action))
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done:
                break  # 回合结束
            state = ob2state(observation)
            action = np.random.choice(env.action_space.n, p=policy[state])
        g = reward  # 回报
        for state, action in state_actions:
            c[state][action] += 1.
            q[state][action] += (g - q[state][action]) / c[state][action]
            a = q[state].argmax()
            policy[state] = 0.
            policy[state][a] = 1.
    return policy, q


# 重要性采样策略评估
def evaluate_monte_carlo_importance_resample(env, policy, behavior_policy, episode_num=500000):
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        # 用行为策略玩一回合
        state_actions = []
        observation, _ = env.reset()
        while True:
            state = ob2state(observation)
            action = np.random.choice(env.action_space.n, p=behavior_policy[state])
            state_actions.append((state, action))
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done:
                break  # 玩好了
        g = reward  # 回报
        rho = 1.  # 重要性采样比率
        for state, action in reversed(state_actions):
            c[state][action] += rho
            q[state][action] += (rho / c[state][action] * (g - q[state][action]))
            rho *= (policy[state][action] / behavior_policy[state][action])
            if rho == 0:
                break  # 提前终止
    return q


# 柔性策略重要性采样最优策略求解
def monte_carlo_importance_resample(env, episode_num=500000):
    policy = np.zeros((22, 11, 2, 2))
    policy[:, :, :, 0] = 1.
    behavior_policy = np.ones_like(policy) * 0.5  # 柔性策略
    q = np.zeros_like(policy)
    c = np.zeros_like(policy)
    for _ in range(episode_num):
        # 用行为策略玩一回合
        state_actions = []
        observation, _ = env.reset()
        while True:
            state = ob2state(observation)
            action = np.random.choice(env.action_space.n, p=behavior_policy[state])
            state_actions.append((state, action))
            observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if done:
                break  # 玩好了
        g = reward  # 回报
        rho = 1.  # 重要性采样比率
        for state, action in reversed(state_actions):
            c[state][action] += rho
            q[state][action] += (rho / c[state][action] * (g - q[state][action]))
            # 策略改进
            a = q[state].argmax()
            policy[state] = 0.
            policy[state][a] = 1.
            if a != action:  # 提前终止
                break
            rho *= (policy[state][action] / behavior_policy[state][action])
    return policy, q


# 主函数
if __name__ == '__main__':
    env = gym.make("Blackjack-v1")  # 使用新版环境名
    observation, _ = env.reset()
    print('观测 = {}'.format(observation))
    while True:
        print('玩家 = {}, 庄家 = {}'.format(env.player, env.dealer))
        action = np.random.choice(env.action_space.n)
        print('动作 = {}'.format(action))
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        print('观测 = {}, 奖励 = {}, 结束指示 = {}\n'.format(observation, reward, done))

        if done:
            break
    '''
    # 定义一个确定性算法并评估
    print('评估一个确定的随机算法'.center(20, '-'))
    policy_certain = np.zeros((22, 11, 2, 2))
    policy_certain[20:, :, :, 0] = 1  # >20时不再要牌
    policy_certain[:20, :, :, 1] = 1  # <20时不再要牌
    q = evaluate_action_monte_carlo(env, policy_certain)  # 动作价值
    v = (q * policy_certain).sum(axis=-1)
    plot(v)

    # 带起始探索的更新策略
    print('带起始探索的更新策略'.center(20, '-'))
    policy_wes, q = monte_carlo_with_exploring_start(env)
    v = q.max(axis=-1)
    plot(policy_wes.argmax(-1))
    plot(v)

    '''
    print('off-policy策略评估'.center(20, '-'))
    policy = np.zeros((22, 11, 2, 2))
    policy[20:, :, :, 0] = 1  # >= 20时收手
    policy[:20, :, :, 1] = 1  # <20时继续
    behavior_policy = np.ones_like(policy) * 0.5
    q = evaluate_monte_carlo_importance_resample(env, policy, behavior_policy)
    v = (q * policy).sum(axis=-1)
    plot(v)

    policy, q = monte_carlo_importance_resample(env)
    v = q.max(axis=-1)
    plot(policy.argmax(-1))
    plot(v)