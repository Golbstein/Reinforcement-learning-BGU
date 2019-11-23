import numpy as np


_EPISODES = 5000
_MAX_STEPS = 100
np.random.seed(12345)


def moving_average(x, n=100):
    return np.convolve(x, np.ones(n)/n, mode='valid')


def episode_weighted_reward(rewards):
    w = np.arange(len(rewards)) / np.arange(len(rewards)).sum()
    return np.sum(w * rewards)


def train_q_learning_agent(env, epsilon, alpha, discount, eps_decay, debug=True):
    
    #  init Q table with zeros
    q_table = np.zeros((env.observation_space.n, env.action_space.n))

    rewards_per_episode = np.zeros(_EPISODES)
    steps_to_goal = np.zeros(_EPISODES)
    for episode in range(_EPISODES):
        state = env.reset()
        episode_reward = 0
        
        # debug
        if episode == 500 and debug:
            q_table_500 = q_table.copy()
        if episode == 2000 and debug:
            q_table_2000 = q_table.copy()
            
            
        for step in range(_MAX_STEPS):

            if np.random.rand() > epsilon:
                action = np.argmax(q_table[state])
            else:  # random action
                action = np.random.randint(env.action_space.n)

            new_state, reward, done, _ = env.step(action)
            current_q = q_table[state, action]
            episode_reward += reward

            if done:
                target = reward
            else:
                max_future_q = q_table[new_state].max()
                target = reward + discount*max_future_q

            new_q = (1 - alpha)*current_q + alpha*target
            q_table[state, action] = new_q
            if done:
                break

            state = new_state

        
        epsilon *= eps_decay
        rewards_per_episode[episode] = episode_reward / (1 + step)
        
        if episode_reward>0:
            steps_to_goal[episode] = 1 + step
        else:
            steps_to_goal[episode] = 100
    
    if debug:
        return q_table, rewards_per_episode, steps_to_goal, q_table_500, q_table_2000
    else:
        return q_table, rewards_per_episode, steps_to_goal