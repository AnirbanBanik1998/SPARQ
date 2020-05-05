import numpy as np
import random as rand
from numpy.random import uniform, exponential, normal
import math
import time

def learn(actions, cell, d2d, noise, cell_SINR_threshold, priority):
    import copy

    '''
    This Q Learning algorithm is a terminating one.
    It is a single agent Q-Learning algo.
    * Agent -> The D2D transmitter in the channel.
    * States are 0 and 1.
        0 -> Cellular QoS not satisfied.
        1 -> Cellular QoS satisfied.
    * Rewards:
        priority*log2(1+SINR(cell)) + log2(1+SINR(D2D)) -> State 1
        -1 -> State 0
    * The terminating criteria are:
        * If State 0 is reached as the final state at any step.
        * If all the power levels are exhausted. They can be used once.
    '''

    action_space_size = len(actions)
    state_space_size = 2

    q_table = np.zeros((state_space_size, action_space_size))

    # Q_Learning data

    num_episodes = 1000
    max_steps_per_episode = 100

    learning_rate = 0.5
    discount_rate = 0.9

    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.001

    # Algorithm

    rewards_all_episodes = 0
    state = 0

    # Channel Gains
    d2d_BS = cell.d2d_channel_gain(d2d.tx, d2d.shadow_fading)
    cell_BS = cell.cell_channel_gain()
    d2d_d2d = d2d.d2d_channel_gain()
    cell_d2d = d2d.cell_channel_gain(cell.loc, cell.shadow_fading)

    print('D2D Channel Gain-> {}'.format(d2d_d2d))
    print('Cell to Rx Channel Gain-> {}'.format(cell_d2d))
    print('Cell to BS Channel Gain-> {}'.format(cell_BS))
    print('D2D Tx to BS Channel Gain-> {}\n'.format(d2d_BS))

    state = 0 # Initial State in the learning process
    #priority = 2 #Priority of cellular Communication
    count = 0

    for episode in range(num_episodes):
        
        done = False #Episode terminated or not
        rewards_current_episode = 0
        actions_copy = copy.deepcopy(actions) #Deepcopy of actions list

        for step in range(max_steps_per_episode):
            #Exploration-exploitation tradeoff
            exploration_rate_threshold = uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_table[state, :]) # Exploitation
            else:
                action = np.random.randint(0, action_space_size) 
                # Exploration randomly

            # Noting which power levels have been used up
            if actions[action] in actions_copy:
                actions_copy.remove(actions[action])
            '''
            For every power level used, D2D signals the BS.
            Cellular User gets this information about the D2D Tx power 
            from BS and calculates the cellular SINR.
            It then sends this to the D2D. Now D2D calculates the state.
            And correspondingly the reward.
            '''
            cell_SINR = cell.SINR_given_power(cell.power, actions[action], 
                    d2d_BS, cell_BS, noise)
            new_state = d2d.get_cell_state(cell_SINR, 
                    cell_SINR_threshold)
            d2d_SINR = d2d.SINR_given_power(actions[action], 
                cell.power, d2d.d2d_channel_gain(), 
                d2d.cell_channel_gain(cell.loc, cell.shadow_fading), noise)

            reward = d2d.calc_reward(new_state, d2d_SINR, cell_SINR, 
                priority)

            # Update Q-table for Q(s, a)
            q_table[state, action] = q_table[state, action] * (1-learning_rate) + \
                learning_rate * (reward+discount_rate*np.max(q_table[new_state, :]))

            rewards_current_episode += reward

            '''
            Checking the Terminating conditions
            '''
            #state = new_state
            if not new_state or len(actions_copy) == 0:
                done = True
            else:
                state = new_state

            if done:
                break

        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * \
            np.exp(-exploration_decay_rate*episode)
        count += 1

        rewards_all_episodes += rewards_current_episode

    print('Average Reward {}'.format(rewards_all_episodes/num_episodes))
    print('Q-Table \n{}\n'.format(q_table))

    best_power = actions[np.argmax(q_table[1, :])]
    return best_power, rewards_all_episodes/num_episodes

def learn_dumb(actions, cell, d2d, noise, cell_SINR_threshold, priority):

    '''
    This Q Learning algorithm is a terminating one.
    It is a single agent Q-Learning algo.
    * Agent -> The D2D transmitter in the channel.
    * States are 0 and 1.
        0 -> Cellular QoS not satisfied.
        1 -> Cellular QoS satisfied.
    * Rewards:
        priority*log2(1+SINR(cell)) + log2(1+SINR(D2D)) -> State 1
        -1 -> State 0
    * The terminating criteria are:
        * If State 0 is reached as the final state at any step.
    '''

    action_space_size = len(actions)
    state_space_size = 2

    q_table = np.zeros((state_space_size, action_space_size))

    # Q_Learning data

    num_episodes = 1000
    max_steps_per_episode = 100

    learning_rate = 0.5
    discount_rate = 0.9

    exploration_rate = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.001

    # Algorithm

    rewards_all_episodes = 0
    state = 0

    # Channel Gains
    d2d_BS = cell.d2d_channel_gain(d2d.tx, d2d.shadow_fading)
    cell_BS = cell.cell_channel_gain()
    d2d_d2d = d2d.d2d_channel_gain()
    cell_d2d = d2d.cell_channel_gain(cell.loc, cell.shadow_fading)

    print('D2D Channel Gain-> {}'.format(d2d_d2d))
    print('Cell to Rx Channel Gain-> {}'.format(cell_d2d))
    print('Cell to BS Channel Gain-> {}'.format(cell_BS))
    print('D2D Tx to BS Channel Gain-> {}\n'.format(d2d_BS))

    state = 0 # Initial State in the learning process
    #priority = 2 #Priority of cellular Communication
    count = 0

    for episode in range(num_episodes):
        
        done = False #Episode terminated or not
        rewards_current_episode = 0

        for step in range(max_steps_per_episode):
            #Exploration-exploitation tradeoff
            exploration_rate_threshold = uniform(0, 1)
            if exploration_rate_threshold > exploration_rate:
                action = np.argmax(q_table[state, :]) # Exploitation
            else:
                action = np.random.randint(0, action_space_size) 
                # Exploration randomly

            '''
            For every power level used, D2D signals the BS.
            Cellular User gets this information about the D2D Tx power 
            from BS and calculates the cellular SINR.
            It then sends this to the D2D. Now D2D calculates the state.
            And correspondingly the reward.
            '''
            cell_SINR = cell.SINR_given_power(cell.power, actions[action], 
                    d2d_BS, cell_BS, noise)
            new_state = d2d.get_cell_state(cell_SINR, 
                    cell_SINR_threshold)
            d2d_SINR = d2d.SINR_given_power(actions[action], 
                cell.power, d2d.d2d_channel_gain(), 
                d2d.cell_channel_gain(cell.loc, cell.shadow_fading), noise)

            reward = d2d.calc_reward(new_state, d2d_SINR, cell_SINR, 
                priority)

            # Update Q-table for Q(s, a)
            q_table[state, action] = q_table[state, action] * (1-learning_rate) + \
                learning_rate * (reward+discount_rate*np.max(q_table[new_state, :]))

            rewards_current_episode += reward

            '''
            Checking the Terminating conditions
            '''
            #state = new_state
            if not new_state:
                done = True
            else:
                state = new_state

            if done:
                break

        exploration_rate = min_exploration_rate + \
            (max_exploration_rate - min_exploration_rate) * \
            np.exp(-exploration_decay_rate*episode)
        count += 1

        rewards_all_episodes += rewards_current_episode

    print('Average Reward {}'.format(rewards_all_episodes/num_episodes))
    print('Q-Table \n{}'.format(q_table))

    return q_table, rewards_all_episodes/num_episodes