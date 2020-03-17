import sys
import numpy as np
import random as rand
from numpy.random import uniform, exponential, normal
import math
import time
import copy

from config import Cell_Model, D2D, Cellular_UE, Channel


def initial_user_locations(cell):

    d_cell_users = []
    d_d2d_users = [[[0, 0], [0, 0]] for _ in range(cell.d2d_pairs)]
    
    '''Initially we assign the locations of all the cellular users 
        as well as the d2d sender.'''
    radius_list = list(uniform(0, cell.cell_radius, 
        (cell.d2d_pairs + cell.cell_users)))
    theta_list = list(uniform(0, 360, (cell.d2d_pairs + cell.cell_users)))
    for i in range(cell.d2d_pairs + cell.cell_users):
        d_cell_users.append([radius_list[i], theta_list[i]])
    #Shuffling all the users so that the d2d users can be selected randomly
    rand.shuffle(d_cell_users)
    for i in range(cell.d2d_pairs):
        d_d2d_users[i][0][0] = d_cell_users[i][0]
        d_d2d_users[i][0][1] = d_cell_users[i][1]
        d_d2d_users[i][1][0] = uniform(0, cell.d2d_radius)
        d_d2d_users[i][1][1] = uniform(0, 360)
        d_cell_users.remove(d_cell_users[i])

    # Initializing Shadow fading for both cellular and d2d
    shadow_fading_d2d = normal(0, 12, cell.d2d_pairs)
    shadow_fading_cell = normal(0, 10, cell.cell_users)
        
    for i in range(cell.cell_users):
        ue = Cellular_UE(i+1, d_cell_users[i], cell.cell_radius, 
                cell.del_cell_rad, cell.del_theta)
        ue.shadow_fading = shadow_fading_cell[i] #Assigning S.F. to UEs
        cell.cellular_list.append(ue)
        
    for i in range(cell.d2d_pairs):
        d2d = D2D(i+1, d_d2d_users[i][0], d_d2d_users[i][1], 
                cell.cell_radius, cell.d2d_radius, cell.del_cell_rad, 
                cell.del_d2d_rad, cell.del_theta)
        d2d.shadow_fading = shadow_fading_d2d[i] #Assigning S.F. to D2Ds
        cell.d2d_list.append(d2d)
        
    return cell


def init_channels(cell):

    #Initialize the channels
    for i in range(cell.channels):
        ch = Channel(i+1)
        if (i+1) <= cell.cell_users:
            cell.cellular_list[i].channel = ch.id
            ch.cell = ch.id
        cell.channel_list.append(ch)

    return cell

def learn(actions, cell, d2d, noise, cell_SINR_threshold):

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
        priority = 2 #Priority of cellular Communication
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
                    action = np.random.randint(action_space_size) 
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
        print('Q-Table \n{}'.format(q_table))


def display(d2d_sinr_threshold, cell_sinr_threshold, noise, d2d, 
        cell, time_gap):
    max_time = 0
    while True:
        try:
            start_time = time.time()
            print('D2D location -> {}, {}'.format(d2d.tx, d2d.rv))
            print('Cellular UE location -> {}'.format(cell.loc))
            '''
            min_power = d2d.power_given_SINR(d2d_sinr_threshold, cell.power, 
                    d2d.d2d_channel_gain(), d2d.cell_channel_gain(cell.loc, 
                        cell.shadow_fading), noise)
            '''
            min_power = 0 #No QoS Requirements for D2D
            power_list = list(uniform(min_power, d2d.max_power, 16))
            #print('Power list {}'.format(power_list))
            learn(power_list, cell, d2d, noise, cell_sinr_threshold)
            d2d.move()
            cell.move()
            time_taken = time.time() - start_time
            max_time = max(max_time, time_taken)
            #time.sleep(time_gap)
            print('Time taken to learn {}'.format(time_taken))
        except KeyboardInterrupt:
            print('Max time taken = {}'.format(max_time))
            break
    
if __name__ == '__main__':
    cell = Cell_Model()
    if not int(sys.argv[1]):
        cell.mobility()
    else:
        cell.mobility(True)
    cell = initial_user_locations(cell)
    cell = init_channels(cell)
    display(cell.d2d_threshold_SINR, cell.cell_threshold_SINR, cell.noise, 
        cell.d2d_list[0], cell.cellular_list[0], cell.time)
