import sys
import numpy as np
import random as rand
from numpy.random import uniform, exponential, normal
import math
import time
import matplotlib.pyplot as plt

from utils import initial_user_locations, init_channels, allocate, swap, swap_new
from q_learn import learn, learn_dumb
from config import Cell_Model, D2D, Cellular_UE, Channel
from display import single_display, double_display



'''
In compute we execute the algorithm.
Firstly in every iteration, we perform the swapping.
Then We perform the learning, and use this learned power for the rest time
'''
def compute(cell, action_size, iterations, swap_func):

    shared_channels = []
    dedicated_channels = []
    for ch in cell.channel_list:
        if ch.cell is not None:
            shared_channels.append(ch)
        else:
            dedicated_channels.append(ch)

    throughput_list = []
    iterations_list = []

    for iteration in range(iterations):

        print('\nIteration {}\n'.format(iteration + 1))
        total_throughput = 0
        iterations_list.append(iteration + 1)

        # Swapping operations Performed
        cell, shared_channels, dedicated_channels = swap_func(cell,
                shared_channels, dedicated_channels, iteration + 1)
        '''
        cell, shared_channels, dedicated_channels = swap_new(cell, 
                shared_channels, dedicated_channels, iteration + 1)
        '''
        for channel in shared_channels:

            print('Channel No. ={}'.format(channel.id))
            print("Cell no. =", channel.cell)
            print('Cellular UE location -> {}'.format(cell.cellular_list[channel.cell - 1].loc))
            print("D2D no. =", channel.d2d)
            print('D2D location -> {}, {}'.format(cell.d2d_list[channel.d2d - 1].tx,
                    cell.d2d_list[channel.d2d - 1].rv))

            # Channel Gains
            d2d_BS = cell.cellular_list[channel.cell - 1].d2d_channel_gain(cell.d2d_list[channel.d2d - 1].tx,
                    cell.d2d_list[channel.d2d - 1].shadow_fading)
            cell_BS = cell.cellular_list[channel.cell - 1].cell_channel_gain()
            d2d_d2d = cell.d2d_list[channel.d2d - 1].d2d_channel_gain()
            cell_d2d = cell.d2d_list[channel.d2d - 1].cell_channel_gain(cell.cellular_list[channel.cell - 1].loc,
                    cell.cellular_list[channel.cell - 1].shadow_fading)

            # Getting the Power list to be used during learning

            '''
            If freshness is 0 then, D2D has just arrived in channel, so 
            Start learning from scratch.
            If not then D2D has already been there in the prev iteration 
            at least, so start adaptive learning.
            '''
            if channel.freshness == 0:
                power_list = cell.d2d_list[channel.d2d - 1].get_power_list(action_size,
                        cell.cellular_list[channel.cell - 1], cell.d2d_threshold_SINR,
                        cell.noise)
            else:
                power_list = cell.d2d_list[channel.d2d - 1].update_power_list(action_size,
                        cell.cellular_list[channel.cell - 1], cell.d2d_threshold_SINR,
                        cell.noise)

            print('Channel Freshness {}\n'.format(channel.freshness))
            channel.freshness += 1

            # Learning the optimal power to transmit
            cell.d2d_list[channel.d2d - 1].power, reward = learn(power_list,
                    cell.cellular_list[channel.cell - 1],
                    cell.d2d_list[channel.d2d - 1], cell.noise,
                    cell.cell_threshold_SINR)
            '''
            learn_dumb(power_list, cell.cell_list[channel.cell - 1], 
                    cell.d2d_list[channel.d2d - 1], cell.noise, 
                    cell.cell_sinr_threshold)
            '''
            print('Cell_Power = {}, D2D Learned Power = {}\n'.format(cell.cellular_list[channel.cell - 1].power,
                    cell.d2d_list[channel.d2d - 1].power))
            channel.reward = reward
            cell_SINR = cell.cellular_list[channel.cell - 1].SINR_given_power(cell.cellular_list[channel.cell - 1].power,
                    cell.d2d_list[channel.cell - 1].power, d2d_BS,
                    cell_BS, cell.noise)
            d2d_SINR = cell.d2d_list[channel.d2d - 1].SINR_given_power(cell.d2d_list[channel.d2d - 1].power,
                    cell.cellular_list[channel.cell - 1].power, d2d_d2d,
                    cell_d2d, cell.noise)

            channel.throughput = channel.calculate_throughput(d2d_SINR,
                    cell_SINR)

            total_throughput += channel.throughput

        for channel in dedicated_channels:

            print('Channel No. ={}'.format(channel.id))
            print('D2D No. ={}'.format(channel.d2d))
            print('D2D location -> {}, {}'.format(cell.d2d_list[channel.d2d - 1].tx,
                    cell.d2d_list[channel.d2d - 1].rv))
            '''
            cell.d2d_list[channel.d2d - 1].power = np.random.random_sample() *cell.d2d_list[channel.d2d - 1].max_power
            '''

            # Channel Gain from d2d tx to rx
            d2d_d2d = cell.d2d_list[channel.d2d - 1].d2d_channel_gain()

            # Calculation of constant
            path_loss = 28 + 40*math.log10(10*50/1000)
            gain = pow(50, -4) * pow(10,
                    (cell.d2d_list[channel.d2d - 1].shadow_fading + path_loss)/10)
            constant = cell.d2d_list[channel.d2d - 1].max_power * gain

            power = max(constant/d2d_d2d,
                    cell.d2d_list[channel.d2d - 1].power_given_SINR(cell.d2d_threshold_SINR,
                            0, d2d_d2d, 0, cell.noise))
            cell.d2d_list[channel.d2d - 1].power = power
            print('D2D Power {}'.format(cell.d2d_list[channel.d2d - 1].power))

            d2d_SINR = cell.d2d_list[channel.d2d - 1].SINR_given_power(cell.d2d_list[channel.d2d - 1].power,
                    0, d2d_d2d,
                    cell_d2d, cell.noise)
            channel.throughput = channel.calculate_throughput(d2d_SINR)

            total_throughput += channel.throughput

        for cellular in cell.cellular_list:
                cellular.move()
        for d2d in cell.d2d_list:
            d2d.move()

        #total_throughput = 10 * math.log10(total_throughput) + 30
        throughput_list.append(total_throughput)

    return iterations_list, throughput_list

if __name__ == '__main__':

    simulations = 10
    iterations_per_simulation = 25
    average_throughput_list = np.zeros(iterations_per_simulation)
    for sim in range(simulations):
        print('\n Simulation {} \n'.format(sim + 1))
        cell = Cell_Model()
        if not int(sys.argv[1]):
            cell.mobility()
        else:
            cell.mobility(True)
        cell = initial_user_locations(cell)
        cell = init_channels(cell)
        cell = allocate(cell)
        iterations_list, throughput_list = compute(cell, 16,
                iterations_per_simulation, swap)
        average_throughput_list = average_throughput_list + np.array(throughput_list)

    average_throughput_list = list(average_throughput_list / simulations)


    average_new_throughput_list = np.zeros(iterations_per_simulation)
    for sim in range(simulations):
        print('\n Simulation {} \n'.format(sim + 1))
        cell = Cell_Model()
        if not int(sys.argv[1]):
            cell.mobility()
        else:
            cell.mobility(True)
        cell = initial_user_locations(cell)
        cell = init_channels(cell)
        cell = allocate(cell)
        iterations_list, throughput_list = compute(cell, 16,
                iterations_per_simulation, swap_new)
        average_new_throughput_list = average_new_throughput_list + np.array(throughput_list)

    average_new_throughput_list = list(average_new_throughput_list / simulations)


    double_display(iterations_list, average_throughput_list, 'b-',
            'Swap', iterations_list, average_new_throughput_list,
            'r-', 'Swap_new', 'Iterations', 'Throughput in Mbps')
