import sys
import numpy as np
import random as rand
from numpy.random import uniform, exponential, normal
import math
import time
import matplotlib.pyplot as plt
from copy import deepcopy

from concurrent.futures import ProcessPoolExecutor, as_completed

from core.utils import initial_user_locations, init_channels, allocate, swap, swap_new, copy_initial_data
from core.q_learn import learn, learn_dumb
from core.config import Cell_Model, D2D, Cellular_UE, Channel


def smart_power_control(channel, loc, cell, action_size, priority, caching, 
        caching_parameter):


    # Channel Gains
    d2d_BS = cell.cellular_list[channel.cell - 1].d2d_channel_gain(cell.d2d_list[channel.d2d - 1].tx, 
            cell.d2d_list[channel.d2d - 1].shadow_fading)
    cell_BS = cell.cellular_list[channel.cell - 1].cell_channel_gain()
    d2d_d2d = cell.d2d_list[channel.d2d - 1].d2d_channel_gain()
    cell_d2d = cell.d2d_list[channel.d2d - 1].cell_channel_gain(cell.cellular_list[channel.cell - 1].loc, 
            cell.cellular_list[channel.cell - 1].shadow_fading)

    # Getting the Power list to be used during learning

    if not caching:
        power_list = cell.d2d_list[channel.d2d - 1].get_power_list(action_size, 
                cell.cellular_list[channel.cell - 1], cell.d2d_threshold_SINR, 
                cell.noise)
    elif caching:
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
            power_list = cell.d2d_list[channel.d2d - 1].update_power_list(action_size, caching_parameter, 
                    cell.cellular_list[channel.cell - 1], cell.d2d_threshold_SINR, 
                    cell.noise)

    
    channel.freshness += 1

    # Learning the optimal power to transmit
    cell.d2d_list[channel.d2d - 1].power, reward = learn(power_list, 
            cell.cellular_list[channel.cell - 1], 
            cell.d2d_list[channel.d2d - 1], cell.noise, 
            cell.cell_threshold_SINR, priority)
    '''
    learn_dumb(power_list, cell.cell_list[channel.cell - 1], 
            cell.d2d_list[channel.d2d - 1], cell.noise, 
            cell.cell_sinr_threshold)
    '''
    channel.reward = reward
    cell_SINR = cell.cellular_list[channel.cell - 1].SINR_given_power(cell.cellular_list[channel.cell - 1].power, 
            cell.d2d_list[channel.d2d - 1].power, d2d_BS, 
            cell_BS, cell.noise)
    d2d_SINR = cell.d2d_list[channel.d2d - 1].SINR_given_power(cell.d2d_list[channel.d2d - 1].power, 
            cell.cellular_list[channel.cell - 1].power, d2d_d2d, 
            cell_d2d, cell.noise)

    channel.throughput = channel.calculate_throughput(d2d_SINR, 
            cell_SINR)

    return cell.d2d_list[channel.d2d - 1], channel, loc


'''
In compute we execute the algorithm.
Firstly in every iteration, we perform the swapping.
Then We perform the learning, and use this learned power for the rest time
'''
def compute(cell, action_size, iterations, swap_func, smart_PC, caching, 
        priority=1, caching_parameter=None, follow_trace=False, 
        cellular_trace=None, d2d_trace=None):

    if not follow_trace:
        cellular_trace=[[] for _ in range(iterations)]
        d2d_trace=[[] for _ in range(iterations)]

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

        #print(cellular_trace)

        # Swapping operations Performed
        if swap_func is not None:
            cell, shared_channels, dedicated_channels = swap_func(cell, 
                    shared_channels, dedicated_channels, iteration + 1)

        # Power Control in Shared D2Ds
        # Either use Smart Power Control with Q-Learning
        # Or use Open-Loop Power Control
        if smart_PC is True:
            with ProcessPoolExecutor() as executor:
                results = [executor.submit(smart_power_control, 
                        channel, loc, cell, 
                        action_size, priority, caching, 
                        caching_parameter) for loc, channel in enumerate(shared_channels)]

                for f in as_completed(results):
                    device, channel, loc = f.result()
                    cell.d2d_list[channel.d2d - 1] = device
                    shared_channels[loc] = channel
                    print('Channel No. ={}'.format(channel.id))
                    print("Cell no. =", channel.cell)
                    print('Cellular UE location -> {}'.format(cell.cellular_list[channel.cell - 1].loc))
                    print("D2D no. =", channel.d2d)
                    print('D2D location -> {}, {}'.format(cell.d2d_list[channel.d2d - 1].tx, 
                            cell.d2d_list[channel.d2d - 1].rv))
                    print('Channel Freshness {}\n'.format(channel.freshness))
                    print('Cell_Power = {}, D2D Learned Power = {}\n'.format(cell.cellular_list[channel.cell - 1].power, 
                            cell.d2d_list[channel.d2d - 1].power))

                    total_throughput += channel.throughput
        else:
            for channel in shared_channels:

                print('Channel No. ={}'.format(channel.id))
                print('D2D No. ={}'.format(channel.d2d))
                print('D2D location -> {}, {}'.format(cell.d2d_list[channel.d2d - 1].tx, 
                        cell.d2d_list[channel.d2d - 1].rv))

                # Channel Gains
                d2d_BS = cell.cellular_list[channel.cell - 1].d2d_channel_gain(cell.d2d_list[channel.d2d - 1].tx, 
                        cell.d2d_list[channel.d2d - 1].shadow_fading)
                cell_BS = cell.cellular_list[channel.cell - 1].cell_channel_gain()
                d2d_d2d = cell.d2d_list[channel.d2d - 1].d2d_channel_gain()
                cell_d2d = cell.d2d_list[channel.d2d - 1].cell_channel_gain(cell.cellular_list[channel.cell - 1].loc, 
                        cell.cellular_list[channel.cell - 1].shadow_fading)

                # Calculation of constant
                path_loss = 28 + 40*math.log10(10*50/1000)
                gain = pow(50, -4) * pow(10, 
                        (cell.d2d_list[channel.d2d - 1].shadow_fading + path_loss)/10)
                constant = cell.d2d_list[channel.d2d - 1].max_power * gain

                power = max(constant/d2d_d2d, 
                        cell.d2d_list[channel.d2d - 1].power_given_SINR(cell.d2d_threshold_SINR, 
                                cell.cellular_list[channel.cell - 1].power, 
                                    d2d_d2d, cell_d2d, cell.noise))
                cell.d2d_list[channel.d2d - 1].power = power
                print('D2D Power {}'.format(cell.d2d_list[channel.d2d - 1].power))

                cell_SINR = cell.cellular_list[channel.cell - 1].SINR_given_power(cell.cellular_list[channel.cell - 1].power, 
                        cell.d2d_list[channel.d2d - 1].power, d2d_BS, 
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
                    0, cell.noise)
            channel.throughput = channel.calculate_throughput(d2d_SINR)

            total_throughput += channel.throughput

             # Whether or not to follow tracing left by the primary one
        '''
        If Cell is primary, then it moves randomly, and leaves traces.
        If Cell is not primary, it moves according to the traces left by
        the primary cell. This helps us to get a more realistic plot, 
        as all the locations being same, for all the cases, we can compare
        the algorithms easily.
        '''

        if not follow_trace:
            for cellular in cell.cellular_list:
                print('Initial loc -> {}'.format(cellular.loc))
                cellular.loc = cellular.move()
                cellular.power = cellular.get_power(cellular.cell_channel_gain())
                print('Final loc -> {}'.format(cellular.loc))
                cellular_trace[iteration].append(cellular.loc)
            for d2d in cell.d2d_list:
                print('Initial loc -> {}, {}'.format(d2d.tx, d2d.rv))
                d2d.tx, d2d.rv = d2d.move()
                print('Final loc -> {}, {}'.format(d2d.tx, d2d.rv))
                d2d_trace[iteration].append([d2d.tx, d2d.rv])

        else:
            for cellular in cell.cellular_list:
                cellular.follow_tracing(cellular_trace[iteration][cellular.id - 1])
            for d2d in cell.d2d_list:
                d2d.follow_tracing(d2d_trace[iteration][d2d.id - 1][0], 
                        d2d_trace[iteration][d2d.id - 1][1])

        #total_throughput = 10 * math.log10(total_throughput) + 30
        throughput_list.append(total_throughput)

    return iterations_list, throughput_list, cellular_trace, d2d_trace