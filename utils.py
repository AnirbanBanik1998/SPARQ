import sys
import numpy as np
import random as rand
from numpy.random import uniform, exponential, normal
import math
import time
import matplotlib.pyplot as plt

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
        d_d2d_users[i][0][0] = d_cell_users[0][0]
        d_d2d_users[i][0][1] = d_cell_users[0][1]
        d_d2d_users[i][1][0] = uniform(0, cell.d2d_radius)
        d_d2d_users[i][1][1] = uniform(0, 360)
        d_cell_users.remove(d_cell_users[0])

    # Initializing Shadow fading for both cellular and d2d
    shadow_fading_d2d = normal(0, 12, cell.d2d_pairs)
    shadow_fading_cell = normal(0, 10, cell.cell_users)
        
    for i in range(cell.cell_users):
        ue = Cellular_UE(i+1, d_cell_users[i], cell.cell_radius, 
                cell.del_cell_rad, cell.del_theta, shadow_fading_cell[i])
        #ue.shadow_fading = shadow_fading_cell[i] #Assigning S.F. to UEs
        cell.cellular_list.append(ue)
        
    for i in range(cell.d2d_pairs):
        d2d = D2D(i+1, d_d2d_users[i][0], d_d2d_users[i][1], 
                cell.cell_radius, cell.d2d_radius, cell.del_cell_rad, 
                cell.del_d2d_rad, cell.del_theta, shadow_fading_d2d[i])
        #d2d.shadow_fading = shadow_fading_d2d[i] #Assigning S.F. to D2Ds
        cell.d2d_list.append(d2d)
        
    return cell


def init_channels(cell):

    #Initialize the channels
    for i in range(cell.channels):
        ch = Channel(i+1)
        if (i+1) <= cell.cell_users:
            ch.cell = ch.id
        cell.channel_list.append(ch)

    return cell

def allocate(cell):
    i = 0
    for ch in cell.channel_list:
        ch.d2d = ch.id
        i += 1
    while i < len(cell.d2d_list):
        cell.unallocated_d2d.put(i+1)
        i += 1

    return cell
        
def swap(cell, shared_channels, dedicated_channels, iteration):

    if iteration is not 1:
        min_reward = shared_channels[0].reward 
        shared_location = 0
        for i in range(len(shared_channels)):
            if min_reward >= shared_channels[i].reward:
                shared_location = i
                min_reward = shared_channels[i].reward

        min_throughput = dedicated_channels[0].throughput 
        dedicated_location = 0
        for i in range(len(dedicated_channels)):
            if min_throughput >= dedicated_channels[i].throughput:
                dedicated_location = i
                min_throughput = dedicated_channels[i].throughput

        cell.unallocated_d2d.put(shared_channels[shared_location].d2d)
        print('D2D {} of Shared Channel {} Unallocated'.format(shared_channels[shared_location].d2d, 
                shared_channels[shared_location].id))

        shared_channels[shared_location].d2d = dedicated_channels[dedicated_location].d2d 
        print('D2D {} of Dedicated Channel {} to Shared Channel {}'.format(dedicated_channels[dedicated_location].d2d, 
                dedicated_channels[dedicated_location].id, 
                shared_channels[shared_location].id))

        dedicated_channels[dedicated_location].d2d = cell.unallocated_d2d.get()
        print('Unallocated D2D {} to Dedicated Channel {}'.format(dedicated_channels[dedicated_location].d2d, 
                dedicated_channels[dedicated_location].id))

        cell.channel_list[shared_channels[shared_location].id - 1] = shared_channels[shared_location]
        cell.channel_list[dedicated_channels[dedicated_location].id - 1] = dedicated_channels[dedicated_location]

    return cell, shared_channels, dedicated_channels


def swap_new(cell, shared_channels, dedicated_channels, iteration):

    if iteration is not 1:
        min_reward = shared_channels[0].reward
        shared_location = None
        for i in range(len(shared_channels)):
            print('Reward for channel {} is {}'.format(shared_channels[i].id, 
                    shared_channels[i].reward))
            if int(shared_channels[i].reward) == -1:
                shared_location = i 
                break
            if int(shared_channels[i].reward) < 1000:
                if min_reward >= shared_channels[i].reward:
                    shared_location = i
                    min_reward = shared_channels[i].reward

        min_throughput = dedicated_channels[0].throughput 
        dedicated_location = 0
        for i in range(len(dedicated_channels)):
            if min_throughput >= dedicated_channels[i].throughput:
                dedicated_location = i
                min_throughput = dedicated_channels[i].throughput

        if shared_location is not None:
            print(str(shared_location)+'\n')
            cell.unallocated_d2d.put(shared_channels[shared_location].d2d)
            print('D2D {} of Shared Channel {} Unallocated'.format(shared_channels[shared_location].d2d, 
                    shared_channels[shared_location].id))

            shared_channels[shared_location].d2d = dedicated_channels[dedicated_location].d2d 
            shared_channels[shared_location].reward = None
            print('D2D {} of Dedicated Channel {} to Shared Channel {}'.format(dedicated_channels[dedicated_location].d2d, 
                    dedicated_channels[dedicated_location].id, 
                    shared_channels[shared_location].id))

            dedicated_channels[dedicated_location].d2d = cell.unallocated_d2d.get()
            print('Unallocated D2D {} to Dedicated Channel {}'.format(dedicated_channels[dedicated_location].d2d, 
                    dedicated_channels[dedicated_location].id))

            cell.channel_list[shared_channels[shared_location].id - 1] = shared_channels[shared_location]

        else:

            cell.unallocated_d2d.put(dedicated_channels[dedicated_location].d2d)
            print('D2D {} of Dedicated Channel {} Unallocated'.format(dedicated_channels[dedicated_location].d2d, 
                    dedicated_channels[dedicated_location].id))

            dedicated_channels[dedicated_location].d2d = cell.unallocated_d2d.get()
            print('Unallocated D2D {} to Dedicated Channel {}'.format(dedicated_channels[dedicated_location].d2d, 
                    dedicated_channels[dedicated_location].id))

        cell.channel_list[dedicated_channels[dedicated_location].id - 1] = dedicated_channels[dedicated_location]

    return cell, shared_channels, dedicated_channels