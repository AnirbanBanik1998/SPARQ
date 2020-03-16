import sys
import numpy as np
import random
from numpy.random import uniform, exponential, normal
import math
import time

from config import Cell_Model, D2D, Cellular_UE, Channel


def initial_user_locations(cell):

    d_cell_users = []
    d_d2d_users = [[[0, 0], [0, 0]] for _ in range(cell.d2d_pairs)]
    
    '''Initially we assign the locations of all the cellular users 
        as well as the d2d sender.'''
    for _ in range(cell.d2d_pairs + cell.cell_users):
        d_cell_users.append([uniform(0, cell.cell_radius), uniform(0, 360)])
    #Shuffling all the users so that the d2d users can be selected randomly
    random.shuffle(d_cell_users)
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



def display(d2d, cell, time_gap):
    while True:
        try:
            print('D2D location -> {}, {}'.format(d2d.tx, d2d.rv))
            print('Cellular UE location -> {}'.format(cell.loc))
            print('D2D Channel Gain-> {}'.format(str(d2d.d2d_channel_gain())))
            print('Cell to Rx Channel Gain-> {}'.format(str(d2d.cell_channel_gain(cell.loc, cell.shadow_fading))))
            print('Cell to BS Channel Gain-> {}'.format(str(cell.cell_channel_gain())))
            print('D2D Tx to BS Channel Gain-> {}'.format(str(cell.d2d_channel_gain(d2d.tx, d2d.shadow_fading))))
            print()
            d2d.move()
            time.sleep(time_gap)
        except KeyboardInterrupt:
            break
    
if __name__ == '__main__':
    cell = Cell_Model()
    if not int(sys.argv[1]):
        cell.mobility()
    else:
        cell.mobility(True)
    cell = initial_user_locations(cell)
    cell = init_channels(cell)
    display(cell.d2d_list[0], cell.cellular_list[0], cell.time)
