import sys
import numpy as np
import random as rand
from numpy.random import uniform, exponential, normal
import math
import time
import matplotlib.pyplot as plt
from copy import deepcopy

from core.utils import initial_user_locations, init_channels, allocate, swap, swap_new, copy_initial_data
from core.q_learn import learn, learn_dumb
from core.config import Cell_Model, D2D, Cellular_UE, Channel
from core.display import quad_display

from algo import compute, smart_power_control


simulations = 2
iterations_per_simulation = 5

average_throughput_list = np.zeros(iterations_per_simulation)
average_throughput_list_1 = np.zeros(iterations_per_simulation)
average_throughput_list_2 = np.zeros(iterations_per_simulation)
average_throughput_list_3 = np.zeros(iterations_per_simulation)

for sim in range(simulations):


    cell0 = Cell_Model() # Primary

    # These move according to the trace provided by the primary cell
    cell1 = Cell_Model() 
    cell2 = Cell_Model()
    cell3 = Cell_Model()

    if not int(sys.argv[1]):
        cell0.mobility()
        cell1.mobility()
        cell2.mobility()
        cell3.mobility()
    else:
        cell0.mobility(True)
        cell1.mobility(True)
        cell2.mobility(True)
        cell3.mobility(True)

    cell0, cell_loc, d2d_loc, sf_cell, sf_d2d = initial_user_locations(cell0)
    cell1 = copy_initial_data(cell1, cell_loc, d2d_loc, sf_cell, sf_d2d)
    cell2 = copy_initial_data(cell2, cell_loc, d2d_loc, sf_cell, sf_d2d)
    cell3 = copy_initial_data(cell3, cell_loc, d2d_loc, sf_cell, sf_d2d)

    cell0 = init_channels(cell0)
    cell0 = allocate(cell0)
    cell1 = init_channels(cell1)
    cell1 = allocate(cell1)
    cell2 = init_channels(cell2)
    cell2 = allocate(cell2)
    cell3 = init_channels(cell3)
    cell3 = allocate(cell3)


    print('\n Simulation {} \n'.format(sim + 1))

    #print('Location -> {}'.format(cell1.cellular_list[0].loc))
    #print('Location -> {}'.format(cell2.cellular_list[0].loc))
    #print('Location -> {}'.format(cell3.cellular_list[0].loc))


    iterations_list, throughput_list, cellular_trace, d2d_trace = compute(cell0,
             16, iterations_per_simulation, swap_new, smart_PC=True, 
                    caching=True, caching_parameter=2, follow_trace=False)
    average_throughput_list = average_throughput_list + np.array(throughput_list)

    #print(cellular_trace)
    #print(d2d_trace)


    print('\n Simulation {} \n'.format(sim + 1))

    #print('Location -> {}'.format(cell1.cellular_list[0].loc))
    #print('Location -> {}'.format(cell2.cellular_list[0].loc))
    #print('Location -> {}'.format(cell3.cellular_list[0].loc))


    iterations_list, throughput_list, _, _ = compute(cell1, 16, 
            iterations_per_simulation, swap_new, smart_PC=True, caching=False, 
                follow_trace=True, 
                cellular_trace=cellular_trace, 
                d2d_trace=d2d_trace)
    average_throughput_list_1 = average_throughput_list_1 + np.array(throughput_list)


    print('\n Simulation {} \n'.format(sim + 1))


    iterations_list, throughput_list, _, _ = compute(cell2, 16, 
            iterations_per_simulation, swap, smart_PC=False, 
                caching=False, follow_trace=True, 
                cellular_trace=cellular_trace, 
                d2d_trace=d2d_trace)
    average_throughput_list_2 = average_throughput_list_2 + np.array(throughput_list)


    print('\n Simulation {} \n'.format(sim + 1))


    iterations_list, throughput_list, _, _ = compute(cell3, 16, 
            iterations_per_simulation, swap_func=None, smart_PC=False, 
                caching=False, follow_trace=True, 
                cellular_trace=cellular_trace, 
                d2d_trace=d2d_trace)
    average_throughput_list_3 = average_throughput_list_3 + np.array(throughput_list)

average_throughput_list = list(average_throughput_list / simulations)
average_throughput_list_1 = list(average_throughput_list_1 / simulations)
average_throughput_list_2 = list(average_throughput_list_2 / simulations)
average_throughput_list_3 = list(average_throughput_list_3 / simulations)


quad_display(iterations_list, average_throughput_list, 'blue', 'mAQRP', 
        average_throughput_list_1, 'green', 'mQRP', 
        average_throughput_list_2, 'red', 'Open-Loop with Swapping', 
        average_throughput_list_3, 'black', 'Open-Loop', 
        'Iterations', 'Throughput in Mbps')

