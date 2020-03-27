import sys
import numpy as np
import random as rand
from numpy.random import uniform, exponential, normal
import math
import time
import matplotlib.pyplot as plt
from copy import deepcopy

from utils import initial_user_locations, init_channels, allocate, swap, swap_new, copy_initial_data
from q_learn import learn, learn_dumb
from config import Cell_Model, D2D, Cellular_UE, Channel
from display import quad_display

from algo import compute


simulations = 5
iterations_per_simulation = 100

average_throughput_list = np.zeros(iterations_per_simulation)
average_throughput_list_1 = np.zeros(iterations_per_simulation)
average_throughput_list_2 = np.zeros(iterations_per_simulation)
average_throughput_list_3 = np.zeros(iterations_per_simulation)

for sim in range(simulations):

    print('\n Simulation {} \n'.format(sim + 1))

    cell = Cell_Model() # Primary

    # These move according to the trace provided by the primary cell
    cell1 = Cell_Model() 
    cell2 = Cell_Model()
    cell3 = Cell_Model()

    if not int(sys.argv[1]):
        cell.mobility()
        cell1.mobility()
        cell2.mobility()
        cell3.mobility()
    else:
        cell.mobility(True)
        cell1.mobility(True)
        cell2.mobility(True)
        cell3.mobility(True)

    cell = initial_user_locations(cell)
    cell1 = copy_initial_data(cell, cell1)
    cell2 = copy_initial_data(cell, cell2)
    cell3 = copy_initial_data(cell, cell3)

    cell = init_channels(cell)
    cell = allocate(cell)
    cell1 = init_channels(cell1)
    cell1 = allocate(cell1)
    cell2 = init_channels(cell2)
    cell2 = allocate(cell2)
    cell3 = init_channels(cell3)
    cell3 = allocate(cell3)

    iterations_list, throughput_list, cellular_trace, d2d_trace = compute(cell,
             16, iterations_per_simulation, swap_new, 
                    caching=True, caching_parameter=1, follow_trace=False)
    average_throughput_list = average_throughput_list + np.array(throughput_list)


    iterations_list, throughput_list, _, _ = compute(cell1, 16, 
            iterations_per_simulation, swap_new, caching=True, 
                caching_parameter=2, follow_trace=True, 
                cellular_trace=cellular_trace, 
                d2d_trace=d2d_trace)
    average_throughput_list_1 = average_throughput_list_1 + np.array(throughput_list)

    iterations_list, throughput_list, _, _ = compute(cell2, 16, 
            iterations_per_simulation, swap_new, caching=True, 
                caching_parameter=3, follow_trace=True, 
                cellular_trace=cellular_trace, 
                d2d_trace=d2d_trace)
    average_throughput_list_2 = average_throughput_list_2 + np.array(throughput_list)

    iterations_list, throughput_list, _, _ = compute(cell3, 16, 
            iterations_per_simulation, swap_new, caching=True, 
                caching_parameter=4, follow_trace=True, 
                cellular_trace=cellular_trace, 
                d2d_trace=d2d_trace)
    average_throughput_list_3 = average_throughput_list_3 + np.array(throughput_list)

average_throughput_list = list(average_throughput_list / simulations)
average_throughput_list_1 = list(average_throughput_list_1 / simulations)
average_throughput_list_2 = list(average_throughput_list_2 / simulations)
average_throughput_list_3 = list(average_throughput_list_3 / simulations)


quad_display(iterations_list, average_throughput_list, 'b-', 'Param-1', 
        average_throughput_list_1, 'b--', 'Param-2', 
        average_throughput_list_2, 'r-', 'Param-3', 
        average_throughput_list_3, 'r--', 'Param-4', 
        'Iterations', 'Throughput in Mbps')