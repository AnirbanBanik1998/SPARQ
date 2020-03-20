import numpy as np
import random
from numpy.random import uniform, exponential, normal
import math
from queue import Queue 

class Cell_Model:

    def __init__(self, cell_radius=500, d2d_radius=50, 
            cell_users=30, d2d_pairs=36, channels=33, time=5, 
            noise=pow(10, -14.6)):
        
        self.cell_radius = cell_radius
        self.d2d_radius = d2d_radius
        self.cell_users = cell_users
        self.d2d_pairs = d2d_pairs
        self.channels = channels
        self.time = time
        self.noise = noise
        
        self.cellular_list = []
        self.d2d_list = []
        self.unnalocated_d2d = Queue()
        self.channel_list = []

        self.cell_threshold_SINR = pow(10, 0.6)
        self.d2d_threshold_SINR = pow(10, -6) # No QoS requirements for D2D
        
    def mobility(self, d2d=False):
    
        self.del_cell_rad = self.cell_radius / 10
        if d2d:
            self.del_d2d_rad = self.d2d_radius / 10
        else:
            self.del_d2d_rad = 0
        self.del_theta = 10 # Degrees

        
class D2D:
    
    def __init__(self, id, tx, rv, cell_radius, d2d_radius, 
            del_cell_rad, del_d2d_rad, del_theta, shadow_fading):
        
        self.id = id
        self.channel = None
        self.tx = tx # Transmitter Location
        self.rv = rv # Receiver Location
        self.d2d_radius = d2d_radius
        self.cell_radius = cell_radius
        self.del_cell_rad = del_cell_rad
        self.del_d2d_rad = del_d2d_rad
        self.del_theta = del_theta
        self.shadow_fading = shadow_fading
        self.pow = 0 # Transmitter power
        self.max_power = pow(10, -0.7)
        self.allocation = 0 # 0->Non-Allocated, 1->Shared, 2->Dedicated
        
    def move(self):
    
        #Considering both cases, location out of cell, and radius -ve
        if (self.tx[0] + self.del_cell_rad + self.d2d_radius) > self.cell_radius:
            self.tx[0] = self.tx[0] - self.del_cell_rad

        elif (self.tx[0] - self.del_cell_rad - self.d2d_radius) < 0:
            self.tx[0] = self.tx[0] + self.del_cell_rad
            
        elif (self.tx[0] + self.del_cell_rad + self.d2d_radius) <= self.cell_radius:
            self.tx[0] = self.tx[0] + uniform(-self.del_cell_rad, 
                    self.del_cell_rad)
            self.tx[1] = self.tx[1] + uniform(-self.del_theta, self.del_theta)
            
        if (self.rv[0] + self.del_d2d_rad) > self.d2d_radius:
            self.rv[0] = self.rv[0] - self.del_d2d_rad

        elif (self.rv[0] - self.del_d2d_rad) < 0:
            self.rv[0] = self.rv[0] + self.del_d2d_rad
            
        elif (self.rv[0] + self.del_d2d_rad) <= self.d2d_radius:
            self.rv[0] = self.rv[0] + uniform(-self.del_d2d_rad, 
                    self.del_d2d_rad)
            self.rv[1] = self.rv[1] + uniform(-self.del_theta, self.del_theta)

    def d2d_channel_gain(self):

        path_loss = 28 + 40*math.log10(10*self.rv[0]/1000)
        gain = pow(self.rv[0], -4) * pow(10, 
                (self.shadow_fading + path_loss)/10)

        return gain

    def cell_channel_gain(self, cell_loc, cell_fading):

        # Calculate distance between D2D Rx and Cellular UE
        tx_x_y = [self.tx[0]*(math.cos(3.14*self.tx[1]/180)), 
                self.tx[0]*(math.sin(3.14*self.tx[1]/180))]
        rv_x_y = [self.rv[0]*(math.cos(3.14*self.rv[1]/180)) + tx_x_y[0], 
                self.rv[0]*(math.sin(3.14*self.rv[1]/180)) + tx_x_y[1]]
        cell_x_y = [cell_loc[0]*(math.cos(3.14*cell_loc[1]/180)), 
                cell_loc[0]*(math.sin(3.14*cell_loc[1]/180))]

        distance = math.sqrt(math.pow((cell_x_y[0] - rv_x_y[0]), 2) + \
                math.pow((cell_x_y[1] - rv_x_y[1]), 2))

        path_loss = 28 + 40*math.log10(10*distance/1000)
        gain = pow(distance, -4) * pow(10, (cell_fading + path_loss)/10)

        return gain

    def power_given_SINR(self, sinr, cell_power, d2d_gain, cell_gain, noise):

        power = sinr * (noise + (cell_power * cell_gain)) / d2d_gain

        return power

    def SINR_given_power(self, power, cell_power, d2d_gain, cell_gain, noise):

        sinr = (power * d2d_gain) / (noise + (cell_power * cell_gain))

        return sinr

    def get_cell_state(self, cell_SINR, cell_threshold_SINR):

        if cell_SINR >= cell_threshold_SINR:
            return 1
        else:
            return 0

    def calc_reward(self, state, d2d_SINR, cell_SINR, priority):

        if state:
            reward = priority * math.log(1+cell_SINR) + math.log(1+d2d_SINR)
        else:
            reward = -1

        return reward



class Cellular_UE:

    def __init__(self, id, location, cell_radius, 
            del_cell_rad, del_theta, shadow_fading):
        
        self.id = id
        self.channel = None
        self.loc = location
        self.cell_radius = cell_radius
        self.del_rad = del_cell_rad
        self.del_theta = del_theta
        self.max_power = 2
        self.shadow_fading = shadow_fading

        '''
        The cellular device modulates its power based on distance
        The Maximum Tx power is when cellular device is at the edge
        '''
        path_loss = 15.3 + 37.6*math.log10(10*500/1000)
        gain = pow(500, -4) * pow(10, 
                (self.shadow_fading + path_loss)/10)
        self.constant = self.max_power * gain
        self.power = self.get_power(self.cell_channel_gain())

        
    def move(self):
        
        if (self.loc[0] + self.del_rad) > self.cell_radius:
            self.loc[0] = self.loc[0] - self.del_rad

        elif (self.loc[0] - self.del_rad) < 0:
            self.loc[0] = self.loc[0] + self.del_rad
            
        elif (self.loc[0] + self.del_rad) <= self.cell_radius:
            self.loc[0] = self.loc[0] + uniform(-self.del_rad, self.del_rad)
            self.loc[1] = self.loc[1] + uniform(-self.del_theta, 
                self.del_theta)

        self.power = self.get_power(self.cell_channel_gain())

    def get_power(self, gain):

        power = self.constant / gain
        return power

    def d2d_channel_gain(self, tx_loc, d2d_fading):

        path_loss = 15.3 + 37.6*math.log10(10*tx_loc[0]/1000)
        gain = pow(tx_loc[0], -4) * pow(10, (d2d_fading + path_loss)/10)

        return gain

    def cell_channel_gain(self):

        path_loss = 15.3 + 37.6*math.log10(10*self.loc[0]/1000)
        gain = pow(self.loc[0], -4) * pow(10, 
                (self.shadow_fading + path_loss)/10)

        return gain

    def power_given_SINR(self, sinr, d2d_power, d2d_gain, cell_gain, noise):

        power = sinr * (noise + (d2d_power * d2d_gain)) / cell_gain

        return power

    def SINR_given_power(self, power, d2d_power, d2d_gain, cell_gain, noise):

        sinr = (power * cell_gain) / (noise + (d2d_power * d2d_gain))

        return sinr
            
class Channel:

    def __init__(self, id):
    
        self.id = id
        self.cell = None
        self.d2d = None
    

class Base_Station:

    def __init__(self, cell_users, d2d_pairs, channels):
    
        self.m = cell_users
        self.n = channels
        self.k = d2d_pairs
        self.cell_users = []
        self.d2d_pairs = []
        self.channels = []
        

