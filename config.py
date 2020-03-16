import numpy as np
import random
from numpy.random import uniform, exponential, normal
import math

class Cell_Model:

    def __init__(self, cell_radius=500, d2d_radius=50, 
            cell_users=30, d2d_pairs=10, channels=33, time=5):
        
        self.cell_radius = cell_radius
        self.d2d_radius = d2d_radius
        self.cell_users = cell_users
        self.d2d_pairs = d2d_pairs
        self.channels = channels
        self.time = time
        
        self.cellular_list = []
        self.d2d_list = []
        
    def mobility(self, d2d=False):
    
        self.del_cell_rad = self.cell_radius / 10
        if d2d:
            self.del_d2d_rad = self.d2d_radius / 10
        else:
            self.del_d2d_rad = 0
        self.del_theta = 10 # Degrees

        
class D2D:
    
    def __init__(self, id, tx, rv, cell_radius, d2d_radius, 
            del_cell_rad, del_d2d_rad, del_theta):
        
        self.id = id
        self.channel = None
        self.tx = tx # Transmitter Location
        self.rv = rv # Receiver Location
        self.d2d_radius = d2d_radius
        self.cell_radius = cell_radius
        self.del_cell_rad = del_cell_rad
        self.del_d2d_rad = del_d2d_rad
        self.del_theta = del_theta
        self.pow = 0 # Transmitter power
        self.shadow_fading = 0
        
    def move(self):
    
        if (self.tx[0] + self.del_cell_rad + self.d2d_radius) > self.cell_radius:
            self.tx[0] = self.tx[0] - self.del_cell_rad
            
        elif (self.tx[0] + self.del_cell_rad + self.d2d_radius) <= self.cell_radius:
            self.tx[0] = self.tx[0] + uniform(-self.del_cell_rad, 
                    self.del_cell_rad)
            self.tx[1] = self.tx[1] + uniform(-self.del_theta, self.del_theta)
            
        if (self.rv[0] + self.del_d2d_rad) > self.d2d_radius:
            self.rv[0] = self.rv[0] - self.del_d2d_rad
            
        elif (self.rv[0] + self.del_d2d_rad) <= self.d2d_radius:
            self.rv[0] = self.rv[0] + uniform(-self.del_d2d_rad, 
                    self.del_d2d_rad)
            self.rv[1] = self.rv[1] + uniform(-self.del_theta, self.del_theta)

    def d2d_channel_gain(self):

        path_loss = 148 + 40*math.log10(10*self.rx/1000)
        gain = pow(self.rx, -4) * pow(10, (self.shadow_fading + path_loss)/10)

        return gain

    def cell_channel_gain(self, cell_loc, cell_fading):

        # Calculate distance between D2D Rx and Cellular UE
        path_loss = 128.1 + 37.6*math.log10(10*distance/1000)
        gain = pow(distance, -4) * pow(10, (cell_fading + path_loss)/10)

        return gain

    def power_given_SINR(self, sinr, cell_power, d2d_gain, cell_gain, noise):

        power = sinr * (noise + (cell_power * cell_gain)) / d2d_gain

        return power

    def SINR_given_power(self, power, cell_power, d2d_gain, cell_gain, noise):

        sinr = (power * d2d_gain) / (noise + (cell_power * cell_gain))

        return sinr



class Cellular_UE:

    def __init__(self, id, location, cell_radius, 
            del_cell_rad, del_theta):
        
        self.id = id
        self.channel = None
        self.loc = location
        self.cell_radius = cell_radius
        self.del_rad = del_cell_rad
        self.del_theta = del_theta
        self.power = 0
        self.shadow_fading = 0
        
    def move(self):
        
        if (self.loc[0] + self.del_rad) > self.cell_radius:
            self.loc[0] = self.loc[0] - self.del_rad
            
        elif (self.loc[0] + self.del_rad) <= self.cell_radius:
            self.loc[0] = self.loc[0] + uniform(-self.del_rad, self.del_rad)
            self.loc[1] = self.loc[1] + uniform(-self.del_theta, 
                self.del_theta)

    def d2d_channel_gain(self, tx_loc, d2d_fading):

        path_loss = 148 + 40*math.log10(10*tx_loc/1000)
        gain = pow(tx_loc, -4) * pow(10, (d2d_fading + path_loss)/10)

        return gain

    def cell_channel_gain(self):

        path_loss = 128.1 + 37.6*math.log10(10*self.loc/1000)
        gain = pow(self.loc, -4) * pow(10, (self.shadow_fading + path_loss)/10)

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
        

