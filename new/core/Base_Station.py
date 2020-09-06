import logging
from uuid import uuid4

class Base_Station:

    '''
    {'BS-Key' : {ch_key : 
           {'cell' : {cell_key:SINR_report}}, 
           'd2d' : {d2d_key:SINR_report}}}}
    '''
    allocation_dic = {}

    def __init__(self, multi_cell, location_x=0.0, location_y=0.0, 
                    channels=10):
        
        self.multi_cell = multi_cell
        self.cell_radius = 500
        self.loc_x = location_x
        self.loc_y = location_y
        self.channels = channels
        self.noise = pow(10, -14.6)

        self.BS_key = self.allocate_key()

        self.cell_threshold_SINR = pow(10, 0.6)
        self.d2d_threshold_SINR = pow(10, -6)

        # For Multi-Cellular
        self.vertices_list = []

    '''
    Function to allocate vertices to a BS based on Poisson Voronoi
    '''
    def allocate_polygon_vertices(self, *args):
        
        for vertex in args:
            self.vertices_list.append(vertex)

    def allocate_key(self):
        return key = uuid4()


    def allocate_channels(self):

        for i in range(1, self.channels+1):
            Base_Station.allocation_dic[self.BS_key][i] = {'cell':{}, 
                        'd2d':{}}

    '''
    Location_x and location_y represent coordinates of either Cellular UE or D2D Tx
    '''
    @classmethod
    def is_device_in_polygon(self, BS_key, location_x, location_y):
        pass

    #device_type 0-> d2d, 1->cellular 
    @classmethod
    def allocate_UE_to_BS(cls, BS_key, device_type, device_key, 
                location_x, location_y):
        in_cell = False #UE within Cell boundaries
        allocated = False #UE can be allocated in cell or not
        if multi_cell:
            if cls.is_device_in_polygon(BS_key, 
                        location_x, location_y):
                in_cell = True

        else:
            if pow((pow(location_x, 2)+pow(location_y, 2)), 0.5) < self.cell_radius:
                in_cell = True

        '''
        If Device Type = Cellular UE. Then it will be allocated 
        if only the number of cellular UEs allocated already is < 
        No. of channels
        When allocating, initialize the SINR report value to 0

        If D2D, then it will be allocated such that there are 
        more or less uniform no. of D2Ds per channel.
        So we check for the channel with minimum no. of D2Ds,
        and allocate the new D2D into that channel. 
        '''

        if in_cell:
            if device_type:
                for ch in range(1, self.channels+1):
                    if cls.allocation_dic[BS_key][ch]['cell'] == {}:
                        cls.allocation_dic[BS_key][ch]['cell'] = {device_key:0}
                        allocated = True
                        break

            else:
                min_size = len(cls.allocation_dic[BS_key][1]['d2d'])

                min_ch_key = 1
                for ch in range(2, self.channels+1):
                    if len(cls.allocation_dic[BS_key][ch]['d2d']) < min_size:
                        min_size = len(cls.allocation_dic[BS_key][ch]['d2d'])
                        min_ch_key = ch

                cls.allocation_dic[BS_key][min_ch_key]['d2d'][device_key] = 0
                allocated = True

        return allocated




    @classmethod
    def report_SINR_to_BS(cls, BS_key, ch_key, device_type, 
                device_key, SINR_report):
        if device_type:
            cls.allocation_dic[BS_key][ch_key]['cell'][device_key] = SINR_report
        else:
            cls.allocation_dic[BS_key][ch_key]['d2d'][device_key] = SINR_report

    @classmethod
    def get_allocation_data_from_BS(cls, BS_key, device_type, 
                device_key):
        device = ''
        if device_type:
            device = 'cell'
        else:
            device = 'd2d'

        for ch in range(1, self.channels+1):
            if cls.allocation_dic[BS_key][ch][device][device_key]:
                return ch 

    @classmethod
    def get_SINR_from_BS(cls, BS_key, ch_key):
        pass

    def reallocate(self):
        pass