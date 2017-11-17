'''
Created on Jan 26, 2017

@author: Juan Manuel Acevedo Valle
'''
from exploration.algorithm.utils.functions import get_random_motor_set, get_random_sensor_set

class OBJECT(object):
    def __init__(self):
        pass

class Random(object):
    '''
    classdocs
    '''
    def __init__(self, agent, mode='sensor'):
        '''
        Constructor
        '''
        self.mode = mode
        self.model = None
        self.params = OBJECT()
        self.params.im_step = -1  #Model steps must be putted out of the model itself

    def train(self, simulation_data):       
        pass
        
    def trainIncremental(self, simulation_data):
        pass
    
    def get_goal(self,agent):
        if self.mode=='art':
            return get_random_motor_set(agent, 1)[0]
        elif self.mode=='sensor':
            return get_random_sensor_set(agent, 1)[0]
        else:
            raise ValueError('Unknown Mode')
        
    def get_goals(self,agent, n_samples=2):
        if self.mode=='art':
            return get_random_motor_set(agent, n_samples)
        elif self.mode=='sensor':
            return get_random_sensor_set(agent, n_samples)
        else:
            raise ValueError('Unknown Mode')