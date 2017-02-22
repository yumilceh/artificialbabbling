'''
Created on Jan 26, 2017

@author: Juan Manuel Acevedo Valle
'''
from ..Algorithm.utils.RndSensorimotorFunctions import get_random_motor_set, get_random_sensor_set
    
class RandomModel(object):
    '''
    classdocs
    '''
    def __init__(self, agent, mode='motor'):
        '''
        Constructor
        '''
        
        self.mode = mode
        
    def train(self, simulation_data):       
        pass
        
    def trainIncremental(self, simulation_data):
        pass
    
    def get_interesting_goal(self,agent):
        if self.mode=='motor':
            return get_random_motor_set(agent, 1)[0]
        elif self.mode=='sensor':
            return get_random_sensor_set(agent, 1)[0]
        else:
            raise ValueError('Unknown Mode')
        
    def get_interesting_goals(self,agent, n_samples=2):
        if self.mode=='motor':
            return get_random_motor_set(agent, n_samples)
        elif self.mode=='sensor':
            return get_random_sensor_set(agent, n_samples)
        else:
            raise ValueError('Unknown Mode')