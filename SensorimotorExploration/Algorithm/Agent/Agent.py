'''
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
'''
from SensorimotorExploration.Models.GMM_SM import GMM_SM

class Agent(object):
    '''
    classdocs
    '''


    def __init__(self, system):
        '''
        Constructor
        '''
        self.system=system;
        self.gmm_sm=GMM_SM(system)
        
        
    