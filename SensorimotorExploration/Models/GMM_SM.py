'''
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
'''
from GeneralModels.Mixture import GMM
import pandas as pd
class GMM_SM(object):
    '''
    classdocs
    '''


    def __init__(self, Agent, n_gauss_components):
        '''
        Constructor
        '''
        self.size_data=Agent.n_motor+Agent.n_sensor
        self.motor_names=Agent.motor_names;
        self.sensor_names=Agent.sensor_names;
        self.GMM=GMM(n_gauss_components)
        
        
    def train(self,simulation_data):
        train_data_tmp=pd.concat([simulation_data.sensor_data.data, simulation_data.motor_data.data], axis=1)
        self.GMM.train(train_data_tmp.as_matrix(columns=None)) 
        