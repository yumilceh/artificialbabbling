'''
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
'''
from GeneralModels.Mixture import GMM
import numpy as np
import pandas as pd
class GMM_SM(object):
    '''
    classdocs
    '''


    def __init__(self, Agent, n_gauss_components, alpha=0.1, sm_step=400):
        '''
        Constructor
        '''
        self.size_data=Agent.n_motor+Agent.n_sensor
        self.motor_names=Agent.motor_names;
        self.sensor_names=Agent.sensor_names;
        self.n_motor=Agent.n_motor;
        self.n_sensor=Agent.n_sensor;
        self.GMM=GMM(n_gauss_components)
        self.alpha=alpha
        self.sm_step=sm_step
        
    def train(self,simulation_data):
        train_data_tmp=pd.concat([simulation_data.motor_data.data,simulation_data.sensor_data.data], axis=1)
        self.GMM.train(train_data_tmp.as_matrix(columns=None))
        
    def trainIncrementalLearning(self,simulation_data):
        sm_step=self.sm_step
        alpha=self.alpha
        motor_data_size=len(simulation_data.motor_data.data.index)
        motor_data=simulation_data.motor_data.data[motor_data_size-sm_step:-1]
        sensor_data_size=len(simulation_data.sensor_data.data.index)
        sensor_data=simulation_data.sensor_data.data[sensor_data_size-sm_step:-1]
        new_data=pd.concat([motor_data,sensor_data],axis=1)
        self.GMM.trainIncrementalLearning(new_data, alpha)
         
    
    def getMotorCommand(self,Agent):
        n_motor=Agent.n_motor;
        n_sensor=Agent.n_sensor;
        sensor_goal=Agent.sensor_goal  #s_g
        m_dims=np.arange(0, n_motor, 1)
        s_dims= np.arange(n_motor, n_motor+n_sensor, 1)
        Agent.motor_command=self.GMM.predict(m_dims, s_dims, sensor_goal)
        
                

        