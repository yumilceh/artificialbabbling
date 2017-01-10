'''
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
'''
from Models.GeneralModels.ILGMM_GREC import ILGMM as GMM
import numpy as np
import pandas as pd
import copy 

class PARAMS(object):
    def __init__(self):
        pass;
    
    
class GMM_SM(object):
    '''
    classdocs
    '''

    def __init__(self, Agent,
                       sm_step = 100,
                       min_components = 3,
                       max_step_components = 30,
                       max_components = 60,
                       a_split = 0.8,
                       forgetting_factor = 0.05, 
                       plot = False, plot_dims=[0,1]):
        '''
        Constructor
        '''

        self.params=PARAMS()
        self.params.size_data=Agent.n_motor+Agent.n_sensor
        self.params.motor_names=Agent.motor_names;
        self.params.sensor_names=Agent.sensor_names;
        self.params.n_motor=Agent.n_motor;
        self.params.n_sensor=Agent.n_sensor;
        self.params.min_components = min_components
        self.params.max_step_components = max_step_components
        self.params.forgetting_factor = forgetting_factor
        self.params.sm_step = sm_step
        
        self.model=GMM(min_components = min_components,
                       max_step_components = max_step_components,
                       max_components = max_components,
                       a_split = a_split,
                       forgetting_factor = forgetting_factor, 
                       plot = plot, plot_dims=plot_dims)

        
    def train(self,simulation_data):
        train_data_tmp=pd.concat([simulation_data.motor_data.data,
                                  simulation_data.sensor_data.data], axis=1)
        self.model.train(train_data_tmp.as_matrix(columns=None))
        
    def trainIncrementalLearning(self,simulation_data):
        #=======================================================================
        # sm_step=self.params.sm_step
        # alpha=self.params.alpha
        # motor_data_size=len(simulation_data.motor_data.data.index)
        # motor_data=simulation_data.motor_data.data[motor_data_size-sm_step:-1]
        # sensor_data_size=len(simulation_data.sensor_data.data.index)
        # sensor_data=simulation_data.sensor_data.data[sensor_data_size-sm_step:-1]
        # new_data=pd.concat([motor_data,sensor_data],axis=1)
        # self.model.trainIncrementalLearning(new_data, alpha)
        #=======================================================================
        train_data_tmp=pd.concat([simulation_data.motor_data.data,
                                  simulation_data.sensor_data.data], axis=1)
        self.model.train(train_data_tmp.as_matrix(columns=None))
         
    
    def getMotorCommand(self,Agent,sensor_goal=None):
        n_motor=Agent.n_motor;
        n_sensor=Agent.n_sensor;
        
        if sensor_goal==None:
            sensor_goal=Agent.sensor_goal  #s_g
        
        m_dims=np.arange(0, n_motor, 1)
        s_dims= np.arange(n_motor, n_motor+n_sensor, 1)
         
        Agent.motor_command = boundMotorCommand(Agent,
                                                 self.model.predict_all_gaussians(
                                                     m_dims, s_dims, sensor_goal)) 
        #=======================================================================
        # This might be deprecated at some time
        #=======================================================================

        #=======================================================================
        # Agent.motor_command=boundMotorCommand(Agent,self.model.predict(m_dims,
        #  s_dims, sensor_goal))   
        #=======================================================================
        
        # return boundMotorCommand(Agent,self.model.predict(m_dims, s_dims, sensor_goal))  #Maybe this is wrong
        return copy.deepcopy(Agent.motor_command)
        
                
def boundMotorCommand(Agent,motor_command):
    n_motor=Agent.n_motor;
    min_motor_values = Agent.min_motor_values;
    max_motor_values = Agent.max_motor_values;
    for i in range(n_motor):
        if (motor_command[i] < min_motor_values[i]):
            motor_command[i] = min_motor_values[i]
        elif (motor_command[i] > max_motor_values[i]):
            motor_command[i] = max_motor_values[i]
    return motor_command