'''
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
'''
from SensorimotorExploration.Models.GeneralModels.IGMMpy import IGMM as GMM
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

    def __init__(self, system,
                       sm_step = 100,
                       min_components = 3,
                       max_step_components = 30,
                       max_components = 60,
                       a_split = 0.8,
                       forgetting_factor = 0.05,
                       sigma_explo_ratio = 0.0,
                       plot = False, plot_dims=[0,1]):
        '''
        Constructor
        '''

        self.params=PARAMS()
        self.params.size_data=system.n_motor+system.n_sensor
        self.params.motor_names=system.motor_names
        self.params.sensor_names=system.sensor_names
        self.params.n_motor=system.n_motor
        self.params.n_sensor=system.n_sensor
        self.params.min_components = min_components
        self.params.max_step_components = max_step_components
        self.params.forgetting_factor = forgetting_factor
        self.params.sm_step = sm_step

        self.delta_motor_values = system.max_motor_values - system.min_motor_values
        self.sigma_expl =  self.delta_motor_values * float(sigma_explo_ratio)
        self.mode = 'explore'

        self.model=GMM(min_components = min_components,
                       max_step_components = max_step_components,
                       max_components = max_components,
                       a_split = a_split,
                       forgetting_factor = forgetting_factor, 
                       plot = plot, plot_dims=plot_dims)

    def train(self, simulation_data):
        train_data_tmp = pd.concat([simulation_data.motor.get_all(),
                                    simulation_data.sensor.get_all()], axis=1)
        self.model.train(train_data_tmp.as_matrix(columns=None))

    def trainIncrementalLearning(self, simulation_data, all=True):
        if all:
            data = np.zeros((simulation_data.motor.current_idx,
                             self.params.n_motor+self.params.n_sensor))
            data_m = simulation_data.motor.get_all().as_matrix()
            data_s = simulation_data.sensor.get_all().as_matrix()
            data[:,:self.params.n_motor] = data_m
            data[:, self.params.n_motor:] = data_s
        else:
            data = np.zeros((self.params.sm_step,
                             self.params.n_motor+self.params.n_sensor))
            data_m = simulation_data.motor.get_last(self.params.sm_step).as_matrix()
            data_s = simulation_data.sensor.get_last(self.params.sm_step).as_matrix()
            data[:,:self.params.n_motor] = data_m
            data[:, self.params.n_motor:] = data_s
        self.model.train(data)

    # def train_old(self,simulation_data):
    #     train_data_tmp=pd.concat([simulation_data.motor.data,
    #                               simulation_data.sensor.data], axis=1)
    #     self.model.train(train_data_tmp.as_matrix(columns=None))
    #
    # def trainIncrementalLearning_old(self,simulation_data):
    #     #=======================================================================
    #     # sm_step=self.params.sm_step
    #     # alpha=self.params.alpha
    #     # motor_data_size=len(simulation_data.motor.data.index)
    #     # motor=simulation_data.motor.data[motor_data_size-sm_step:-1]
    #     # sensor_data_size=len(simulation_data.sensor.data.index)
    #     # sensor=simulation_data.sensor.data[sensor_data_size-sm_step:-1]
    #     # new_data=pd.concat([motor,sensor],axis=1)
    #     # self.model.trainIncrementalLearning(new_data, alpha)
    #     #=======================================================================
    #     train_data_tmp=pd.concat([simulation_data.motor.data,
    #                               simulation_data.sensor.data], axis=1)
    #     self.model.train(train_data_tmp.as_matrix(columns=None))
         
    
    def get_action(self, system, sensor_goal=None):
        n_motor=system.n_motor
        n_sensor=system.n_sensor
        
        if sensor_goal is None:
            sensor_goal=system.sensor_goal  #s_g
        
        m_dims=np.arange(0, n_motor, 1)
        s_dims= np.arange(n_motor, n_motor+n_sensor, 1)

        motor_command = self.model.predict(m_dims, s_dims, sensor_goal)

        if self.mode == 'explore':
            motor_command[self.sigma_expl > 0] = np.random.normal(motor_command[self.sigma_expl > 0],
                                                                  self.sigma_expl[self.sigma_expl > 0])

        motor_command = boundMotorCommand(system, motor_command)
        system.motor_command = motor_command
        
        # return boundMotorCommand(system,self.model.predict(m_dims, s_dims, sensor_goal))  #Maybe this is wrong
        return motor_command.copy()

    def set_sigma_explo_ratio(self, new_value):
        self.sigma_expl = self.delta_motor_values * float(new_value)

    def set_sigma_explo(self, new_sigma):
        self.sigma_expl = new_sigma

    def get_sigma_explo(self):
        return self.sigma_expl

def boundMotorCommand(system,motor_command):
    n_motor=system.n_motor
    min_motor_values = system.min_motor_values
    max_motor_values = system.max_motor_values
    for i in range(n_motor):
        if (motor_command[i] < min_motor_values[i]):
            motor_command[i] = min_motor_values[i]
        elif (motor_command[i] > max_motor_values[i]):
            motor_command[i] = max_motor_values[i]
    return motor_command