'''
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
'''
from igmm import IGMM as GMM
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
                       somato=False,
                       plot = False, plot_dims=[0,1]):
        '''
        Constructor
        '''

        if somato:
            n_sensor = system.n_somato
            sensor_names = system.somato_names
            sensor_space = 'somato'
        else:
            n_sensor = system.n_sensor
            sensor_names = system.sensor_names
            sensor_space = 'sensor'

        self.params = PARAMS()
        self.params.sensor_space = sensor_space
        self.params.size_data = system.n_motor+ n_sensor
        self.params.motor_names = system.motor_names
        self.params.sensor_names = sensor_names

        self.params.n_motor=system.n_motor
        self.params.n_sensor = n_sensor

        self.params.min_components = min_components
        self.params.max_step_components = max_step_components
        self.params.forgetting_factor = forgetting_factor
        self.params.sm_step = sm_step

        self.delta_motor_values = system.max_motor_values - system.min_motor_values
        self.sigma_expl = self.delta_motor_values * float(sigma_explo_ratio)
        self.mode = 'exploit'

        m_dims = np.arange(0,self.params. n_motor, 1)
        s_dims = np.arange(self.params.n_motor, self.params.n_motor + self.params.n_sensor, 1)

        self.model=GMM(min_components = min_components,
                       max_step_components = max_step_components,
                       max_components = max_components,
                       a_split = a_split,
                       forgetting_factor = forgetting_factor,
                       x_dims = m_dims,
                       y_dims = s_dims)

    def train(self, simulation_data):
        sensor_data = getattr(simulation_data, self.params.sensor_space)
        train_data_tmp = pd.concat([simulation_data.motor.get_all(),
                                    sensor_data.get_all()], axis=1)
        self.model.train(train_data_tmp.as_matrix(columns=None))

    def train_incremental(self, simulation_data, all=False):
        sensor_data = getattr(simulation_data, self.params.sensor_space)
        if all:
            data = np.zeros((simulation_data.motor.current_idx,
                             self.params.n_motor+self.params.n_sensor))
            data_m = simulation_data.motor.get_all().as_matrix()
            data_s = sensor_data.get_all().as_matrix()
            data[:,:self.params.n_motor] = data_m
            data[:, self.params.n_motor:] = data_s
        else:
            data = np.zeros((self.params.sm_step,
                             self.params.n_motor+self.params.n_sensor))
            data_m = simulation_data.motor.get_last(self.params.sm_step).as_matrix()
            data_s = sensor_data.get_last(self.params.sm_step).as_matrix()
            data[:,:self.params.n_motor] = data_m
            data[:, self.params.n_motor:] = data_s
        self.model.train(data)
    
    def get_action(self, system, sensor_goal=None):
        n_motor=system.n_motor
        n_sensor=self.params.n_sensor
        
        if sensor_goal is None:
            sensor_goal = getattr(system, self.params.sensor_space+'_goal')  #s_g
        
        m_dims=np.arange(0, n_motor, 1)
        s_dims= np.arange(n_motor, n_motor+n_sensor, 1)

        motor_command = self.model.infer(m_dims, s_dims, sensor_goal)

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

    def generate_log(self):
        params_to_logs = ['sm_step',
                          'min_components',
                          'max_step_components',
                          'max_components',
                          'a_split',
                          'forgetting_factor',
                          'sigma_explo_ratio',
                          'somato']
        log = 'model: IGMM_SM\n'

        for attr_ in params_to_logs:
            if hasattr(self.params, attr_):
                try:
                    attr_log = getattr(self.params, attr_).generate_log()
                    log+=(attr_ + ': {\n')
                    log+=(attr_log)
                    log+=('}\n')
                except IndexError:
                    print("INDEX ERROR in ILGMM log generation")
                except AttributeError:
                    log+=(attr_ + ': ' + str(getattr(self.params, attr_)) + '\n')
        return log

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

