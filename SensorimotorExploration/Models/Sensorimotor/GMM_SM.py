'''
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
'''
from SensorimotorExploration.Models.GeneralModels.Mixture import GMM
import numpy as np
import pandas as pd


class PARAMS(object):
    def __init__(self):
        pass


class GMM_SM(object):
    '''
    classdocs
    '''

    def __init__(self, system, n_gauss_components, sigma_explo_ratio = 0.0, alpha=0.1, sm_step=400):
        '''
        Constructor
        '''

        self.params = PARAMS()
        self.params.size_data = system.n_motor + system.n_sensor
        self.params.motor_names = system.motor_names
        self.params.sensor_names = system.sensor_names
        self.params.n_motor = system.n_motor
        self.params.n_sensor = system.n_sensor
        self.params.alpha = alpha
        self.params.sm_step = sm_step

        self.delta_motor_values = system.max_motor_values - system.min_motor_values
        self.sigma_expl =  self.delta_motor_values * float(sigma_explo_ratio)
        self.mode = 'explore'

        self.model = GMM(n_gauss_components)

    def train(self, simulation_data):
        train_data_tmp = pd.concat([simulation_data.motor.data, simulation_data.sensor.data], axis=1)
        self.model.train(train_data_tmp.as_matrix(columns=None))

    def trainIncrementalLearning(self, simulation_data):
        alpha = self.params.alpha
        self.model.train_incremental(self.returnTrainData(simulation_data), alpha)

    def returnTrainData(self, simulation_data):
        sm_step = self.params.sm_step
        alpha = self.params.alpha
        motor_data_size = len(simulation_data.motor.data.index)
        if motor_data_size<sm_step:
            motor_data = simulation_data.motor.data
            sensor_data = simulation_data.sensor.data
        else:
            motor_data = simulation_data.motor.data[motor_data_size - sm_step:-1]
            sensor_data = simulation_data.sensor.data[motor_data_size - sm_step:-1]

        new_data = pd.concat([motor_data, sensor_data], axis=1)
        return new_data.as_matrix(columns=None)

    def get_action(self, system, sensor_goal=None):
        n_motor = system.n_motor
        n_sensor = system.n_sensor

        if sensor_goal is None:
            sensor_goal = system.sensor_goal  # s_g

        m_dims = np.arange(0, n_motor, 1)
        s_dims = np.arange(n_motor, n_motor + n_sensor, 1)

        motor_command = self.model.predict(m_dims, s_dims, sensor_goal)

        if self.mode == 'explore':
            motor_command[self.sigma_expl>0] = np.random.normal(motor_command[self.sigma_expl > 0],
                                                                self.sigma_expl[self.sigma_expl > 0])

        motor_command = boundMotorCommand(system, motor_command)
        system.motor_command = motor_command
        return motor_command.copy()

    def set_sigma_explo_ratio(self, new_value):
        self.sigma_expl = self.delta_motor_values * float(new_value)

    def set_sigma_explo(self, new_sigma):
        self.model.sigma_expl = new_sigma

    def get_sigma_explo(self):
        return self.model.sigma_expl

def boundMotorCommand(system, motor_command):
    n_motor = system.n_motor
    min_motor_values = system.min_motor_values
    max_motor_values = system.max_motor_values
    for i in range(n_motor):
        if (motor_command[i] < min_motor_values[i]):
            motor_command[i] = min_motor_values[i]
        elif (motor_command[i] > max_motor_values[i]):
            motor_command[i] = max_motor_values[i]
    return motor_command
