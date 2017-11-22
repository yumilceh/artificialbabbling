"""
Created on Nov 20, 2017

@author: Juan Manuel Acevedo Valle
"""
import numpy as np
import copy

class PARAMS(object):
    def __init__(self):
        pass


class Sensorimotor():
    def __init__(self, system,
                 somato = False,
                 sm_step=100,
                 sigma_expl_ratio=0.,
                 mode = 'exploit'):
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
        self.params.size_data = system.n_motor + n_sensor
        self.params.motor_names = system.motor_names
        self.params.sensor_names = sensor_names

        self.params.n_motor = system.n_motor
        self.params.n_sensor = n_sensor

        self.params.sm_step = sm_step

        self.params.min_motor_values = system.min_motor_values
        self.params.max_motor_values = system.max_motor_values

        self.delta_motor_values = system.max_motor_values - system.min_motor_values
        self.params.sigma_expl = self.delta_motor_values * float(sigma_expl_ratio)
        self.params.mode = mode

        self.params.m_dims = np.arange(0, self.params.n_motor, 1)
        self.params.s_dims = np.arange(self.params.n_motor, self.params.n_motor + self.params.n_sensor, 1)

    def train(self, simulation_data):
        raise Exception("\"train\" method for Sensorimotor system not implemented.")

    def train_incremental(self, simulation_data, all=False):
        raise Exception("\"train_incremental\" method for Sensorimotor system not implemented.")

    def get_action(self, system, sensor_goal=None):
        raise Exception("\"get_action\" method for Sensorimotor system not implemented.")

    def set_sigma_expl_ratio(self, new_value):
        self.params.sigma_expl = self.delta_motor_values * float(new_value)

    def set_sigma_expl(self, new_sigma):
        self.params.sigma_expl = new_sigma

    def get_sigma_expl(self):
        return copy.copy(self.params.sigma_expl)

    def generate_log(self):
        raise Exception("\"generate_log\" method for Sensorimotor system not implemented.")

    def save(self, file_name):
        raise Exception("\"save\" method for Sensorimotor system not implemented.")

    def apply_sigma_expl(self, motor_command):
        if self.params.mode == 'explore':
            # n_motor = self.params.n_motor
            # for i in range(n_motor):
            #     motor_command[i] += np.random.normal(0.0, self.params.sigma_expl[i], 1)
            idx = np.where(self.params.sigma_expl > 0)
            motor_command[idx] = np.random.normal(motor_command[idx],self.params.sigma_expl[idx])
        return self.bound_action(motor_command)

    def bound_action(self, motor_command):
        n_motor = self.params.n_motor
        min_motor_values = self.params.min_motor_values
        max_motor_values = self.params.max_motor_values
        for i in range(n_motor):
            if (motor_command[i] < min_motor_values[i]):
                motor_command[i] = min_motor_values[i]
            elif (motor_command[i] > max_motor_values[i]):
                motor_command[i] = max_motor_values[i]
        return motor_command

