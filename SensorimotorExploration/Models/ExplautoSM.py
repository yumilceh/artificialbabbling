'''
Created on Jan 22, 2017

@author: Juan Manuel Acevedo Valle
'''

from explauto.sensorimotor_model.sensorimotor_model import SensorimotorModel
from explauto.sensorimotor_model.non_parametric import NonParametric
import numpy as np


class PARAMS(object):
    def __init__(self):
        pass


class ExplautoSM(object):
    """
    Implemented for non-parametric models
    """
    def __init__(self, system, model_type = 'nearest_neighbor', model_conf='default'):
        conf = generateConfigurationExplauto(system)
        self.conf = conf

        if model_type is 'non_parametric':
            self.model = NonParametric(conf, **model_conf)
        else:
            self.model = SensorimotorModel.from_configuration(conf, model_type, model_conf)

        self.params = PARAMS()
        self.params.sm_step=1 #only ok with non-parametric
       
    def get_action(self, system, sensor_goal=None):
        if sensor_goal is None:
            sensor_goal=system.sensor_goal  #s_g
        
        system.motor_command = self.model.inverse_prediction(sensor_goal)
        return  system.motor_command
    
    def train(self, simulation_data):
        m = simulation_data.motor_data.data.iloc[-1]
        s = simulation_data.sensor_data.data.iloc[-1]
        self.model.update(m,s)

        
    def trainIncrementalLearning(self, simulation_data):
        self.train(simulation_data)
        
    def set_sigma_explo_ratio(self, new_value):
        conf = self.conf
        self.model.sigma_expl = (conf.m_maxs - conf.m_mins) * float(new_value)


def generateConfigurationExplauto(system):
    conf = PARAMS()
    conf.m_maxs = system.max_motor_values
    conf.m_mins = system.min_motor_values
    conf.s_maxs = system.max_sensor_values
    conf.s_mins = system.min_sensor_values
   
    n_motor = system.n_motor
    n_sensor = system.n_sensor
    
    conf.m_ndims = n_motor
    conf.s_ndims = n_sensor

    conf.m_dims = np.arange(0, n_motor, 1).tolist()
    conf.s_dims = np.arange(n_motor, n_motor+n_sensor, 1).tolist()

    conf.bounds = np.zeros((2, n_motor + n_sensor))
    conf.bounds[0,:] = np.array(np.hstack((conf.m_mins, conf.s_mins))).flatten()
    conf.bounds[1,:] = np.array(np.hstack((conf.m_maxs, conf.s_maxs))).flatten()
    return conf
    
    

    
    
    