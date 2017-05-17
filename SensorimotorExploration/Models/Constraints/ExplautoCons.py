"""
Created on Jan 22, 2017

@author: Juan Manuel Acevedo Valle
"""

from explauto.sensorimotor_model.sensorimotor_model import SensorimotorModel
from explauto.sensorimotor_model.non_parametric import NonParametric
import numpy as np


class PARAMS(object):
    def __init__(self):
        pass


class ExplautoCons(object):
    """
    Implemented for non-parametric models
    """
    def __init__(self, system, model_type='nearest_neighbor', model_conf ="default"):
        conf = generateConfigurationExplauto(system)
        self.conf = conf

        if model_type is 'non_parametric':
            self.model = NonParametric(conf, **model_conf)
        else:
            self.model = SensorimotorModel.from_configuration(conf, model_type, model_conf)

        self.set_sigma_explo(0.)  # For conssensory data we are not interested on exploratory noise
        self.params = PARAMS()
        self.params.cons_step=1 #only ok with non-parametric
       
    def predict_cons(self, system, motor_command=None):
        if motor_command is None:
            motor_command = system.motor_command  #s_g
        
        system.cons_prediction = self.model.forward_prediction(motor_command)
        return  system.cons_prediction

    def train(self, simulation_data):
        m = simulation_data.motor.get_last(1).iloc[-1]
        s = simulation_data.cons.get_last(1).iloc[-1]
        # print('Trainign with m {} and som {}'.format(m,s))
        self.model.update(m,s)

    def train_old(self, simulation_data):
        m = simulation_data.motor.data.iloc[-1]
        s = simulation_data.cons.data.iloc[-1]
        # print('Trainign with m {} and som {}'.format(m,s))
        self.model.update(m,s)
        
    def train_incremental(self, simulation_data):
        self.train(simulation_data)
        
    def set_sigma_explo_ratio(self, new_value):
        conf = self.conf
        self.model.sigma_expl = (conf.m_maxs - conf.m_mins) * float(new_value)

    def set_sigma_explo(self, new_sigma):
        self.model.sigma_expl = new_sigma


    def get_sigma_explo(self):
        return self.model.sigma_expl
        
def generateConfigurationExplauto(system):
    conf = PARAMS()
    conf.m_maxs = system.max_motor_values
    conf.m_mins = system.min_motor_values
    conf.s_maxs = system.max_cons_values
    conf.s_mins = system.min_cons_values
   
    n_motor = system.n_motor
    n_sensor = system.n_cons
    
    conf.m_ndims = n_motor
    conf.s_ndims = n_sensor

    conf.m_dims = np.arange(0, n_motor, 1).tolist()
    conf.s_dims = np.arange(n_motor, n_motor+n_sensor, 1).tolist()

    conf.bounds = np.zeros((2, n_motor + n_sensor))
    conf.bounds[0,:] = np.array(np.hstack((conf.m_mins, conf.s_mins))).flatten()
    conf.bounds[1,:] = np.array(np.hstack((conf.m_maxs, conf.s_maxs))).flatten()
    return conf
    
    

    
    
    