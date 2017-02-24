"""
Created on Jan 22, 2017

@author: Juan Manuel Acevedo Valle
"""

from explauto.sensorimotor_model.sensorimotor_model import SensorimotorModel
import numpy as np


class PARAMS(object):
    def __init__(self):
        pass


class ExplautoSS(object):
    """
    Implemented for non-parametric models
    """
    def __init__(self, agent, model_type, model_conf = "default"):
        conf = generateConfigurationExplauto(agent)
        self.conf = conf
        self.model = SensorimotorModel.from_configuration(conf, model_type, model_conf)
        
        self.params = PARAMS()
        self.params.sm_step=1 #only ok with non-parametric
       
    def getSomatoPrediction(self,agent,motor_command=None):          
        if motor_command==None:
            motor_command = agent.sensor_goal  #s_g
        
        agent.somato_prediction = self.model.forward_prediction(motor_command)
        return  agent.somato_prediction
    
    def train(self, simulation_data):
        m = simulation_data.motor_data.data.iloc[-1]
        s = simulation_data.somato_data.data.iloc[-1]
        self.model.update(m,s)
        
    def trainIncrementalLearning(self, simulation_data):
        self.train(simulation_data)
        
    def set_sigma_explo_ratio(self, new_value):
        conf = self.conf
        self.model.sigma_expl = (conf.m_maxs - conf.m_mins) * float(new_value)
     
        
def generateConfigurationExplauto(agent):
    conf = PARAMS()
    conf.m_maxs = agent.max_motor_values
    conf.m_mins = agent.min_motor_values
    conf.s_maxs = agent.max_somato_values
    conf.s_mins = agent.min_somato_values
   
    n_motor = agent.n_motor;
    n_sensor = agent.n_somato;
    
    conf.m_ndims = n_motor
    conf.s_ndims = n_sensor

    conf.m_dims = np.arange(0, n_motor, 1).tolist()
    conf.s_dims = np.arange(n_motor, n_motor+n_sensor, 1).tolist()

    conf.bounds = np.zeros((2, n_motor + n_sensor))
    conf.bounds[0,:] = np.array(np.hstack((conf.m_mins, conf.s_mins))).flatten()
    conf.bounds[1,:] = np.array(np.hstack((conf.m_maxs, conf.s_maxs))).flatten()
    return conf
    
    

    
    
    