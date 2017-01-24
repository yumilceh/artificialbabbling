'''
Created on Jan 24, 2017

@author: Juan Manuel Acevedo Valle
'''

from explauto.interest_model import  interest_models
import numpy as np

class PARAMS(object):
    def __init__(self):
        pass

class explauto_IM(object):
    '''
    Implemented for non-parametric models
    '''
    def __init__(self, agent, model_type, competence_func):
        conf = generateConfigurationExplauto(agent)
        self.conf = conf

                #-------------------------------------- ['discretized_progress',
                #------------------------------------------------------- 'tree',
                #----------------------------------------------------- 'random',
                #------------------------------------------ 'miscRandom_global',
                #------------------------------------------ 'gmm_progress_beta',
                #------------------------------------------- 'miscRandom_local']
                
        self.model = SensorimotorModel.from_configuration(conf, model_type, model_conf)
        
        self.params = PARAMS()
        self.params.sm_step = 1 #only ok with non-parametric
       
    def getMotorCommand(self,Agent,sensor_goal=None):          
        if sensor_goal == None:
            sensor_goal = Agent.sensor_goal  #s_g
        
        Agent.motor_command = self.model.inverse_prediction(sensor_goal)
        return  Agent.motor_command
    
    def train(self, simulation_data):
        m = simulation_data.motor_data.data.iloc[-1]
        s = simulation_data.sensor_data.data.iloc[-1]
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
    conf.s_maxs = agent.max_sensor_values
    conf.s_mins = agent.min_sensor_values
   
    n_motor = agent.n_motor;
    n_sensor = agent.n_sensor;
    
    conf.m_ndims = n_motor
    conf.s_ndims = n_sensor

    conf.m_dims = np.arange(0, n_motor, 1).tolist()
    conf.s_dims = np.arange(n_motor, n_motor+n_sensor, 1).tolist()

    conf.bounds = np.zeros((2, n_motor + n_sensor))
    conf.bounds[0,:] = np.array([conf.m_mins, conf.s_mins]).flatten()
    conf.bounds[1,:] = np.array([conf.m_maxs, conf.s_maxs]).flatten()
    return conf
    
    

    
    
    