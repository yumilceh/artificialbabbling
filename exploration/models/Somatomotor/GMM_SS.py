'''
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
'''
from exploration.models.GeneralModels.Mixture import GMM
import numpy as np
import pandas as pd

class PARAMS(object):
    def __init__(self):
        pass;

class GMM_SS(object):
    '''
    classdocs
    '''
    def __init__(self, Agent, n_gauss_components, alpha=0.1, ss_step=400):
        '''
        Constructor
        '''
        
        self.params=PARAMS()
        self.params.size_data=Agent.n_motor+Agent.n_somato
        self.params.motor_names=Agent.motor_names;
        self.params.somato_names=Agent.somato_names;
        self.params.alpha=alpha
        self.params.ss_step=ss_step
        
        self.model=GMM(n_gauss_components)
        
    def train(self,simulation_data):
        train_data_tmp=pd.concat([simulation_data.motor.data, simulation_data.somato.data], axis=1)
        self.model.train(train_data_tmp.as_matrix(columns=None))
        
    def trainIncrementalLearning(self,simulation_data):
        ss_step=self.params.ss_step
        alpha=self.params.alpha
        motor_data_size=len(simulation_data.motor.data.index)
        motor_data=simulation_data.motor.data[motor_data_size-ss_step:-1]
        somato_data_size=len(simulation_data.somato.data.index)
        somato_data=simulation_data.somato.data[somato_data_size-ss_step:-1]
        new_data=pd.concat([motor_data,somato_data],axis=1)
        self.model.train_incremental(new_data, alpha)
        
    def predictProprioceptiveEffect(self,Agent,motor_command = None):
        n_motor=Agent.n_motor;
        n_somato=Agent.n_somato;
        
        if motor_command == None:
            motor_command=Agent.motor_command  #s_g
        
        m_dims=np.arange(0, n_motor, 1)
        s_dims=np.arange(n_motor, n_motor+n_somato, 1)
        
        
        
        Agent.proprioceptive_prediction=self.model.predict(s_dims, m_dims, motor_command.flatten())
        return boundProprioceptivePrediction(Agent,self.model.predict(s_dims, m_dims, motor_command.flatten()))
    
def boundProprioceptivePrediction(Agent, proprioceptive_prediction):
    n_somato=Agent.n_somato;
    min_somato_values = Agent.min_somato_values
    max_somato_values = Agent.max_somato_values
    somato_threshold = Agent.somato_threshold
    for i in range(n_somato):
        if ((proprioceptive_prediction[i] < min_somato_values[i]) or (proprioceptive_prediction[i] <= somato_threshold)):
            proprioceptive_decision = 0
   
        elif ((proprioceptive_prediction[i] > max_somato_values[i]) or (proprioceptive_prediction[i] > somato_threshold)):
            proprioceptive_decision = 1
    return proprioceptive_decision