'''
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
'''
from Models.GeneralModels.Mixture import GMM
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
        train_data_tmp=pd.concat([simulation_data.motor_data.data, simulation_data.somato_data.data], axis=1)
        self.model.train(train_data_tmp.as_matrix(columns=None))
        
    def trainIncrementalLearning(self,simulation_data):
        ss_step=self.params.ss_step
        alpha=self.params.alpha
        motor_data_size=len(simulation_data.motor_data.data.index)
        motor_data=simulation_data.motor_data.data[motor_data_size-ss_step:-1]
        somato_data_size=len(simulation_data.somato_data.data.index)
        somato_data=simulation_data.somato_data.data[somato_data_size-ss_step:-1]
        new_data=pd.concat([motor_data,somato_data],axis=1)
        self.model.trainIncrementalLearning(new_data, alpha)
        
        
    def predictPriprioceptiveEffect(self,Agent):
        n_motor=Agent.n_motor;
        n_somato=Agent.n_somato;
        motor_command=Agent.motor_command  #s_g
        m_dims=np.arange(0, n_motor-1, n_motor)
        s_dims= np.arange(n_motor, n_motor+n_somato-1, n_somato),
        Agent.motor_command=self.model.predict(s_dims, m_dims, motor_command)