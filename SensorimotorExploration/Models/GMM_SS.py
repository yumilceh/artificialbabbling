'''
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
'''
from GeneralModels.Mixture import GMM
import numpy as np
import pandas as pd
class GMM_SS(object):
    '''
    classdocs
    '''


    def __init__(self, Agent, n_gauss_components, alpha=0.1, ss_step=400):
        '''
        Constructor
        '''
        self.size_data=Agent.n_motor+Agent.n_somato
        self.motor_names=Agent.motor_names;
        self.somato_names=Agent.somato_names;
        self.GMM=GMM(n_gauss_components)
        self.alpha=alpha
        self.ss_step=ss_step
        
        
    def train(self,simulation_data):
        train_data_tmp=pd.concat([simulation_data.motor_data.data, simulation_data.somato_data.data], axis=1)
        self.GMM.train(train_data_tmp.as_matrix(columns=None))
        
    def trainIncrementalLearning(self,simulation_data):
        ss_step=self.ss_step
        alpha=self.alpha
        motor_data_size=len(simulation_data.motor_data.data.index)
        motor_data=simulation_data.motor_data.data[motor_data_size-ss_step:-1]
        somato_data_size=len(simulation_data.somato_data.data.index)
        somato_data=simulation_data.somato_data.data[somato_data_size-ss_step:-1]
        new_data=pd.concat([motor_data,somato_data],axis=1)
        self.GMM.trainIncrementalLearning(new_data, alpha)
        
        
    def predictPriprioceptiveEffect(self,Agent):
        n_motor=Agent.n_motor;
        n_somato=Agent.n_somato;
        motor_command=Agent.motor_command  #s_g
        m_dims=np.arange(0, n_motor-1, n_motor)
        s_dims= np.arange(n_motor, n_motor+n_somato-1, n_somato),
        Agent.motor_command=self.GMM.predict(s_dims, m_dims, motor_command)