'''
Created on Feb 22, 2016

@author: Juan Manuel Acevedo Valle
'''
from GeneralModels.Mixture import GMM
class GMM_SM(object):
    '''
    classdocs
    '''


    def __init__(self, Agent, n_gauss_components):
        '''
        Constructor
        '''
        self.size_data=Agent.n_motor+Agent.n_sensor
        self.GMM=GMM(n_gauss_components)
        
        
    def train(self,tabular_data):
        self.GMM.train(tabular_data.data.as_matrix(columns=None)) 
        