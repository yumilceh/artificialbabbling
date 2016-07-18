'''
Created on May 26, 2016

@author: Juan Manuel Acevedo Valle
'''

import numpy as np
import numpy.linalg as linalg

def get_competence_Moulin2013(agent):
    y=agent.sensorOutput
    y_g=agent.sensor_goal    
    err_norm=linalg.norm(np.asarray(y_g)-np.asarray(y))
    c=np.exp(-err_norm)
    agent.competence_result=c
    
def get_competence_Baraglia2015(agent):
    sigma=0.1
    y=agent.sensorOutput
    y_g=agent.sensor_goal    
    err_norm=linalg.norm(np.asarray(y_g)-np.asarray(y))
    c = np.exp(-np.power(err_norm,2)/(2*sigma*sigma))/(sigma*np.sqrt(2*np.pi))
    agent.competence_result=c