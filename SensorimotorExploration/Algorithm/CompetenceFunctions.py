'''
Created on May 26, 2016

@author: Juan Manuel Acevedo Valle
'''

import numpy as np
import numpy.linalg as linalg

def get_competence_Moulin2013(agent):
    y=agent.sensorOutput
    y_g=agent.sensor_goal      
    agent.competence_result = get_competence_Moulin2013_explauto(y_g, y)
    
def get_competence_Baraglia2015(agent):
    y=agent.sensorOutput
    y_g=agent.sensor_goal    
    agent.competence_result = get_competence_Baraglia2015_explauto(y_g, y)
    
def get_competence_Moulin2013_explauto(target, reached, dist_min=0., dist_max = 1.):
    err_norm=linalg.norm(np.asarray(target)-np.asarray(reached))
    return np.exp(-err_norm)

def get_competence_Baraglia2015_explauto(target, reached):
    sigma=0.1
    err_norm=linalg.norm(np.asarray(traget)-np.asarray(reached))
    return np.exp(-np.power(err_norm,2)/(2*sigma*sigma))/(sigma*np.sqrt(2*np.pi))
    