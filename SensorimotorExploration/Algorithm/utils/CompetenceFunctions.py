'''
Created on May 26, 2016

@author: Juan Manuel Acevedo Valle
'''

import numpy as np
import numpy.linalg as linalg

def comp_Moulin2013(agent):
    y=agent.sensor_out
    y_g=agent.sensor_goal      
    agent.competence_result = comp_Moulin2013_expl(y_g, y)
    
def get_competence_Baraglia2015(agent):
    y=agent.sensor_out
    y_g=agent.sensor_goal    
    agent.competence_result = get_competence_Baraglia2015_explauto(y_g, y)
    
def comp_Moulin2013_expl(target, reached, dist_min=0., dist_max = 1.):
    err_norm=linalg.norm(np.asarray(target)-np.asarray(reached))
    return np.exp(-err_norm)

def get_competence_Baraglia2015_explauto(target, reached, dist_min=0., dist_max = 1.):
    sigma=0.1
    err_norm=linalg.norm(np.asarray(target)-np.asarray(reached))
    return np.exp(-np.power(err_norm,2)/(2*sigma*sigma))/(sigma*np.sqrt(2*np.pi))
    