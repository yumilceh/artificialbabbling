'''
Created on May 26, 2016

@author: Juan Manuel Acevedo Valle
'''

import numpy as np
import numpy.linalg as linalg
def comp_mix(agent,sensor_space = 'sensor'):
    y = getattr(agent,  sensor_space+'_out')
    y_g = getattr(agent, sensor_space+'_goal')
    agent.competence_result = comp_mix_expl(y_g, y)

def comp_Moulin2013(agent, sensor_space = 'sensor'):
    y = getattr(agent,  sensor_space+'_out')
    y_g = getattr(agent, sensor_space+'_goal')
    agent.competence_result = comp_Moulin2013_expl(y_g, y)

def comp_Baraglia2015(agent,sensor_space = 'sensor'):
    y = getattr(agent,  sensor_space+'_out')
    y_g = getattr(agent, sensor_space+'_goal')
    agent.competence_result = comp_Baraglia2015_expl(y_g, y)

def comp_mix_expl(target, reached, dist_min=0., dist_max = 1.):
    err = np.asarray(target)-np.asarray(reached)
    return  exp_norm_error_exp(err)
    
def comp_Moulin2013_expl(target, reached, dist_min=0., dist_max = 1.):
    err = np.asarray(target)-np.asarray(reached)
    return  exp_norm_error(err)

def comp_Baraglia2015_expl(target, reached, dist_min=0., dist_max = 1.):
    err = np.asarray(target)-np.asarray(reached)
    return exp_norm_moderate_error(err)

def exp_norm_error(err):
    err_norm = linalg.norm(err)
    return np.exp(-err_norm)

def exp_norm_moderate_error(err):
    center = 1.2   # 1.2: 1.4; 1.:Peor; 1.5:Peor
    sigma = 0.5   # 1: 1.4, 2 worse,  1.5, 0.8
    err_norm = linalg.norm(err)
    return np.exp(-np.power(err_norm - center, 2) / (2 * sigma * sigma)) / (sigma * np.sqrt(2 * np.pi))

def exp_norm_error_exp(err):
    center = 0.5  #0.5
    sigma = 0.7   #0.7,
    exp_err_norm = exp_norm_error(err)
    return np.exp(-np.power(exp_err_norm - center, 2) / (2 * sigma * sigma)) / (sigma * np.sqrt(2 * np.pi))