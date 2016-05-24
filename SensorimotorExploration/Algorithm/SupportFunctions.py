'''
Created on May 23, 2016

@author: Juan Manuel Acevedo Valle
'''
import numpy as np
import numpy.linalg as linalg
def get_random_motor_set(agent,n_samples,
                                 min_values=None,
                                 max_values=None,
                                 random_seed=np.random.randint(999,size=(1,1))):
    '''
        All vector inputs must be horizontal vectors
    '''
    n_motor=agent.n_motor  
    
    raw_rnd_data=np.random.random((n_samples,n_motor))
    
    if min_values== None:
        min_values=agent.min_motor_values;
    if max_values== None:
        max_values=agent.max_motor_values;
         
    min_values=np.array(n_samples*[np.array(min_values)])
    max_values=np.array(n_samples*[np.array(max_values)])
    
    motor_commands=min_values+raw_rnd_data*(max_values-min_values)

    return motor_commands

def get_competence_Moulin2013(agent):
    y=agent.sensorOutput
    y_g=agent.sensor_goal    
    err_norm=linalg.norm(y_g-y)
    c=np.exp(-err_norm)
    agent.competence_result=c
