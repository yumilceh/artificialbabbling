'''
Created on May 26, 2016

@author: Juan Manuel Acevedo Valle
'''
import numpy as np

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

def get_random_sensor_set(agent,n_samples,
                                 min_values=None,
                                 max_values=None,
                                 random_seed=np.random.randint(999,size=(1,1))):
    '''
        All vector inputs must be horizontal vectors
    '''
    n_sensor=agent.n_sensor  
    
    raw_rnd_data=np.random.random((n_samples,n_sensor))
    
    if min_values== None:
        min_values=agent.min_sensor_values;
    if max_values== None:
        max_values=agent.max_sensor_values;
         
    min_values=np.array(n_samples*[np.array(min_values)])
    max_values=np.array(n_samples*[np.array(max_values)])
    
    sensor_commands=min_values+raw_rnd_data*(max_values-min_values)

    return sensor_commands
