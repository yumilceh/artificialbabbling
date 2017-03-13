"""
Created on May 26, 2016

@author: Juan Manuel Acevedo Valle
"""
import numpy as np

def generate_motor_grid(system, n_samples):
    """ Currently works for 2D motor systems"""
    xmin = system.min_motor_values[0]
    xmax = system.max_motor_values[0]
    ymin = system.min_motor_values[1]
    ymax = system.max_motor_values[1]

    np_dim = np.ceil(np.sqrt(n_samples))

    m1, m2 = np.meshgrid(np.linspace(xmin,xmax,np_dim), np.linspace(ymin,ymax,np_dim))
    #grid = np.vstack([X.ravel(), Y.ravel()])
    return m1, m2

def get_random_motor_set(system, n_samples,
                         min_values=None,
                         max_values=None,
                         random_seed=np.random.randint(999, size=(1, 1))):
    """
        All vector inputs must be horizontal vectors
    """
    n_motor = system.n_motor

    raw_rnd_data = np.random.random((n_samples, n_motor))

    if min_values == None:
        min_values = system.min_motor_values
    if max_values == None:
        max_values = system.max_motor_values

    min_values = np.array(n_samples * [np.array(min_values)])
    max_values = np.array(n_samples * [np.array(max_values)])

    motor_commands = min_values + raw_rnd_data * (max_values - min_values)

    return motor_commands


def get_random_sensor_set(system, n_samples,
                          min_values=None,
                          max_values=None,
                          random_seed=np.random.randint(999, size=(1, 1))):
    """
        All vector inputs must be horizontal vectors
    """
    n_sensor = system.n_sensor

    raw_rnd_data = np.random.random((n_samples, n_sensor))

    if min_values == None:
        min_values = system.min_sensor_values
    if max_values == None:
        max_values = system.max_sensor_values

    min_values = np.array(n_samples * [np.array(min_values)])
    max_values = np.array(n_samples * [np.array(max_values)])

    sensor_commands = min_values + raw_rnd_data * (max_values - min_values)

    return sensor_commands
