"""
Created on Abr 6, 2016
Validate the DIVA agent used for the Epirob 2017's paper
but using divapy
@author: Juan Manuel Acevedo Valle
"""
import numpy as np
import numpy.linalg as la
import time

from SensorimotorExploration.Systems.Diva2016a import Diva2016a as Divaml
from SensorimotorExploration.Systems.Diva2016b import Diva2016b as Divapy
from SensorimotorExploration.Algorithm.utils.functions import get_random_motor_set

if __name__  == '__main__':
    divaml = Divaml()
    divapy = Divapy()
    n_samples = 1000

    actions = get_random_motor_set(divapy,n_samples)

    print('Running test is matlab')
    t = time.time()
    for i in range(n_samples):
        divaml.set_action(actions[i,:])
        divaml.executeMotorCommand()
    elpsed_ml = time.time() - t

    print('Running test is python')
    t = time.time()
    for i in range(n_samples):
        divapy.set_action(actions[i,:])
        divapy.executeMotorCommand()
    elpsed_py = time.time() - t

    print('Time running in Matlab is {}. Time running in Python is {}.'.format(elpsed_ml,elpsed_py))