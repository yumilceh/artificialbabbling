"""
Created on Abr 6, 2016
Validate the DIVA agent used for the Epirob 2017's paper
but using divapy
@author: Juan Manuel Acevedo Valle
"""
import numpy as np
import numpy.linalg as la

from SensorimotorExploration.Systems.Diva2016a import Diva2016a as Divaml
from SensorimotorExploration.Systems.Diva2016b import Diva2016b as Divapy
from SensorimotorExploration.Algorithm.utils.functions import get_random_motor_set

if __name__  == '__main__':
    divaml = Divaml()
    divapy = Divapy()
    n_samples = 1000

    actions = get_random_motor_set(divapy,n_samples)

    for i in range(n_samples):
        divaml.set_action(actions[i,:])
        divapy.set_action(actions[i,:])

        divaml.executeMotorCommand()
        divapy.executeMotorCommand()

        err_aud = la.norm(np.subtract(divapy.sensor_out,divaml.sensor_out))
        err_som = la.norm(np.subtract(divapy.somato_out,divaml.somato_out))

        if err_aud>1e-4:
            print('Error in auditory ouput matching')
        if err_som > 1e-4:
            print('Error in somato ouput matching')
# Time running in Matlab is 184.061073065. Time running in Python is 85.1974360943.