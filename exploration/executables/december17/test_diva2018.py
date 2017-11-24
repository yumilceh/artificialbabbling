"""
Created on Novemeber,2017
@author: Juan Manuel Acevedo Valle
"""

from exploration.systems.diva2018 import Diva2018
from exploration.algorithm.utils.functions import get_random_motor_set

n_samples = 100
system= Diva2018()
motor_commands = get_random_motor_set(system, n_samples)
for i in range(n_samples):
    system.set_action(motor_commands[i])
    system.execute_action()
    print(system.sensor_out)
    if sum(system.sensor_out)>0:
        pass
