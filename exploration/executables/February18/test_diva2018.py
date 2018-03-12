"""
Created on Novemeber,2017
@author: Juan Manuel Acevedo Valle
"""

from exploration.systems.diva2018 import Diva2018
from exploration.systems.Diva2017a import Diva2017a
from exploration.algorithm.utils.functions import get_random_motor_set

n_samples = 100
system= Diva2018(sensori_out='mfcc')
system_base = Diva2017a()
motor_commands = get_random_motor_set(system, n_samples)

#####Code to check mfcc output
# for i in range(n_samples):
#     system.set_action(motor_commands[i])
#     system.execute_action()
#     print(len(system.sensor_out))
#     print(system.sensor_out)

##### Code to compare wuth old implementation
for i in range(n_samples):
    system.set_action(motor_commands[i])
    system.execute_action()
    system_base.set_action(motor_commands[i])
    system_base.execute_action()
    print(system.cons_out-system_base.cons_out)
    # if sum(system.sensor_out)>0:
    #     pass
