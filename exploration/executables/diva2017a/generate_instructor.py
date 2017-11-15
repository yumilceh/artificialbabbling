"""
Created on Mar 20, 2017

@author: Juan Manuel Acevedo Valle
"""
import h5py
import itertools
import numpy as np

# Adding libraries##
from exploration.systems.Diva2017a import Diva2017a as System
from exploration.data.data import SimulationData_v2 as SimulationData

if __name__ == '__main__':

    system = System()
    data = SimulationData(system)

    file = '../../systems/datasets/german_vowels_dataset_1.h5'
    f = h5py.File(file, 'r')
    keys = f.keys()

    comb_keys = itertools.product(keys, keys)
    vowel_order = ''
    for k in comb_keys:
        m1 = f[k[0]][:].flatten()
        m2 = f[k[1]][:].flatten()
        m = np.concatenate([m1,m2])
        system.set_action(m)
        system.execute_action()
        data.append_data(system)
        vowel_order += k[0] + '-' + k[1] + '\n'

    for k in keys:
        m1 = f[k][:].flatten()
        m2 = f[k][:].flatten() * 0.
        m = np.concatenate([m1,m2])
        system.set_action(m)
        system.execute_action()
        data.append_data(system)
        vowel_order += k + '-' + '\n'

    for k in keys:
        m1 = f[k][:].flatten() * 0.
        m2 = f[k][:].flatten()
        m = np.concatenate([m1, m2])
        system.set_action(m)
        system.execute_action()
        data.append_data(system)
        vowel_order += '-' + k + '\n'

    data.cut_final_data()
    data.save_data('../../systems/datasets/german_dataset_somato.h5')
    with open('../../systems/datasets/german_dataset_somato.txt', "w") as text_file:
        text_file.write(vowel_order)


