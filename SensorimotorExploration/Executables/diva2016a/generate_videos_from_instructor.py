'''
Created on Sept 8, 2016

@author: Juan Manuel Acevedo Valle
'''
import os,sys
import numpy as np
#=======================================================================================
#
# Experiment report and videos
#
#=======================================================================================
if __name__ == '__main__' and True:
     
    sys.path.append(os.getcwd())
    from SensorimotorExploration.Systems.Diva2016a import Instructor
    import h5py
    
    diva_output_scale=[100.0,500.0,1500.0,3000.0]
    file_name = '../../Systems/datasets/german_dataset_2'

    system = Instructor()
    system.change_dataset(file_name + '.h5')
    with  open(file_name + '.txt', "r") as text_file:
        seq = text_file.read()
        seq = seq.split('\n')[:-1]

    save_to = 'german_instructor/'
    for i in range(system.n_units):
        system.set_action(system.data.motor.data.iloc[i])
        system.executeMotorCommand()
        system.getVocalizationVideo(show=0, file_name=save_to + 'vt' + seq[i])  # no extension in files

