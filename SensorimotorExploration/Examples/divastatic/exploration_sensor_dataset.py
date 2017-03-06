'''
Created on Feb 17, 2017

@author: Juan Manuel Acevedo Valle
'''
import sys, datetime
import numpy as np

'''
#=======================================================================================
#
# ORIGINAL EXPERIMENT: Script to generate the recheable space of vowels using the 
# diva vocal tract 
#
#=======================================================================================
'''
explore = True
report = False
now = datetime.datetime.now().strftime("DSG_%Y_%m_%d_%H_%M_")
file_name = "exploration_" + now

if __name__ == '__main__' and explore:
    from SensorimotorExploration.Systems.DivaStatic import DivaStatic
    from SensorimotorExploration.Algorithm.DatasetGenerator import DatasetGenerator as Generator

    n_experiments = 1000000
    min_dist = 0.5
    n_save = 50000  # Save data each n_save samples

    system = DivaStatic()

    generator = Generator(system,
                          min_dist=min_dist,
                          n_experiments=n_experiments,
                          random_seed=np.random.random((1, 1)),
                          n_save_data=n_save,
                          file_prefix=file_name)

    generator.generete_dataset()
    reduced_data = generator.reduce_dataset()
    reduced_data.saveData('reduced_' + file_name)
