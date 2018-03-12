"""
Created on March, 17th 2017

@author: Juan Manuel Acevedo Valle
"""
import datetime
import numpy as np

# explore = False  # Explore  for dataset
# explore_vowels  =  False
# report = False
# file_conv = False
# data_set_vowels = True
now = datetime.datetime.now().strftime("DSG_%Y_%m_%d_%H_%M_") # Data set Goal Space
file_name = "exploration_" + now

if __name__ == '__main__':
    from exploration.systems.trash.Parabola_Test import ParabolicRegion as System
    from exploration.algorithm.DatasetGenerator import DatasetGenerator as Generator

    n_experiments = 100000
    min_dist = 0.3
    n_save = 50000  # Save data each n_save samples

    system = System()

    generator = Generator(system,
                          min_dist=min_dist,
                          n_experiments=n_experiments,
                          random_seed=np.random.random((1, 1)),
                          n_save_data=n_save,
                          file_prefix=file_name)

    generator.generete_dataset()
    reduced_data = generator.reduce_dataset()
    reduced_data.save_data('reduced_' + file_name)