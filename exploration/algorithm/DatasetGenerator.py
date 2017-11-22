'''
Created on Feb 18, 2017

@author: Juan Manuel Acevedo Valle

'''
import numpy as np
import datetime 
import pandas as pd

from ..data.data import SimulationData, load_sim_h5
from ..algorithm.utils.functions import get_random_motor_set

now = datetime.datetime.now().strftime("DSG_%Y_%m_%d_%H_%M_")

class OBJECT(object):
    def __init__(self):
        pass
       
class DatasetGenerator(object):
    '''
        This class is used to generate randoms data sets of sensory spaces that later can be used for 
        the evaluation of models. It is prepared also to take simulation files and create new datasets
        or enrich existing ones.
    '''

    def __init__(self,  system,  
                        min_dist = 0.1,
                        n_experiments = 1000,
                        random_seed = np.random.random((1,1)),
                        n_save_data = 100,
                        file_prefix = now
                        ): 
        '''
        system,                                         sensorimotor System
        min_dist = 0.1,                                 Minimum distance between samples
        n_experiments = 1000,                           Number of samples to be explored
        random_seed = np.random.random((1,1)),          Random seed (NOT WORKING YET)
        n_save_data = 100,                              Save data each given number of samples  
        file_prefix = now.strftime("DSG_%Y_%m_%d_%H_%M_")
        '''
        self.params = OBJECT()
        self.params.n_experiments = n_experiments
        self.params.random_seed = random_seed
        
        self.params.n_save_data = n_save_data
        
        self.system = system
        self.min_dist = min_dist
    
        self.data = SimulationData(self.system)
        self.data.file_prefix=file_prefix
           
    def generete_dataset(self):    
        print('Exploring...')
        n_experiments = self.params.n_experiments
        n_save_data = self.params.n_save_data;
        motor_commands =  get_random_motor_set(self.system,
                                               n_experiments)    
        
        for i in range(n_experiments):
            self.system.set_action(motor_commands[i, :])
            self.system.execute_action()
            self.data.append_data(self.system)
            if (np.mod(i,n_save_data) == 0):
                print('Saving data in sample {}'.format(i))
                self.data.save_data(self.data.file_prefix + 'data.h5')
        
        
    def reduce_dataset(self):
        min_dist = self.min_dist
        data_tmp = self.data.copy(self.system)
        data_tmp = reduce_(data_tmp, min_dist, self.system)
        return data_tmp
        
def mix_datasets(dataset1, dataset2):
    pass
        
def reduce_file_dataset(self, system, min_dist, file_name=None):
    if type(None) == type(file_name):
        file_name = self.data.file_prefix +'data.h5'
    
    data_tmp = load_sim_h5(file_name, system)
    data_tmp = reduce_(data_tmp, min_dist, system)
    return data_tmp    
        
def reduce_(data, min_distance, system):
    change = True
    sensor_data_as_matrix = data.sensor.data.as_matrix()
    motor_data_as_matrix = data.motor.data.as_matrix()

    for i in range(sensor_data_as_matrix.shape[0]):
        print(i)
        if sensor_data_as_matrix.shape[0]<i:
            break
        for j in reversed(range(i+1,sensor_data_as_matrix.shape[0])):
            distance_ij = np.linalg.norm(sensor_data_as_matrix[i,:]-sensor_data_as_matrix[j,:])
            if distance_ij < min_distance:
                sensor_data_as_matrix = np.delete(sensor_data_as_matrix, j, 0)
                motor_data_as_matrix = np.delete(motor_data_as_matrix, j, 0)
                    
    dataset = SimulationData(system)
    dataset.sensor.data = pd.DataFrame(sensor_data_as_matrix)
    dataset.motor.data = pd.DataFrame(motor_data_as_matrix)
    return dataset
