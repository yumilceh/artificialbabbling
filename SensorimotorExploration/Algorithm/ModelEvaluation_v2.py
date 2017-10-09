"""
Created on Jun 30, 2016

@author: Juan Manuel Acevedo Valle
"""

import numpy as np
import random

from ..DataManager.SimulationData import SimulationData_v2, load_sim_h5_v2
from .utils.data_storage_funcs import saveSimulationData, loadSimulationData
from .utils.functions import get_random_sensor_set

from .utils.competence_funcs import comp_Moulin2013


class PARAMS(object):
    def __init__(self):
        pass


class SM_ModelEvaluation(object):
    """
        This class uses data in order to estimate a sensorimotor model and evaluate it.
    """

    def __init__(self, system,
                 model,
                 comp_func=comp_Moulin2013,
                 data=None,
                 file_prefix='',
                 ratio_samples_val=0.2):
        """
            Initialize
        """
        self.agent = system
        self.data = PARAMS()
        self.data = data
        self.comp_func = comp_func

        
        self.file_prefix = file_prefix
        self.model = model
        self.ratio_samples_val = ratio_samples_val

    def load_eval_dataset(self, file_name):
        self.dataset_file = file_name
        self.data, foo = load_sim_h5_v2(file_name)
        n_samples = len(self.data.sensor.data)
        self.n_samples_val = n_samples
        self.random_indexes_val = range(n_samples)


    def set_eval_dataset(self, data):
        n_samples = len(data.sensor.data)
        self.data = data
        self.n_samples_val = n_samples
        self.random_indexes_val = range(n_samples)

    def evaluate(self, saveData=False, space=None):
        if space is None:
            space = 'sensor'
        # Validation against Validation set
        validation_valSet_data = SimulationData_v2(self.agent, prelocated_samples=self.n_samples_val+1)
        progress = 1
        print('Evaluating model with validation data set of {} samples...'.format(self.n_samples_val))
        for i in self.random_indexes_val:
            #  print('Testing using sample {current} of {total} in the validation set'.format(current=progress,
            #                                                                     total=self.n_samples_val))  #Slow
            y_ = getattr(self.data, space).data.iloc[i].as_matrix()

            setattr(self.agent, space+'_goal',  y_)
            self.model.get_action(self.agent)
            self.agent.execute_action()
            self.comp_func(self.agent, sensor_space = space)
            validation_valSet_data.appendData(self.agent)
            progress = progress + 1
        print('Evaluation has been finished.')

        # validation_trainSet_data.cut_final_data()
        validation_valSet_data.cut_final_data()

        if (saveData):
            validation_valSet_data.saveData(self.file_prefix + space +'_eva_valset.h5')
            return validation_valSet_data
        else:
            return validation_valSet_data

