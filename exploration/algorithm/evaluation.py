"""
Created on Jun 30, 2016
Modified on Nov 15, 2017

@author: Juan Manuel Acevedo Valle
"""

# import numpy as np
import os
from ..data.data import SimulationData, load_sim_h5
# from .utils.functions import get_random_sensor_set

from .utils.competence_funcs import comp_Moulin2013


class OBJECT(object):
    def __init__(self):
        pass

class Evaluation():
    """
        Class to evaluate models used for artificial autonomous development architectures
    """
    def __init__(self, system,
                 model,
                 comp_func=comp_Moulin2013,
                 data=None,
                 file_prefix='',
                 type='sensorimotor' #'sensorimotor', 'somatosensorimotor'
                 ):
        self.agent = system
        self.data = {}
        self.comp_func = comp_func
        file_prefix = file_prefix.replace('/', os.sep)
        self.file_prefix = file_prefix
        self.model = model

    def load_eval_dataset(self, file_name, name=None):
        file_name = file_name.replace('/', os.sep)
        if self.data is None:
            self.data = {}
        data, foo = load_sim_h5(file_name)
        key = name
        if key is None:
            key = file_name
        self.data.update({key: data})

    def set_eval_dataset(self, data, name=None):
        if isinstance(data, SimulationData):
            if name is None:
                keys = self.data.keys()
                i = 0
                while str(i) in keys():
                    i += 1
                name = str(i)
            self.data.update({name: data})
        else:
            self.data.update(data)

    def evaluate(self, save_data=False, space='sensor'):
        # Validation against evaluation_dataset
        evaluation_data = {}
        for key in self.data.keys():
            n_samples = len(self.data[key].sensor.data.iloc[:])
            evaluation_data.update({key: SimulationData(self.agent,
                                                        prelocated_samples=n_samples)})
            print('Evaluating model with {} ({} samples)...'.format(key, n_samples))
            for i in range(n_samples):
                y_ = getattr(self.data[key], space).data.iloc[i].as_matrix()
                setattr(self.agent, space+'_goal',  y_)
                self.model.get_action(self.agent)
                self.agent.execute_action()
                self.comp_func(self.agent, sensor_space = space)
                evaluation_data[key].append_data(self.agent)
            evaluation_data[key].cut_final_data()
            print('Evaluation with {} has been finished.'.format(key))

            if (save_data):
                evaluation_data[key].save_data(self.file_prefix + '_' + key + '_' + space + '_eva_valset.h5')
        return evaluation_data
