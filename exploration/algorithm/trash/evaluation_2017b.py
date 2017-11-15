"""
Created on Jun 30, 2016

@author: Juan Manuel Acevedo Valle
"""

import numpy as np
import random

from exploration.data.data import SimulationData_v2 as SimulationData
from exploration.data.data import load_sim_h5
from exploration.algorithm.utils.data_storage_funcs import loadSimulationData
from exploration.algorithm.utils.functions import get_random_sensor_set

from exploration.algorithm.utils.competence_funcs import comp_Moulin2013


class PARAMS(object):
    def __init__(self):
        pass


class SMModelEvaluation(object):
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

    def loadEvaluationDataSet(self, file_name):
        self.data = load_sim_h5(file_name, self.agent)
        n_samples = len(self.data.sensor.data)
        self.n_samples_val = n_samples
        self.random_indexes_val = range(n_samples)

    def setValidationEvaluationSets(self):

        if isinstance(self.data, int):
            n_samples = self.data
            rnd_data = get_random_sensor_set(self.agent, n_samples)
            data = SimulationData(self.agent);
            data.sensor.append_data(rnd_data)
            self.data = data
            self.n_samples_val = n_samples
            self.random_indexes_val = range(n_samples)
        else:
            n_samples = len(self.data.motor.data)
            self.n_samples = n_samples
            ratio_samples_val = self.ratio_samples_val
            n_samples_val = np.ceil(ratio_samples_val * n_samples).astype(int)
            n_samples_train = n_samples - n_samples_val;
            random_indexes_val = random.sample(xrange(0, n_samples), n_samples_val)
            random_indexes_train = [index for index in range(0, n_samples) if index not in random_indexes_val]
            self.n_samples_val = n_samples_val
            self.n_samples_train = n_samples_train
            self.random_indexes_val = random_indexes_val
            self.random_indexes_train = random_indexes_train

    def trainModel(self):
        # Training
        progress = 1;
        sm_step = self.model.params.sm_step
        steps = np.arange(0, self.n_samples + 1, sm_step)
        for i in range(len(steps) - 1):
            data_tmp = self.data.cut_data(self.agent, steps[i], steps[i + 1])
            print('Training with block {current} of {total}'.format(current=progress, total=len(steps)))
            self.model.train_incremental(data_tmp)
            progress = progress + 1;

    def evaluateModel(self, saveData=False, eva_train_set=0):
        # Validation against Training set
        if (eva_train_set > 0):
            n_samples_evatrain = np.ceil(eva_train_set * self.n_samples_train).astype(int)
            random_indexes_evatrain = random.sample(self.random_indexes_train, n_samples_evatrain)
            validation_trainSet_data = SimulationData(self.agent,prelocated_samples=n_samples_evatrain+1)
            progress = 1;
            for i in random_indexes_evatrain:
                #  print('Testing using sample {current} of {total} in the training set'.format(current=progress,
                #                         total=n_samples_evatrain)) #  Slow
                y_ = self.data.sensor.data.iloc[i].as_matrix()

                self.agent.sensor_goal = y_
                self.model.get_action(self.agent)
                self.agent.getMotorDynamics()
                self.agent.execute_action()
                self.comp_func(self.agent)
                validation_trainSet_data.append_data(self.agent)
                progress = progress + 1;

            if (saveData):
                validation_trainSet_data.save_data([self.file_prefix + 'eval_trainset.h5'])

        # Validation against Validation set
        validation_valSet_data = SimulationData(self.agent, prelocated_samples=self.n_samples_val+1)
        progress = 1;
        print('Evaluating model with validation data set of {} samples...'.format(self.n_samples_val))
        for i in self.random_indexes_val:
            #  print('Testing using sample {current} of {total} in the validation set'.format(current=progress,
            #                                                                     total=self.n_samples_val))  #Slow
            y_ = self.data.sensor.data.iloc[i].as_matrix()

            self.agent.sensor_goal = y_
            self.model.get_action(self.agent)
            self.agent.execute_action()
            self.comp_func(self.agent)
            validation_valSet_data.append_data(self.agent)
            progress = progress + 1;
        print('Evaluation has been finished.')

        # validation_trainSet_data.cut_final_data()
        validation_valSet_data.cut_final_data()

        if (saveData):
            validation_valSet_data.save_data(self.file_prefix + 'eva_valset.h5')
            if (eva_train_set > 0):
                # saveSimulationData([self.file_prefix + 'validation_trainSet_data.h5',
                #                     self.file_prefix + 'validation_valSet_data.h5'],
                #                    [self.file_prefix + 'validation_results.tar.gz'])
                return validation_trainSet_data, validation_valSet_data
            else:
                # saveSimulationData([self.file_prefix + 'validation_valSet_data.h5'],
                #                    self.file_prefix + 'validation_results.tar.gz')
                return validation_valSet_data
        else:
            if (eva_train_set > 0):
                return validation_trainSet_data, validation_valSet_data
            else:
                return validation_valSet_data


def generateTestDatafromrawData(data_in, min_distance=0.01):
    ''' This function takes raw data and reduces it to data with a min_distance between samples 
        data_in: numpy matrix (n_samples x n_dims)
        min_distance is a float
    '''
    change = True

    data = data_in.copy()

    while change:  # No optimal
        change = False
        for i in range(data.shape[0]):
            if change == True:
                break
            for j in range(i + 1, data.shape[0]):
                distance_ij = np.linalg.norm(data[i, :] - data[j, :])
                if distance_ij < min_distance:
                    data = np.delete(data, j, 0)
                    change = True
                    break

    return data


def loadEvaluationResults(file_name, agent):
    return loadSimulationData(file_name, agent)
