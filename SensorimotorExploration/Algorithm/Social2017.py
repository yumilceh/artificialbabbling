"""
Created on Feb 20, 2017

@author: Juan Manuel Acevedo Valle
"""
# ===============================================================================
# from ..Algorithm.utils.CompetenceFunctions import get_competence_Moulin2013 as get_competence
# from Algorithm.utils.CompetenceFunctions import get_competence_Baraglia2015 as get_competence
# ===============================================================================
import numpy as np
from numpy import linalg
import datetime  # copy.deepcopy
from ..DataManager.SimulationData import SimulationData
from ..Algorithm.utils.RndSensorimotorFunctions import get_random_motor_set
from ..Algorithm.ModelEvaluation import SM_ModelEvaluation
from ..Algorithm.utils.StorageDataFunctions import saveSimulationData
# import logging
# logging.basicConfig(filename='cylinder.log', level=logging.DEBUG,\
#         format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
# logging.info("Volume of a Cylinder with radius {} and height {} is {}".format(args.radius, args.height, volume))

now = datetime.datetime.now().strftime("Social_%Y_%m_%d_%H_%M_")


class OBJECT(object):
    def __init__(self):
        pass


class DATA(object):
    def __init__(self, alg):
        self.init_motor_commands = get_random_motor_set(alg.learner,
                                                        alg.params.n_initialization_experiments)
        self.initialization_data_sm_ss = SimulationData(alg.learner)
        self.initialization_data_im = SimulationData(alg.learner)
        self.simulation_data = SimulationData(alg.learner)


class Social(object):
    """
    Approach to social learning of vowel like sounds
    """

    def __init__(self, learner,
                 models,
                 n_experiments,
                 competence_func,
                 n_initialization_experiments=100,
                 random_seed=np.random.random((1, 1)),
                 g_im_initialization_method='non-zero',  # 'non-zero' 'all' 'non-painful'
                 n_save_data=-1,
                 sm_all_samples=False,
                 evaluation = None,
                 eval_step = -1,
                 file_prefix=now):
        """
        Social(learner, instructor, models, n_experiments, competence_func)
            optional parameters:
            n_initialization_experiments=100,
            random_seed=np.random.random((1, 1)),
            g_im_initialization_method='non-zero',  # 'non-zero' 'all' 'non-painful'
            n_save_data=50000,
            sm_all_samples=False,
            evaluation=None,
            file_prefix=now
        """

        self.params = OBJECT()
        self.params.sm_all_samples = None
        self.params.n_initialization_experiments = n_initialization_experiments
        self.params.n_experiments = n_experiments
        self.params.random_seed = random_seed
        self.params.g_im_initialization_method = g_im_initialization_method

        if n_save_data == -1:
            n_save_data = np.floor(n_experiments/5)

        self.params.n_save_data = n_save_data

        self.params.sm_all_samples = sm_all_samples

        self.learner = learner
        self.initialization_models = OBJECT()
        self.models = models
        self.get_competence = competence_func

        self.data = DATA(self)
        self.data.file_prefix = file_prefix

        self.evaluation = evaluation
        self.evaluate = False
        if not type(evaluation) == type(None):
            self.evaluation_error = [np.infty]
            self.evaluate = True
            if eval_step == -1:
                eval_step = np.floor(n_experiments / 5)
        self.params.eval_step = eval_step

    def run(self, proprio = True):
        if proprio:
            self.run_proprio()
        else:
            self.run_simple()

    def run_proprio(self):
        n_init = self.params.n_initialization_experiments
        motor_commands = self.data.init_motor_commands

        print('SM Exploration (Proprio), Line 1: Initializing G_SM and G_SS')
        for i in range(n_init):
            self.learner.setMotorCommand(motor_commands[i, :])
            self.learner.executeMotorCommand()
            self.data.initialization_data_sm_ss.appendData(self.learner)
            # print('SM , Line 1: Initialize G_SM and G_SS, experiment: {} of {}'.format(i + 1,n_init)) # Slow
        self.models.f_sm.train(self.data.initialization_data_sm_ss)
        self.models.f_ss.train(self.data.initialization_data_sm_ss)
        self.initialization_models.f_sm = self.models.f_sm.model.return_copy()
        self.initialization_models.f_ss = self.models.f_ss.model.return_copy()
        print('SM Exploration (Proprio), Line 1: G_SM and G_SS were initialized')

        print('SM Exploration (Proprio), Line 1: First evaluation of G_SM and G_SS')
        if self.evaluate:
            self.evaluation.model = self.models.f_sm.return_copy()
            eval_data = self.evaluation.evaluateModel()
            error_ = np.linalg.norm(eval_data.sensor_goal_data.data - eval_data.sensor_data.data, axis=1)
            self.evaluation_error = np.append(self.evaluation_error, np.mean(error_))

        # print('Algorithm 1 (Proprioceptive), Line 1: Initialize G_SM and G_SS,
        # experiment {} of {}'.format(i + 1, n_init)) # Slow

        self.data.initialization_data_sm_ss.saveData(self.data.file_prefix + 'initialization_data_sm_ss.h5')

        g_im_initialization_method = self.params.g_im_initialization_method
        if g_im_initialization_method == 'non-zero':
            print('SM Exploration (Proprio),, Line 2: Initialize G_IM, Non-null sensory result considered ')
            sensor_goals = self.data.initialization_data_sm_ss.sensor_data.data.as_matrix()
            for i in range(n_init):
                # print('Algorithm 1 (Proprioceptive), Line 2: Initialize G_IM,
                # experiment: {} of {}'.format(i + 1, n_init))
                if (linalg.norm(sensor_goals[i]) > 0):
                    self.learner.sensor_goal = sensor_goals[i]
                    self.models.f_sm.getMotorCommand(self.learner)
                    self.learner.executeMotorCommand()
                    self.get_competence(self.learner)
                    self.data.initialization_data_im.appendData(self.learner)

        elif g_im_initialization_method == 'non-painful':
            print('SM Exploration (Proprio), Line 2: Initialize G_IM, Non-painful sensory result considered ')
            sensor_goals = self.data.initialization_data_sm_ss.sensor_data.data.as_matrix()
            proprio_data = self.data.initialization_data_sm_ss.somato_data.data.as_matrix()
            for i in range(n_init):
                # print('Algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM,
                # experiment: {} of {}'.format(i, n_init))
                if proprio_data[i] == 0:
                    self.learner.sensor_goal = sensor_goals[i]
                    self.models.f_sm.getMotorCommand(self.learner)
                    self.learner.executeMotorCommand()
                    self.get_competence(self.learner)
                    self.data.initialization_data_im.appendData(self.learner)


        elif g_im_initialization_method == 'all':
            print('SM Exploration (Proprio), Line 2: Initialize G_IM, All sensory result considered ')
            sensor_goals = self.data.initialization_data_sm_ss.sensor_data.data.as_matrix()
            for i in range(n_init):
                # print('Algorithm 1 (Proprioceptive), Line 2: Initialize G_IM,'
                #       ' experiment: {} of {}'.format(i + 1, n_init))
                self.learner.sensor_goal = sensor_goals[i]
                self.models.f_sm.getMotorCommand(self.learner)
                self.learner.executeMotorCommand()
                self.get_competence(self.learner)
                self.data.initialization_data_im.appendData(self.learner)

        self.data.initialization_data_im.saveData(self.data.file_prefix + 'initialization_data_im.h5')
        self.models.f_im.train(self.data.initialization_data_im)
        self.initialization_models.f_im = self.models.f_im.model.return_copy()
        print('SM Exploration (Proprio), Line 2: G_IM was initialized')

        n_save_data = self.params.n_save_data;
        n_experiments = self.params.n_experiments
        print('SM Exploration (Proprio), Lines 4-22: : Main simulation running...')
        for i in range(n_experiments):
            self.learner.sensor_goal = self.models.f_im.get_goal(self.learner, self.models.f_sm, self.models.f_ss)
            self.models.f_sm.getMotorCommand(self.learner)
            self.learner.executeMotorCommand()
            self.get_competence(self.learner)
            self.data.simulation_data.appendData(self.learner)

            ''' Train Interest Model'''
            if ((i + 1) % self.models.f_im.params.im_step) == 0:
                # print('Algorithm 1 (Proprioceptive), Line 4-22: Experiment: Training Model IM')
                if i < self.models.f_im.params.n_training_samples:
                    self.models.f_im.train(
                        self.data.initialization_data_im.mixDataSets(self.learner, self.data.simulation_data))
                else:
                    self.models.f_im.train(self.data.simulation_data)

            ''' Train Sensorimotor Model'''
            if ((i + 1) % self.models.f_sm.params.sm_step) == 0:
                # print('Algorithm 1 (Proprioceptive), Line 4-22: Experiment: Training Model SM')
                if (i < n_init or self.params.sm_all_samples):  ###BE CAREFUL WITH MEMORY
                    self.models.f_sm.trainIncrementalLearning(
                        self.data.simulation_data.mixDataSets(self.learner,
                                                              self.data.initialization_data_im.mixDataSets(self.learner,
                                                                                                           self.data.initialization_data_sm_ss)))
                else:
                    self.models.f_sm.trainIncrementalLearning(self.data.simulation_data)
                if not self.evaluation == None:
                    self.evaluation.model = self.models.f_sm
                    eval_data = self.evaluation.evaluateModel()
                    error_ = np.linalg.norm(eval_data.sensor_goal_data.data - eval_data.sensor_data.data, axis=1)
                    self.evaluation_error = np.append(self.evaluation_error, np.mean(error_))

            ''' Train Somatosensory model'''
            if ((i + 1) % self.models.f_ss.params.ss_step) == 0:
                # print('SM Exploration (Proprio), Line 4-22: Experiment: Training Model SS')
                if (i < n_init or self.params.sm_all_samples):  ###BE CAREFUL WITH MEMORY
                    self.models.f_ss.trainIncrementalLearning(
                        self.data.simulation_data.mixDataSets(self.learner,
                                                              self.data.initialization_data_im.mixDataSets(self.learner,
                                                                                                           self.data.initialization_data_sm_ss)))
                else:
                    self.models.f_sm.trainIncrementalLearning(self.data.simulation_data)

            # print('SM Exploration (Proprio), Line 4-22: Experiment: {} of {}'.format(i + 1, n_experiments)) # Slow
            if (np.mod(i, n_save_data) == 0):
                self.data.simulation_data.saveData(self.data.file_prefix + 'simulation_data.h5')
                print('SM Exploration (Proprio), Line 4-22: Experiment: Saving data at samples {} of {}'.format(i + 1, n_experiments))

        self.data.simulation_data.saveData('simulation_data.h5')
        saveSimulationData([self.data.file_prefix + 'initialization_data_sm_ss.h5',
                            self.data.file_prefix + 'initialization_data_im.h5',
                            self.data.file_prefix + 'simulation_data.h5'], 'simulation_data.tar.gz')

        print('SM Exploration (Proprio), Experiment was finished')

    def run_simple(self):
        if self.params.g_im_initialization_method is 'non-painful':
            print('G_IM, init: non-painful method not defined for non-proprioceptive agents.')
            print('G_IM, init: Switching to all samples method')
            self.params.g_im_initialization_method = 'all'
        n_init = self.params.n_initialization_experiments
        motor_commands = self.data.init_motor_commands

        print('SM Exploration (Simple), Line 1: Initializing G_SM')
        for i in range(n_init):
            self.learner.setMotorCommand(motor_commands[i, :])
            self.learner.executeMotorCommand()
            self.data.initialization_data_sm_ss.appendData(self.learner)
            # print('SM , Line 1: Initialize G_SM, experiment: {} of {}'.format(i + 1,n_init)) # Slow
        self.models.f_sm.train(self.data.initialization_data_sm_ss)
        self.models.f_ss.train(self.data.initialization_data_sm_ss)
        try:
            self.initialization_models.f_sm = self.models.f_sm.model.return_copy()
            self.initialization_models.f_ss = self.models.f_ss.model.return_copy()
        except AttributeError:
            self.initialization_models.f_sm = self.models.f_sm.model
            self.initialization_models.f_ss = self.models.f_ss.model

        print('SM Exploration (Simple), Line 1: G_SM awas initialized')

        print('SM Exploration (Simple), Line 1: First evaluation of G_SM')
        if self.evaluate:
            try:
                self.evaluation.model = self.models.f_sm.return_copy()
            except AttributeError:
                self.evaluation.model = self.models.f_sm
            eval_data = self.evaluation.evaluateModel()
            error_ = np.linalg.norm(eval_data.sensor_goal_data.data - eval_data.sensor_data.data, axis=1)
            self.evaluation_error = np.append(self.evaluation_error, np.mean(error_))

        # print('Algorithm 1 , Line 1: Initialize G_SM and G_SS,
        # experiment {} of {}'.format(i + 1, n_init)) # Slow

        self.data.initialization_data_sm_ss.saveData(self.data.file_prefix + 'initialization_data_sm.h5')

        g_im_initialization_method = self.params.g_im_initialization_method
        if g_im_initialization_method == 'non-zero':
            print('SM Exploration (Simple),, Line 2: Initialize G_IM, Non-null sensory result considered ')
            sensor_goals = self.data.initialization_data_sm_ss.sensor_data.data.as_matrix()
            for i in range(n_init):
                # print('Algorithm 1 (Proprioceptive), Line 2: Initialize G_IM,
                # experiment: {} of {}'.format(i + 1, n_init))
                if (linalg.norm(sensor_goals[i]) > 0):
                    self.learner.sensor_goal = sensor_goals[i]
                    self.models.f_sm.getMotorCommand(self.learner)
                    self.learner.executeMotorCommand()
                    self.get_competence(self.learner)
                    self.data.initialization_data_im.appendData(self.learner)

        elif g_im_initialization_method == 'all':
            print('SM Exploration (Simple), Line 2: Initialize G_IM, All sensory result considered ')
            sensor_goals = self.data.initialization_data_sm_ss.sensor_data.data.as_matrix()
            for i in range(n_init):
                # print('Algorithm 1 , Line 2: Initialize G_IM,'
                #       ' experiment: {} of {}'.format(i + 1, n_init))
                self.learner.sensor_goal = sensor_goals[i]
                self.models.f_sm.getMotorCommand(self.learner)
                self.learner.executeMotorCommand()
                self.get_competence(self.learner)
                self.data.initialization_data_im.appendData(self.learner)

        self.data.initialization_data_im.saveData(self.data.file_prefix + 'initialization_data_im.h5')
        self.models.f_im.train(self.data.initialization_data_im)

        try:
            self.initialization_models.f_im = self.models.f_im.model.return_copy()
        except AttributeError:
            self.initialization_models.f_im = self.models.f_im.model

        print('SM Exploration (Simple), Line 2: G_IM was initialized')

        n_save_data = self.params.n_save_data;
        n_experiments = self.params.n_experiments
        eval_step = self.params.eval_step

        print('SM Exploration (Simple), Lines 4-22: : Main simulation running...')
        for i in range(n_experiments):
            self.learner.sensor_goal = self.models.f_im.get_goal(self.learner)
            self.models.f_sm.getMotorCommand(self.learner)
            self.learner.executeMotorCommand()
            self.get_competence(self.learner)
            self.data.simulation_data.appendData(self.learner)

            ''' Train Interest Model'''
            if (i + 1)%self.models.f_im.params.im_step==0:
                # print('Algorithm 1 (Proprioceptive), Line 4-22: Experiment: Training Model IM')
                if i < self.models.f_im.params.n_training_samples:
                    self.models.f_im.train(
                        self.data.initialization_data_im.mixDataSets(self.learner, self.data.simulation_data))
                else:
                    self.models.f_im.train(self.data.simulation_data)

            ''' Train Sensorimotor Model'''
            if (i + 1)%self.models.f_sm.params.sm_step==0:
                # print('Algorithm 1 (Proprioceptive), Line 4-22: Experiment: Training Model SM')
                if (i < n_init or self.params.sm_all_samples):  ###BE CAREFUL WITH MEMORY
                    self.models.f_sm.trainIncrementalLearning(
                        self.data.simulation_data.mixDataSets(self.learner,
                                                              self.data.initialization_data_im.mixDataSets(
                                                                  self.learner,self.data.initialization_data_sm_ss)))
                else:
                    self.models.f_sm.trainIncrementalLearning(self.data.simulation_data)
            if self.evaluate and (i + 1)%eval_step == 0:
                self.evaluation.model = self.models.f_sm
                eval_data = self.evaluation.evaluateModel()
                error_ = np.linalg.norm(eval_data.sensor_goal_data.data - eval_data.sensor_data.data, axis=1)
                self.evaluation_error = np.append(self.evaluation_error, np.mean(error_))


            # print('SM Exploration (Simple), Line 4-22: Experiment: {} of {}'.format(i + 1, n_experiments)) # Slow
            if (i + 1)%n_save_data == 0:
                self.data.simulation_data.saveData(self.data.file_prefix + 'simulation_data.h5')
                print('SM Exploration (Simple), Line 4-22: Experiment: Saving data at samples {} of {}'.format(i + 1,
                                                                                                         n_experiments))

        self.data.simulation_data.saveData('simulation_data.h5')
        saveSimulationData([self.data.file_prefix + 'initialization_data_sm.h5',
                            self.data.file_prefix + 'initialization_data_im.h5',
                            self.data.file_prefix + 'simulation_data.h5'], 'simulation_data.tar.gz')

        print('SM Exploration (Simple), Experiment was finished and data saved')


def get_eval_error(simulation):
    evaluation = SM_ModelEvaluation(simulation.system,
                                    10,
                                    simulation.models.f_sm)
