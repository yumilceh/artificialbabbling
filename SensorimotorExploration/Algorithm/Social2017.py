"""
Created on Feb 20, 2017

@author: Juan Manuel Acevedo Valle
"""
# ===============================================================================
# from ..Algorithm.utils.CompetenceFunctions import get_competence_Moulin2013 as get_competence
# from Algorithm.utils.CompetenceFunctions import get_competence_Baraglia2015 as get_competence
# ===============================================================================
import numpy as np
import datetime, copy  # copy.deepcopy
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
        self.init_motor_commands = get_random_motor_set(alg.agent,
                                                        alg.params.n_initialization_experiments)
        self.initialization_data_sm_ss = SimulationData(alg.agent)
        self.initialization_data_im = SimulationData(alg.agent)
        self.simulation_data = SimulationData(alg.agent)


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
                 n_save_data=50000,
                 sm_all_samples=False,
                 evaluation=None,
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
        self.params.n_initialization_experiments = n_initialization_experiments
        self.params.n_experiments = n_experiments
        self.params.random_seed = random_seed
        self.params.g_im_initialization_method = g_im_initialization_method
        self.params.n_save_data = n_save_data
        self.params.sm_all_samples = sm_all_samples

        self.learner = learner
        self.initialization_models = OBJECT()
        self.models = models

        self.data = DATA(self)
        self.data.file_prefix = file_prefix

        self.evaluation = evaluation
        if not type(evaluation) == type(None):
            self.evaluation_error = [1.0]

    def run(self, proprio = True):
        if proprio:
            self.run_proprio()
        else:
            self.run_simple()

    def run_proprio(self):
        n_init = self.params.n_initialization_experiments
        motor_commands = self.data.init_motor_commands

        print('SM Exploration, Line 1: Initializing G_SM and G_SS')
        for i in range(n_init):
            self.agent.setMotorCommand(motor_commands[i, :])
            self.agent.executeMotorCommand()
            self.data.initialization_data_sm_ss.appendData(self.agent)
            # print('SM , Line 1: Initialize G_SM and G_SS, experiment: {} of {}'.format(i + 1,n_init)) #Slow
        self.models.f_sm.train(self.data.initialization_data_sm_ss)
        self.models.f_ss.train(self.data.initialization_data_sm_ss)
        self.initialization_models.f_sm = self.models.f_sm.model.returnCopy()
        self.initialization_models.f_ss = self.models.f_ss.model.returnCopy()

        if not self.evaluation == None:
            self.evaluation.model = self.models.f_sm
            eval_data = self.evaluation.evaluateModel()
            error_ = np.linalg.norm(eval_data.sensor_goal_data.data - eval_data.sensor_data.data, axis=1)
            self.evaluation_error = np.append(self.evaluation_error, np.mean(error_))

        print(
            'Algorithm 1 (Proprioceptive), Line 1: Initialize G_SM and G_SS, experiment {} of {}'.format(i + 1, n_init))

        self.data.initialization_data_sm_ss.saveData(self.data.file_prefix + 'initialization_data_sm_ss.h5')

        g_im_initialization_method = self.params.g_im_initialization_method
        if (g_im_initialization_method == 'non-zero'):
            sensor_goals = self.data.initialization_data_sm_ss.sensor_data.data.as_matrix()
            for i in range(n_init):
                print(
                    'Algorithm 1 (Proprioceptive), Line 2: Initialize G_IM, experiment: {} of {}'.format(i + 1, n_init))
                if (linalg.norm(sensor_goals[i]) > 0):
                    self.agent.sensor_goal = sensor_goals[i]
                    self.models.f_sm.getMotorCommand(self.agent)
                    self.agent.executeMotorCommand()
                    get_competence(self.agent)
                    self.data.initialization_data_im.appendData(self.agent)
                    print('Algorithm 1 (Proprioceptive), Line 2: Initialize G_IM, Non-null sensory result considered ')
        elif (g_im_initialization_method == 'non-painful'):
            sensor_goals = self.data.initialization_data_sm_ss.sensor_data.data.as_matrix()
            proprio_data = self.data.initialization_data_sm_ss.somato_data.data.as_matrix()
            for i in range(n_init):
                print(
                    'Algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM, experiment: {} of {}'.format(i, n_init))
                if ((proprio_data[i]) == 0):
                    self.agent.sensor_goal = sensor_goals[i]
                    self.models.f_sm.getMotorCommand(self.agent)
                    self.agent.executeMotorCommand()
                    get_competence(self.agent)
                    self.data.initialization_data_im.appendData(self.agent)
                    print(
                        'Algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM, Non-painful sensory result considered ')

        elif (g_im_initialization_method == 'all'):
            sensor_goals = self.data.initialization_data_sm_ss.sensor_data.data.as_matrix()
            for i in range(n_init):
                print(
                    'Algorithm 1 (Proprioceptive), Line 2: Initialize G_IM, experiment: {} of {}'.format(i + 1, n_init))
                self.agent.sensor_goal = sensor_goals[i]
                self.models.f_sm.getMotorCommand(self.agent)
                self.agent.executeMotorCommand()
                get_competence(self.agent)
                self.data.initialization_data_im.appendData(self.agent)
                print('Algorithm 1 (Proprioceptive), Line 2: Initialize G_IM, All sensory result considered ')
        self.data.initialization_data_im.saveData(self.data.file_prefix + 'initialization_data_im.h5')
        self.models.f_im.train(self.data.initialization_data_im)
        self.initialization_models.f_im = self.models.f_im.model.returnCopy()

        n_save_data = self.params.n_save_data;
        n_experiments = self.params.n_experiments
        for i in range(n_experiments):
            self.agent.sensor_goal = self.models.f_im.get_interesting_goal_proprioception(self.agent, self.models.f_sm,
                                                                                          self.models.f_ss)
            self.models.f_im.activateAllComponents()
            self.models.f_sm.getMotorCommand(self.agent)
            self.agent.executeMotorCommand()
            get_competence(self.agent)
            self.data.simulation_data.appendData(self.agent)

            ''' Train Interest Model'''
            if ((i + 1) % self.models.f_im.params.im_step) == 0:
                print('Algorithm 1 (Proprioceptive), Line 4-22: Experiment: Training Model IM')
                if i < self.models.f_im.params.n_training_samples:
                    self.models.f_im.train(
                        self.data.initialization_data_im.mixDataSets(self.agent, self.data.simulation_data))
                else:
                    self.models.f_im.train(self.data.simulation_data)

            ''' Train Sensorimotor Model'''
            if ((i + 1) % self.models.f_sm.params.sm_step) == 0:
                print('Algorithm 1 (Proprioceptive), Line 4-22: Experiment: Training Model SM')
                if (i < n_init or self.params.sm_all_samples):  ###BE CAREFUL WITH MEMORY
                    self.models.f_sm.trainIncrementalLearning(
                        self.data.simulation_data.mixDataSets(self.agent,
                                                              self.data.initialization_data_im.mixDataSets(self.agent,
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
                print('Algorithm 1 (Proprioceptive), Line 4-22: Experiment: Training Model SS')
                if (i < n_init or self.params.sm_all_samples):  ###BE CAREFUL WITH MEMORY
                    self.models.f_ss.trainIncrementalLearning(
                        self.data.simulation_data.mixDataSets(self.agent,
                                                              self.data.initialization_data_im.mixDataSets(self.agent,
                                                                                                           self.data.initialization_data_sm_ss)))
                else:
                    self.models.f_sm.trainIncrementalLearning(self.data.simulation_data)

            print('Algorithm 1 (Proprioceptive), Line 4-22: Experiment: {} of {}'.format(i + 1, n_experiments))
            if (np.mod(i, n_save_data) == 0):
                self.data.simulation_data.saveData(self.data.file_prefix + 'simulation_data.h5')

        self.data.simulation_data.saveData('simulation_data.h5')
        saveSimulationData([self.data.file_prefix + 'initialization_data_sm_ss.h5',
                            self.data.file_prefix + 'initialization_data_im.h5',
                            self.data.file_prefix + 'simulation_data.h5'], 'simulation_data.tar.gz')


    def run_simple(self):
        n_init = self.params.n_initialization_experiments
        motor_commands = self.data.init_motor_commands
        for i in range(n_init):
            self.agent.setMotorCommand(motor_commands[i, :])
            self.agent.executeMotorCommand()
            self.data.initialization_data_sm_ss.appendData(self.agent)
            print('Algorithm 1 (Non-proprioceptive), Line 1: Initialize G_SM and G_SS, experiment: {} of {}'.format(i,
                                                                                                                    n_init))
        self.models.f_sm.train(self.data.initialization_data_sm_ss)
        self.models.f_ss.train(self.data.initialization_data_sm_ss)
        self.initialization_models.f_sm = self.models.f_sm.model.returnCopy()
        self.initialization_models.f_ss = self.models.f_ss.model.returnCopy()

        if not self.evaluation == None:
            self.evaluation.model = self.models.f_sm
            eval_data = self.evaluation.evaluateModel()
            error_ = np.linalg.norm(eval_data.sensor_goal_data.data - eval_data.sensor_data.data, axis=1)
            self.evaluation_error = np.append(self.evaluation_error, np.mean(error_))

        print(
        'Algorithm 1 (Non-proprioceptive), Line 1: Initialize G_SM and G_SS, experiment {} of {}'.format(i, n_init))

        self.data.initialization_data_sm_ss.saveData(self.data.file_prefix + 'initialization_data_sm_ss.h5')

        g_im_initialization_method = self.params.g_im_initialization_method
        if (g_im_initialization_method == 'non-zero'):
            sensor_goals = self.data.initialization_data_sm_ss.sensor_data.data.as_matrix()
            for i in range(n_init):
                print(
                'Algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM, experiment: {} of {}'.format(i, n_init))
                if (linalg.norm(sensor_goals[i]) > 0):
                    self.agent.sensor_goal = sensor_goals[i]
                    self.models.f_sm.getMotorCommand(self.agent)
                    self.agent.executeMotorCommand()
                    get_competence(self.agent)
                    self.data.initialization_data_im.appendData(self.agent)
                    print(
                    'Algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM, Non-null sensory result considered ')
        elif (g_im_initialization_method == 'non-painful'):
            sensor_goals = self.data.initialization_data_sm_ss.sensor_data.data.as_matrix()
            proprio_data = self.data.initialization_data_sm_ss.somato_data.data.as_matrix()
            for i in range(n_init):
                print(
                'Algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM, experiment: {} of {}'.format(i, n_init))
                if ((proprio_data[i]) == 0):
                    self.agent.sensor_goal = sensor_goals[i]
                    self.models.f_sm.getMotorCommand(self.agent)
                    self.agent.executeMotorCommand()
                    get_competence(self.agent)
                    self.data.initialization_data_im.appendData(self.agent)
                    print(
                    'Algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM, Non-painful sensory result considered ')
        elif (g_im_initialization_method == 'all'):
            sensor_goals = self.data.initialization_data_sm_ss.sensor_data.data.as_matrix()
            for i in range(n_init):
                print(
                'Algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM, experiment: {} of {}'.format(i, n_init))
                self.agent.sensor_goal = sensor_goals[i]
                self.models.f_sm.getMotorCommand(self.agent)
                self.agent.executeMotorCommand()
                get_competence(self.agent)
                self.data.initialization_data_im.appendData(self.agent)
                print('Algorithm 1 (Non-proprioceptive), Line 2: Initialize G_IM, All sensory result considered ')
        self.data.initialization_data_im.saveData(self.data.file_prefix + 'initialization_data_im.h5')
        self.models.f_im.train(self.data.initialization_data_im)

        # =======================================================================
        # self.models.f_im.model.interactiveModel(self.models.f_im.get_train_data(self.data.initialization_data_im))
        # =======================================================================

        self.initialization_models.f_im = self.models.f_im.model.returnCopy()

        n_save_data = self.params.n_save_data;
        n_experiments = self.params.n_experiments
        for i in range(n_experiments):
            self.agent.sensor_goal = self.models.f_im.get_interesting_goal(self.agent)
            self.models.f_sm.getMotorCommand(self.agent)
            self.agent.executeMotorCommand()
            get_competence(self.agent)
            self.data.simulation_data.appendData(self.agent)

            ''' Train Interest Model'''
            if ((i + 1) % self.models.f_im.params.im_step) == 0:
                print('Algorithm 1 (Non-proprioceptive), Line 4-22: Experiment: Training Model IM')
                if i < self.models.f_im.params.n_training_samples:
                    self.models.f_im.train(
                        self.data.initialization_data_im.mixDataSets(self.agent, self.data.simulation_data))
                else:
                    self.models.f_im.train(self.data.simulation_data)

            ''' Train Sensorimotor Model'''
            if ((i + 1) % self.models.f_sm.params.sm_step) == 0:
                print('Algorithm 1 (Non-proprioceptive), Line 4-22: Experiment: Training Model SM')
                if (i < n_init or self.params.sm_all_samples):  ###BE CAREFUL WITH MEMORY
                    self.models.f_sm.trainIncrementalLearning(
                        self.data.simulation_data.mixDataSets(self.agent,
                                                              self.data.initialization_data_im.mixDataSets(self.agent,
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
                print('Algorithm 1 (Non-proprioceptive), Line 4-22: Experiment: Training Model SS')
                if (i < n_init or self.params.sm_all_samples):  ###BE CAREFUL WITH MEMORY
                    self.models.f_ss.trainIncrementalLearning(
                        self.data.simulation_data.mixDataSets(self.agent,
                                                              self.data.initialization_data_im.mixDataSets(self.agent,
                                                                                                           self.data.initialization_data_sm_ss)))
                else:
                    self.models.f_sm.trainIncrementalLearning(self.data.simulation_data)

            print('Algorithm 1 (Non-proprioceptive), Line 4-22: Experiment: {} of {}'.format(i + 1, n_experiments))
            if (np.mod(i, n_save_data) == 0):
                self.data.simulation_data.saveData(self.data.file_prefix + 'simulation_data.h5')

        self.data.simulation_data.saveData('simulation_data.h5')
        saveSimulationData([self.data.file_prefix + 'initialization_data_sm_ss.h5',
                            self.data.file_prefix + 'initialization_data_im.h5',
                            self.data.file_prefix + 'simulation_data.h5'], 'simulation_data.tar.gz')

def returnEvaluationError(simulation):
    evaluation = SM_ModelEvaluation(simulation.system,
                                    10,
                                    simulation.models.f_sm)
