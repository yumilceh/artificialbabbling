"""
Created on Feb 20, 2017

@author: Juan Manuel Acevedo Valle
"""
# ===============================================================================
# from ..Algorithm.utils.CompetenceFunctions import get_competence_Moulin2013 as get_competence
# from Algorithm.utils.CompetenceFunctions import comp_Baraglia2015 as get_competence
# ===============================================================================
import numpy as np
from numpy import linalg
import datetime  # copy.deepcopy
from ..DataManager.SimulationData import SimulationData
from ..Algorithm.utils.functions import get_random_motor_set
from ..Algorithm.ModelEvaluation import SM_ModelEvaluation
from ..Algorithm.utils.logging import write_config_log
from ..Algorithm.utils.data_storage_funcs import ndarray_to_h5
# import logging
# logging.basicConfig(filename='cylinder.log', level=logging.DEBUG,\
#         format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
# logging.info("Volume of a Cylinder with radius {} and height {} is {}".format(args.radius, args.height, volume))

now = datetime.datetime.now().strftime("Alg2017a_%Y_%m_%d_%H_%M_")


class OBJECT(object):
    def __init__(self):
        pass


class InteractionAlgorithm(object):
    """
    Approach to social learning of vowel like sounds
    """

    def __init__(self, learner,
                 models,
                 n_experiments,
                 competence_func,
                 instructor = None,
                 n_initialization_experiments=100,
                 random_seed=np.random.random((1, 1)),
                 g_im_initialization_method='non-zero',  # 'non-zero' 'all' 'non-painful'
                 n_save_data=np.nan,
                 sm_all_samples=False,
                 evaluation=None,
                 eval_step=-1,
                 file_prefix=now):

        self.name = 'InteractionAlgorithm2017a'
        self.type = 'Simple'
        self.params = OBJECT()
        self.params.sm_all_samples = None
        self.params.n_initialization_experiments = n_initialization_experiments
        self.params.n_experiments = n_experiments
        self.params.random_seed = random_seed
        self.params.g_im_initialization_method = g_im_initialization_method

        if n_save_data == -1:
            n_save_data = np.floor(n_experiments / 5)

        self.params.n_save_data = n_save_data

        self.params.sm_all_samples = sm_all_samples

        self.learner = learner
        self.initialization_models = OBJECT()
        self.models = models
        self.get_competence = competence_func

        self.init_motor_commands = get_random_motor_set(learner,
                                                        n_initialization_experiments)
        self.mode = 'autonomous'
        if instructor is not None:
            self.instructor = instructor
            self.imitation = []
            self.mode = 'social'

        self.data = SimulationData(learner)
        self.data.file_prefix = file_prefix

        self.evaluation = evaluation
        self.evaluate = False
        if evaluation is not None:
            self.evaluation_error = [np.infty]
            self.evaluate = True
            if eval_step == -1:
                eval_step = np.floor(n_experiments / 5)
        self.params.eval_step = eval_step

    def run(self, proprio=True):
        if proprio:
            self.type = 'Proprio'
        if self.params.n_save_data is not np.nan:
            write_config_log(self, self.data.file_prefix + 'conf.txt')
        self.run_()

    def run_(self):
        if self.params.g_im_initialization_method is 'non-painful':
            print('G_IM, init: non-painful method not defined for non-proprioceptive agents.')
            print('G_IM, init: Switching to all samples method')
            self.params.g_im_initialization_method = 'all'
        n_init = self.params.n_initialization_experiments
        motor_commands = self.init_motor_commands


        n_save_data = self.params.n_save_data

        print('SM Exploration ({}, {}), Line 1: Initializing G_SM'.format(self.type, self.mode))
        for i in range(n_init):
            self.learner.set_action(motor_commands[i, :])
            self.learner.executeMotorCommand()
            self.data.appendData(self.learner)
            self.do_training(i, up_=['sm'])

        self.do_training(i, up_=['sm','ss'], force=True)

        print('G_SM initialized')

        if n_save_data is not np.nan:
            self.data.saveData(self.data.file_prefix + 'sim_data.h5')

        f_im_init = self.params.g_im_initialization_method

        sensor_goals = self.get_im_init_data(f_im_init)

        for i in range(sensor_goals.shape[0]):
            self.learner.sensor_goal = sensor_goals[i, :]
            self.models.f_sm.get_action(self.learner)
            self.learner.executeMotorCommand()
            self.get_competence(self.learner)
            self.data.appendData(self.learner)
            self.do_training(i, up_=['im', 'ss','sm'])

        print('G_IM initialized')
        if n_save_data is not np.nan:
            self.data.saveData(self.data.file_prefix + 'sim_data.h5')
        self.do_training(i, up_=['sm', 'ss', 'im'], force=True)

        print('SM Exploration ({}, {}), First evaluation of G_SM'.format(self.type, self.mode))
        self.do_evaluation(-1)

        n_experiments = self.params.n_experiments
        eval_step = self.params.eval_step

        print('SM Exploration ({}, {}), Lines 4-22: : Main simulation running ({} samples)...'.
              format(self.type, self.mode, n_experiments))

        i = 0
        while i < n_experiments:
            if self.type is 'Proprio':
                self.learner.sensor_goal = self.models.f_im.get_goal_proprio(self.learner,
                                                                         self.models.f_sm,
                                                                         self.models.f_ss)
            else:
                self.learner.sensor_goal = self.models.f_im.get_goal(self.learner)

            self.models.f_sm.get_action(self.learner)
            self.learner.executeMotorCommand()
            self.get_competence(self.learner)
            self.data.appendData(self.learner)
            i += 1

            if self.instructor is not None:
                reinforce = self.instructor.interaction(self.learner.sensor_out)
                if reinforce:
                    self.imitation += [i, self.instructor.min_idx]
                    if self.mode is 'social':
                        self.learner.sensor_goal = self.instructor.sensor_out
                        self.models.f_sm.get_action(self.learner)
                        self.learner.executeMotorCommand()
                        self.get_competence(self.learner)
                        self.data.appendData(self.learner)
                        i += 1

            self.do_training(i, up_=['sm', 'ss', 'im'])
            self.do_evaluation(i)

            # print('SM Exploration (Simple), Line 4-22: Experiment: {} of {}'.format(i + 1, n_experiments)) # Slow
            if (i + 1) % n_save_data == 0:
                print('SM Exploration ({}, {}), Line 4-22: Experiment: Saving data at samples {} of {}'. \
                      format(self.type, self.mode, i + 1, n_experiments))
                self.data.saveData(self.data.file_prefix + 'sim_data.h5')
                ndarray_to_h5(self.imitation, 'imitation', self.data.file_prefix + 'social_data.h5')

        self.do_training(i, up_=['sm', 'ss', 'im'], force=True)

        self.models.f_sm.set_sigma_explo_ratio(0.)
        self.do_evaluation(-1)

        if n_save_data is not np.nan:
            print('SM Exploration ({}, {}), Saving data...'.format(self.type, self.mode))
            self.data.saveData(self.data.file_prefix + 'sim_data.h5')
            ndarray_to_h5(self.imitation, 'imitation', self.data.file_prefix + 'social_data.h5')
        print('SM Exploration ({}, {}), Experiment was finished.'.format(self.type, self.mode))


    def get_im_init_data(self, method='all'):  # Non-painful is missing
        if method == 'non-zero':
            print('SM Exploration: IM initialization: Non-null sensory result considered')
            sensor_goals = self.data.sensor_data.data.as_matrix()
            return sensor_goals[np.where(linalg.norm(sensor_goals, axis=1) > 0), :]

        elif method == 'all':
            print('SM Exploration: IM initialization: All sensory result considered')
            return self.data.sensor_data.data.as_matrix()

    def do_training(self, i, up_=['sm', 'ss', 'im'], force=False):

        """ Train Interest Model"""
        if 'im' in up_ and ((i + 1) % self.models.f_im.params.im_step == 0 or force):
            # print('Algorithm 1 (Proprioceptive), Line 4-22: Experiment: Training Model IM')
            self.models.f_im.train(self.data)

        """Train Sensorimotor Model"""
        if 'sm' in up_ and ((i + 1) % self.models.f_sm.params.sm_step == 0 or force):
            # print('Algorithm 1 (Proprioceptive), Line 4-22: Experiment: Training Model SM')
            self.models.f_sm.trainIncrementalLearning(self.data)

        if hasattr(self.models, 'f_ss'):
            if 'ss' in up_ and ((i + 1) % self.models.f_ss.params.ss_step == 0 or force):
                self.models.f_ss.trainIncrementalLearning(self.data)

    def do_evaluation(self, i):
        tmp_sigma = self.evaluation.model.get_sigma_explo()
        self.evaluation.model.set_sigma_explo_ratio(0)
        if self.evaluate and (i + 1) % self.params.eval_step == 0:
            self.evaluation.model = self.models.f_sm
            eval_data = self.evaluation.evaluateModel()
            error_ = np.linalg.norm(eval_data.sensor_goal_data.data.as_matrix() -
                                    eval_data.sensor_data.data.as_matrix(), axis=1)
            self.evaluation_error = np.append(self.evaluation_error, np.mean(error_))
            print('Evaluation finished.')
        self.evaluation.model.set_sigma_explo(tmp_sigma)
        #  print(self.evaluation.model.get_sigma_explo())

def get_eval_error(simulation):
    evaluation = SM_ModelEvaluation(simulation.system,
                                    10,
                                    simulation.models.f_sm)
