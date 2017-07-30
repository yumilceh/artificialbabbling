"""
Created on May 13, 2017

@author: Juan Manuel Acevedo Valle
"""
import numpy as np
import random
from numpy import linalg
import datetime
from ..DataManager.SimulationData import SimulationData_v2 as SimulationData
from ..Algorithm.utils.functions import get_random_motor_set
from ..Algorithm.utils.logging import write_config_log

now = datetime.datetime.now().strftime("Alg2017a_%Y_%m_%d_%H_%M_")

class OBJECT(object):
    def __init__(self):
        pass

class Algorithm(object):
    """
    Approach to social learning
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

        random.seed(random_seed)
        np.random.seed(random_seed)

        self.name = 'InteractionAlgorithm2017a'
        self.type = 'simple'
        self.params = OBJECT()
        self.params.sm_all_samples = None
        self.params.n_initialization_experiments = n_initialization_experiments
        self.params.n_experiments = n_experiments
        self.params.random_seed = random_seed
        self.params.g_im_initialization_method = g_im_initialization_method

        self.params.expl_space = 'sensor' # 'sensor' for salient and 'somato' for haptic

        if n_save_data == -1:
            n_save_data = np.floor(n_experiments / 5)

        self.params.n_save_data = n_save_data

        self.params.sm_all_samples = sm_all_samples

        self.learner = learner
        self.initialization_models = OBJECT()
        self.models = models
        self.get_competence = competence_func

        try:
            self.init_motor_commands = get_random_motor_set(learner,
                                                            n_initialization_experiments,
                                                            min_values = learner.self.min_motor_values_init,
                                                            max_values = learner.self.max_motor_values_init)
        except AttributeError:
            self.init_motor_commands = get_random_motor_set(learner,
                                                            n_initialization_experiments)
        self.mode = 'autonomous'
        if instructor is not None:
            self.instructor = instructor
            # self.imitation = []
            self.mode = 'social'

        self.data = SimulationData(learner, prelocated_samples=n_experiments+2*n_initialization_experiments+1)
        self.data.file_prefix = file_prefix

        self.evaluation = evaluation
        self.evaluate = False
        if evaluation is not None:
            self.evaluation_error = []
            self.evaluate = True
            if eval_step == -1:
                eval_step = np.floor(n_experiments / 5)
        self.params.eval_step = eval_step

    def run(self, proprio=True):
        if proprio:
            self.type = 'proprio'
        if self.params.n_save_data is not np.nan:
            write_config_log(self, self.data.file_prefix + 'conf.txt')
        self.run_()

    def run_(self):
        if self.params.g_im_initialization_method is 'non-painful': # Check this
            print('G_IM, init: non-painful method not defined for non-proprioceptive agents.')
            print('G_IM, init: Switching to all samples method')
            self.params.g_im_initialization_method = 'all'
        n_init = self.params.n_initialization_experiments
        motor_commands = self.init_motor_commands


        n_save_data = self.params.n_save_data

        print('SM Exploration ({}, {}), Line 1: Initializing G_SM'.format(self.type, self.mode))
        for i in range(n_init):
            self.learner.set_action(motor_commands[i, :])
            self.learner.execute_action()
            self.data.appendData(self.learner)
            self.do_training(i, up_=['sm','ss'])

        self.do_training(i, up_=['sm','ss','cons'], force=True, all=True)

        print('G_SM initialized')

        if n_save_data is not np.nan:
            self.data.saveData(self.data.file_prefix + 'sim_data.h5')

        f_im_init = self.params.g_im_initialization_method

        sensor_goals = self.get_im_init_data(f_im_init)

        for i in range(sensor_goals.shape[0]):
            setattr(self.learner, self.params.expl_space+'_goal' ,sensor_goals[i, :])
            self.select_expl_model().get_action(self.learner)
            self.learner.execute_action()


            self.get_competence(self.learner, sensor_space=self.params.expl_space)
            self.data.appendData(self.learner)
            self.do_training(i, up_=['im', 'cons','sm','ss'])

        print('G_IM initialized')
        if n_save_data is not np.nan:
            self.data.saveData(self.data.file_prefix + 'sim_data.h5')
        self.do_training(i, up_=['sm','ss', 'cons', 'im'], force=True, all=True)

        print('SM Exploration ({}, {}), First evaluation of G_SM'.format(self.type, self.mode))

        self.do_evaluation(-1,space=self.params.expl_space)

        n_experiments = self.params.n_experiments
        eval_step = self.params.eval_step

        print('SM Exploration ({}, {}), Lines 4-22: : Main simulation running ({} samples)...'.
              format(self.type, self.mode, n_experiments))

        i = 0
        while i < n_experiments:
            self.learner.sensor_instructor.fill(np.nan)
            if self.type is 'proprio':
                setattr(self.learner, self.params.expl_space+'_goal',
                            self.models.f_im.get_goal_proprio(self.learner,
                                                             self.select_expl_model(),
                                                             self.models.f_cons))
                self.select_expl_model().get_action(self.learner)
                self.learner.execute_action()
                self.get_competence(self.learner, sensor_space=self.params.expl_space)
                if self.learner.cons_out > self.learner.cons_threshold:
                    self.learner.competence_result = 0.7 * self.learner.competence_result
            else:
                setattr(self.learner, self.params.expl_space+'_goal',
                            self.models.f_im.get_goal())
                self.select_expl_model().get_action(self.learner)
                self.learner.execute_action()
                self.get_competence(self.learner, sensor_space=self.params.expl_space)

            self.data.appendData(self.learner)
            i += 1
            self.do_training(i, up_=['sm','ss', 'cons', 'im'])
            self.do_evaluation(i,space=self.params.expl_space)

            if self.instructor is not None:
                reinforce, self.learner.sensor_instructor = self.instructor.interaction(self.learner.sensor_out)
                if reinforce is 1:
                    #self.imitation += [i, self.instructor.min_idx]
                    if self.mode is 'social':
                        if self.type is 'proprio':
                            tmp_goal = self.learner.sensor_instructor
                            tmp_motor = self.select_expl_model().get_action(self.learner, sensor_goal=tmp_goal)
                            tmp_cons = self.models.f_cons.predict_cons(self.learner, motor_command=tmp_motor)
                            if tmp_cons < self.learner.cons_threshold:
                                self.learner.sensor_goal = tmp_goal
                                self.learner.set_action(tmp_motor)
                                self.learner.execute_action()
                                self.get_competence(self.learner, sensor_space=self.params.expl_space)
                                if self.learner.cons_out > self.learner.cons_threshold:
                                    self.learner.competence_result = 0.7 * self.learner.competence_result

                                self.data.appendData(self.learner)
                                i += 1
                                # self.learner.sensor_instructor.fill(np.nan)
                                self.do_training(i, up_=['sm','ss', 'cons', 'im'])
                                self.do_evaluation(i,space=self.params.expl_space)

                        else:
                            self.learner.sensor_goal = self.learner.sensor_instructor
                            self.select_expl_model().get_action(self.learner)
                            #self.models.self.select_expl_model().get_action(self.learner)
                            self.learner.execute_action()
                            self.get_competence(self.learner, sensor_space=self.params.expl_space)
                            self.data.appendData(self.learner)
                            i += 1#
                            # self.learner.sensor_instructor.fill(np.nan)
                            self.do_training(i, up_=['sm','ss', 'cons', 'im'])
                            self.do_evaluation(i,space=self.params.expl_space)

            # print('SM Exploration (Simple), Line 4-22: Experiment: {} of {}'.format(i + 1, n_experiments)) # Slow
            if (i + 1) % n_save_data == 0:
                print('SM Exploration ({}, {}), Line 4-22: Experiment: Saving data at samples {} of {}'. \
                      format(self.type, self.mode, i + 1, n_experiments))
                self.data.saveData(self.data.file_prefix + 'sim_data.h5')
                # ndarray_to_h5(self.imitation, 'imitation', self.data.file_prefix + 'social_data.h5')

        self.do_training(i, up_=['sm','ss', 'cons', 'im'], force=True)

        self.select_expl_model().set_sigma_explo_ratio(0.)
        self.do_evaluation(-1,space=self.params.expl_space)
        self.data.cut_final_data()

        if n_save_data is not np.nan:
            print('SM Exploration ({}, {}), Saving data...'.format(self.type, self.mode))
            self.data.saveData(self.data.file_prefix + 'sim_data.h5')
            # ndarray_to_h5(self.imitation, 'imitation', self.data.file_prefix + 'social_data.h5')
        print('SM Exploration ({}, {}), Experiment was finished.'.format(self.type, self.mode))


    def get_im_init_data(self, method='all'):  # Non-painful is missing
        if method == 'non-zero':
            print('SM Exploration: IM initialization: Non-null sensory result considered')
            sensor_data = getattr(self.data,self.params.expl_space)
            sensor_goals = sensor_data.get_all().as_matrix()
            return sensor_goals[np.where(linalg.norm(sensor_goals, axis=1) > 0)[0], :]

        elif method == 'all':
            print('SM Exploration: IM initialization: All sensory result considered')
            sensor_data = getattr(self.data, self.params.expl_space)
            sensor_goals = sensor_data.get_all().as_matrix()
            return sensor_goals

        elif method == 'non-painful':
            print('SM Exploration: IM initialization: Non-painful result considered')
            cons_results = self.data.cons.get_all().as_matrix()
            sensor_data = getattr(self.data,self.params.expl_space)
            sensor_goals = sensor_data.get_all().as_matrix()
            return sensor_goals[np.where(cons_results == 0.)[0], :]

    def do_training(self, i, up_=['sm','ss', 'cons', 'im'], force=False, all=False):


        """ Train Interest Model"""
        if 'im' in up_ and ((i + 1) % self.models.f_im.params.im_step == 0 or force):
            # print('Algorithm 1 (Proprioceptive), Line 4-22: Experiment: Training Model IM')
            self.models.f_im.train(self.data)

        """Train Sensorimotor Model"""
        if 'sm' in up_ and ((i + 1) % self.models.f_sm.params.sm_step == 0 or force):
            # if i > 3200:
            #     pass
            # print('Algorithm 1 (Proprioceptive), Line 4-22: Experiment: Training Model SM')
            self.models.f_sm.train_incremental(self.data, all = all)
            # print('i: {}, sum w: {}'.format(i,sum(self.models.f_sm.model.weights_)))

        try:
            if 'ss' in up_ and ((i + 1) % self.models.f_ss.params.sm_step == 0 or force):
                # print('Algorithm 1 (Proprioceptive), Line 4-22: Experiment: Training Model SM')
                self.models.f_ss.train_incremental(self.data, all = all)
        except:
            pass
        # if 'ss' in up_ and ((i + 1) % self.models.f_cons.params.sm_step == 0 or force):
        #     # print('Algorithm 1 (Proprioceptive), Line 4-22: Experiment: Training Model SM')
        #     self.models.f_sm.train_incremental(self.data, all = all)

        if hasattr(self.models, 'f_cons'):
            if 'cons' in up_ and ((i + 1) % self.models.f_cons.params.cons_step == 0 or force):
                self.models.f_cons.train_incremental(self.data)

    def do_evaluation(self, i, space=None, force=False, save_data=False):
        if self.evaluate and (i + 1) % self.params.eval_step == 0 or force:
            if space is None:
                space = self.params.expl_space
                self.evaluation.model = self.select_expl_model()
            else:
                self.evaluation.model = self.select_expl_model(space=space)

            tmp_sigma = self.evaluation.model.get_sigma_explo()
            self.evaluation.model.set_sigma_explo_ratio(0)

            eval_data = self.evaluation.evaluate(saveData=save_data, space=space)

            sensor_goal_data = getattr(eval_data, self.params.expl_space+'_goal').get_all().as_matrix()
            sensor_data = getattr(eval_data, self.params.expl_space).get_all().as_matrix()

            error_ = np.linalg.norm(sensor_goal_data - sensor_data, axis=1)

            self.evaluation_error += [[i, np.mean(error_)]]

            if self.params.n_save_data is not np.nan and force is False:
                with open(self.data.file_prefix + 'eval_error.txt', "a") as log_file:
                    log_file.write('{}: {}\n'.format(i, np.mean(error_)))

            print('Evaluation finished.')
            self.evaluation.model.set_sigma_explo(tmp_sigma)
        #  print(self.evaluation.model.get_sigma_explo())

    def set_expl_space(self, space = 'sensor'):
        self.params.expl_space = space  # 'sensor' for salient and 'somato' for haptic
        print('Exploration in {} space'.format(space))
        if space is 'somato':
            self.instructor = None

    def select_expl_model(self, space=None):
        if space is None:
            space = self.params.expl_space
        if space == 'somato':
            return self.models.f_ss
        elif space == 'sensor':
            return self.models.f_sm
        else:
            NotImplementedError()
