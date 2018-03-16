"""
Created on Sep 12, 2017
Modified on Nov 15, 2017

@author: Juan Manuel Acevedo Valle
"""
import numpy as np
import random, os
from numpy import linalg
import datetime
from ..data.data import SimulationData as SimulationData
from ..algorithm.utils.functions import get_random_motor_set
from ..algorithm.utils.logging import write_config_log

now = datetime.datetime.now().strftime("alg2017a_%Y_%m_%d_%H_%M_")
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
                 instructor=None,
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

        self.name = 'Algorithm2017'
        self.type = 'simple'
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

        try: #Check id the system has option for  initialization bounds
            self.init_motor_commands = get_random_motor_set(learner,
                                                            n_initialization_experiments,
                                                            min_values=learner.self.min_motor_values_init,
                                                            max_values=learner.self.max_motor_values_init)
        except AttributeError:
            self.init_motor_commands = get_random_motor_set(learner,
                                                            n_initialization_experiments)
        self.mode = 'autonomous'
        if instructor is not None:
            self.instructor = instructor
            # self.imitation = []
            self.mode = 'social'
        else:
            #pass
            self.instructor = None


        # Analyze whether is necessary to  sum 1 and 2
        self.data = SimulationData(learner, prelocated_samples=n_experiments + 2 * n_initialization_experiments + 1)
        file_prefix = file_prefix.replace('/', os.sep)
        self.data.file_prefix = file_prefix

        self.evaluation = evaluation
        self.evaluate = False
        if evaluation is not None:
            self.evaluation_error = {}
            for key in evaluation.data.keys():
                self.evaluation_error.update({key: []})
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
        # random.seed(self.random_seed)    #Added October 2017 and not necessary
        # np.random.seed(self.random_seed) #Added October 2017 and not necessary
        if self.params.g_im_initialization_method is 'non-painful':
            print('G_IM, init: non-painful method not defined for non-proprioceptive agents.')
            print('G_IM, init: Switching to all samples method')
            #self.params.g_im_initialization_method = 'all' #Commented on 16th March 2018
        n_init = self.params.n_initialization_experiments
        motor_commands = self.init_motor_commands

        n_save_data = self.params.n_save_data

        print('SM Exploration ({}, {}), Line 1: Initializing G_SM'.format(self.type, self.mode))
        for i in range(n_init):
            self.learner.set_action(motor_commands[i, :])
            self.learner.execute_action()
            self.data.append_data(self.learner)
            self.do_training(i, up_=['sm'])

        self.do_training(i, up_=['sm', 'cons'], force=True, all=True)

        print('G_SM initialized')

        if n_save_data is not np.nan:
            self.data.save_data(self.data.file_prefix + 'sim_data.h5')

        f_im_init = self.params.g_im_initialization_method

        sensor_goals = self.get_im_init_data(f_im_init)

        for i in range(sensor_goals.shape[0]):
            self.learner.sensor_goal = sensor_goals[i, :]
            self.models.f_sm.get_action(self.learner)
            self.learner.execute_action()
            self.get_competence(self.learner)
            self.data.append_data(self.learner)
            self.do_training(i, up_=['im', 'cons', 'sm'])

        print('G_IM initialized')
        if n_save_data is not np.nan:
            self.data.save_data(self.data.file_prefix + 'sim_data.h5')
        self.do_training(i, up_=['sm', 'cons', 'im'], force=True, all=True)

        print('SM Exploration ({}, {}), First evaluation of G_SM'.format(self.type, self.mode))
        self.do_evaluation(-1)

        n_experiments = self.params.n_experiments
        eval_step = self.params.eval_step

        print('SM Exploration ({}, {}), Lines 4-22: : Main simulation running ({} samples)...'.
              format(self.type, self.mode, n_experiments))

        i = 0
        while i < n_experiments:
            # print(i)
            self.learner.sensor_instructor.fill(np.nan)
            if self.type == 'proprio':
                self.learner.sensor_goal = self.models.f_im.get_goal_proprio(self.learner,
                                                                             self.models.f_sm,
                                                                             self.models.f_cons)
                self.models.f_sm.get_action(self.learner)
                self.learner.execute_action()
                self.get_competence(self.learner)
                if self.learner.cons_out > self.learner.cons_threshold:
                    self.learner.competence_result = 0.7 * self.learner.competence_result
            else:
                self.learner.sensor_goal = self.models.f_im.get_goal()
                self.models.f_sm.get_action(self.learner)
                self.learner.execute_action()
                self.get_competence(self.learner)

            self.data.append_data(self.learner)
            i += 1
            self.do_training(i, up_=['sm', 'cons', 'im'])
            self.do_evaluation(i)

            reinforce = 0 #ADDED on 15/03/2018
            if self.instructor is not None:
                reinforce, self.learner.sensor_instructor =\
                    self.instructor.interaction(self.learner.sensor_out.copy())
                if reinforce == 1:
                    # self.imitation += [i, self.instructor.min_idx]
                    if self.mode == 'social':
                        if self.type == 'proprio':
                            tmp_goal = self.learner.sensor_instructor
                            tmp_motor = self.models.f_sm.get_action(self.learner, sensor_goal=tmp_goal)
                            tmp_cons = self.models.f_cons.predict_cons(self.learner, motor_command=tmp_motor)
                            if tmp_cons < self.learner.cons_threshold:
                                self.learner.sensor_goal = tmp_goal
                                self.learner.set_action(tmp_motor)
                                self.learner.execute_action()
                                self.get_competence(self.learner)
                                if self.learner.cons_out > self.learner.cons_threshold:
                                    self.learner.competence_result = 0.7 * self.learner.competence_result

                                self.data.append_data(self.learner)
                                i += 1
                                # self.learner.sensor_instructor.fill(np.nan)
                                self.do_training(i, up_=['sm', 'cons', 'im'])
                                self.do_evaluation(i)

                        else:
                            self.learner.sensor_goal = self.learner.sensor_instructor
                            self.models.f_sm.get_action(self.learner)
                            self.learner.execute_action()
                            self.get_competence(self.learner)
                            self.data.append_data(self.learner)
                            i += 1  #
                            # self.learner.sensor_instructor.fill(np.nan)
                            self.do_training(i, up_=['sm', 'cons', 'im'])
                            self.do_evaluation(i)

            # print('SM Exploration (Simple), Line 4-22: Experiment: {} of {}'.format(i + 1, n_experiments)) # Slow
            if (i + 1) % n_save_data == 0 or (reinforce and i % n_save_data == 0):
                print('SM Exploration ({}, {}), Line 4-22: Experiment: Saving data at samples {} of {}'. \
                      format(self.type, self.mode, i + 1, n_experiments))
                self.data.save_data(self.data.file_prefix + 'sim_data.h5')
                # ndarray_to_h5(self.imitation, 'imitation', self.data.file_prefix + 'social_data.h5')

        self.do_training(i, up_=['sm', 'cons', 'im'], force=True)

        self.models.f_sm.set_sigma_expl_ratio(0.)
        self.do_evaluation(-1)
        self.data.cut_final_data()

        if n_save_data is not np.nan:
            print('SM Exploration ({}, {}), Saving data...'.format(self.type, self.mode))
            self.data.save_data(self.data.file_prefix + 'sim_data.h5')
            # ndarray_to_h5(self.imitation, 'imitation', self.data.file_prefix + 'social_data.h5')
        print('SM Exploration ({}, {}), Experiment was finished.'.format(self.type, self.mode))

    def get_im_init_data(self, method='all'):  # Non-painful is missing
        if method == 'non-zero':
            print('SM Exploration: IM initialization: Non-null sensory result considered')
            sensor_goals = self.data.sensor.get_all().as_matrix()
            return sensor_goals[np.where(linalg.norm(sensor_goals, axis=1) > 0)[0], :]

        elif method == 'all':
            print('SM Exploration: IM initialization: All sensory result considered')
            return self.data.sensor.get_all().as_matrix()

        elif method == 'non-painful':
            print('SM Exploration: IM initialization: Non-painful result considered')
            cons_results = self.data.cons.get_all().as_matrix()
            sensor_goals = self.data.sensor.get_all().as_matrix()
            return sensor_goals[np.where(cons_results == 0.)[0], :]

    def do_training(self, i, up_=['sm', 'cons', 'im'], force=False, all=False):

        """ Train Interest Model"""
        if 'im' in up_ and ((i + 1) % self.models.f_im.params.im_step == 0 or force):
            # print('algorithm 1 (Proprioceptive), Line 4-22: Experiment: Training Model IM')
            self.models.f_im.train(self.data)

        """Train sensorimotor Model"""
        if 'sm' in up_ and ((i + 1) % self.models.f_sm.params.sm_step == 0 or force):
            # print('algorithm 1 (Proprioceptive), Line 4-22: Experiment: Training Model SM')
            self.models.f_sm.train_incremental(self.data, all=all)

        if hasattr(self.models, 'f_cons'):
            if 'cons' in up_ and ((i + 1) % self.models.f_cons.params.cons_step == 0 or force):
                self.models.f_cons.train_incremental(self.data)

    def do_evaluation(self, i, force=False, save_data=False):
        if self.evaluate and (i + 1) % self.params.eval_step == 0 or force:
            tmp_sigma = self.evaluation.model.get_sigma_expl()
            self.evaluation.model = self.models.f_sm
            self.evaluation.model.set_sigma_expl_ratio(0)
            eval_data = self.evaluation.evaluate(save_data=save_data)
            for key in eval_data.keys():
                error_ = np.linalg.norm(eval_data[key].sensor_goal.get_all().as_matrix() -
                                        eval_data[key].sensor.get_all().as_matrix(), axis=1)
                self.evaluation_error[key] += [i, np.mean(error_)]
                if self.params.n_save_data is not np.nan:
                    try:
                        with open(self.data.file_prefix + '_' + key + '_eval_error.txt', "a") as log_file:
                            log_file.write('{}: {}\n'.format(i, np.mean(error_)))
                    except:
                        with open(self.data.file_prefix + '_' + key + '_eval_error.txt', "w") as log_file:
                            log_file.write('{}: {}\n'.format(i, np.mean(error_)))
            print('Evaluations finished. Resuming exploration...')
            self.evaluation.model.set_sigma_expl(tmp_sigma)
            #  print(self.evaluation.model.get_sigma_expl())
