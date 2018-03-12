"""
Created on Mar 13, 2017

@author: Juan Manuel Acevedo Valle
"""
from numpy import linspace
from numpy import random as np_rnd
import datetime

now = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")

class OBJECT(object):
    def __init__(self):
        pass

if __name__ == '__main__':
    #  Adding the projects folder to the path##
    import os, sys, random

    # sys.path.append("../../")

    #  Adding libraries##
    from exploration.systems.Diva2017a import Diva2017a as System
    from exploration.systems.Diva2017a import Instructor
    from exploration.algorithm.algorithm2017 import Algorithm as Algorithm
    from exploration.algorithm.algorithm2017 import OBJECT
    from exploration.algorithm.evaluation import Evaluation
    from exploration.data.PlotTools import *

    from diva_configurations import model_, comp_func

    # models
    f_sm_key = 'igmm_sm'
    f_cons_key = 'explauto_cons'
    f_im_key = 'explauto_im'

    """
       'gmm_sm':  GMM_SM,
       'gmm_ss':  GMM_SS,
       'igmm_sm': IGMM_SM,
       'igmm_ss': IGMM_SS,
       'gmm_im':  GMM_IM,
       'explauto_im': ea_IM,
       'explauto_sm': ea_SM,
       'explauto_ss': ea_SS,
       'random':  RdnM
    """

    # To guarantee reproducible experiments
    random_seed = 3487  # 1234 3487 9751

    n_initialization = 1000
    n_experiments = 10000
    n_save_data = 1000   # np.nan to not save, -1 to save 5 times during exploration

    eval_step = 5000

    random.seed(random_seed)
    np_rnd.seed(random_seed)

    # Creating Agent ##
    system = System()
    instructor = Instructor(slope = 1.)

    # Creating models ##
    models = OBJECT()

    models.f_sm = model_(f_sm_key, system)
    models.f_cons = model_(f_cons_key, system)
    models.f_im = model_(f_im_key, system)

    # Creating Simulation object, running simulation and plotting experiments##
    # tree/DP Interest Model
    directory = 'test'
    file_prefix = directory + '/IEEE_SI_optGMM_gruppies_numpy_' + now

    evaluation_sim = Evaluation(system,
                                models.f_sm, comp_func=comp_func,
                                file_prefix=file_prefix)

    #evaluation_sim.load_eval_dataset('../../systems/datasets/german_dataset_3.h5')
    #Use only the vowels that are used by the instructor
    evaluation_sim.data = instructor.data
    evaluation_sim.n_samples_val = len(evaluation_sim.data.sensor.data)
    evaluation_sim.random_indexes_val = range(evaluation_sim.n_samples_val)
    proprio = True
    mode = 'social'

    simulation = Algorithm(system,
                           models,
                           n_experiments,
                           comp_func,
                           instructor = instructor,
                           n_initialization_experiments=n_initialization,
                           random_seed=random_seed,#1321,
                           g_im_initialization_method='all',
                           n_save_data=n_save_data,
                           evaluation=evaluation_sim,
                           eval_step=eval_step,
                           sm_all_samples=False,
                           file_prefix=file_prefix)

    simulation.f_sm_key = f_sm_key
    simulation.f_cons_key = f_cons_key
    simulation.f_im_key = f_im_key

    simulation.mode = mode

    simulation.run(proprio=proprio)

    sim_data = simulation.data

    evaluation_sim.model.set_sigma_explo_ratio(0.)
    val_data = evaluation_sim.evaluate(save_data=True)
    error_ = np.linalg.norm(val_data.sensor_goal.data.as_matrix() -
                            val_data.sensor.data.as_matrix(), axis=1)
    print("Mean evaluation error is {} (max: {}, min: {})".format(np.mean(error_),
                                                                  np.max(error_),
                                                                  np.min(error_)))
