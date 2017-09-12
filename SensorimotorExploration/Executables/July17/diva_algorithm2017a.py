"""
Created on Mar 13, 2017

@author: Juan Manuel Acevedo Valle
"""
from numpy import linspace
from numpy import random as np_rnd
import datetime

now = datetime.datetime.now().strftime("Social_%Y_%m_%d_%H_%M_%S_")

class OBJECT(object):
    def __init__(self):
        pass

if __name__ == '__main__':
    #  Adding the projects folder to the path##
    import os, sys, random

    # sys.path.append("../../")

    #  Adding libraries##
    from SensorimotorExploration.Systems.Diva2017a import Diva2017a as System
    from SensorimotorExploration.Systems.Diva2017a import Instructor
    from SensorimotorExploration.Algorithm.algorithm_2017a import InteractionAlgorithm as Algorithm
    from SensorimotorExploration.Algorithm.algorithm_2017a import OBJECT
    from SensorimotorExploration.Algorithm.ModelEvaluation_v2 import SM_ModelEvaluation
    from SensorimotorExploration.DataManager.PlotTools import *

    from diva_configurations import model_, comp_func

    # Models
    f_sm_key = 'igmm_old'
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
    random_seed = 9751  # 1234 3487 9751

    n_initialization = 100
    n_experiments = 20000
    n_save_data = 5000   # np.nan to not save, -1 to save 5 times during exploration

    eval_step = 2000

    random.seed(random_seed)
    np_rnd.seed(random_seed)

    # Creating Agent ##
    system = System()
    instructor = Instructor()

    # Creating Models ##
    models = OBJECT()

    models.f_sm = model_(f_sm_key, system)
    models.f_cons = model_(f_cons_key, system)
    models.f_im = model_(f_im_key, system)

    # Creating Simulation object, running simulation and plotting experiments##
    # tree/DP Interest Model
    directory = 'RndExperiments'
    file_prefix = directory + '/Vowels_Tree_' + now

    evaluation_sim = SM_ModelEvaluation(system,
                                        models.f_sm, comp_func=comp_func,
                                        file_prefix=file_prefix)

    evaluation_sim.load_eval_dataset('../../Systems/datasets/german_dataset_3.h5')

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
    val_data = evaluation_sim.evaluate(saveData=True)
    error_ = np.linalg.norm(val_data.sensor_goal.data.as_matrix() -
                            val_data.sensor.data.as_matrix(), axis=1)
    print("Mean evaluation error is {} (max: {}, min: {})".format(np.mean(error_),
                                                                  np.max(error_),
                                                                  np.min(error_)))

    from diva_plot_results import show_results

    show_results(system, simulation, val_data)